from __future__ import print_function

import os

import torch
from tensorboardX import SummaryWriter
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.optim as optim
from torchvision import datasets, transforms, models
import s3fs

WORLD_SIZE = int(os.environ.get('WORLD_SIZE', 1))
BATCH_SIZE = 64
NUM_EPOCHS = 16
LOG_INTERVAL = 64

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(5*5*50, 500)
        self.fc2 = nn.Linear(500, 100)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 5*5*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def should_distribute() -> bool:
    return dist.is_available() and WORLD_SIZE > 1

def train(model, device, train_loader, optimizer, epoch, writer):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = F.log_softmax(model(data), dim=-1)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tloss={:.4f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()), flush=True)
            niter = epoch * len(train_loader) + batch_idx
            writer.add_scalar('train/loss', loss.item(), niter)

def test(model, device, test_loader, writer, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = F.log_softmax(model(data), dim=-1)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\naccuracy={:.4f}\n'.format(float(correct) / len(test_loader.dataset)), flush=True)
    writer.add_scalar('test/accuracy', float(correct) / len(test_loader.dataset), epoch)
    writer.add_scalar('test/loss', test_loss, epoch)

def main():
    writer = SummaryWriter(os.environ.get("AICHOR_TENSORBOARD_PATH", "./runs"))
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print('Using CUDA')
    device = torch.device("cuda" if use_cuda else "cpu")

    if should_distribute():
        if use_cuda and dist.is_nccl_available():
            backend = dist.Backend.NCCL
        elif dist.is_mpi_available():
            backend = dist.Backend.MPI
        else:
            backend = dist.Backend.GLOO
        print("distribution enabled with " + backend)
        dist.init_process_group(backend=backend)
    
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100("./data", "train", download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])),
        batch_size=BATCH_SIZE, shuffle=True, **kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100("./data", "test", download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])),
        batch_size=BATCH_SIZE, shuffle=False, **kwargs
    )

    model = models.efficientnet_b0(num_classes=100).to(device)

    if dist.is_available() and dist.is_initialized():
        model = nn.parallel.DistributedDataParallel(model)

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    
    for epoch in range(1, NUM_EPOCHS + 1):
        train(model, device, train_loader, optimizer, epoch, writer)
        test(model, device, test_loader, writer, epoch)

    writer.flush()
    writer.close()
    
    s3_endpoint = os.environ['S3_ENDPOINT']
    s3_key = os.environ['AWS_ACCESS_KEY_ID']
    s3_secret_key = os.environ['AWS_SECRET_ACCESS_KEY']
    s3 = s3fs.S3FileSystem(client_kwargs={"endpoint_url": s3_endpoint},
        key=s3_key,
        secret=s3_secret_key
    )
    with s3.open(os.environ["AICHOR_OUTPUT_PATH"] + "model_final.pt", mode="wb") as f:
        torch.save(model.state_dict(), f)


if __name__ == '__main__':
    main()
    print("DONE")
