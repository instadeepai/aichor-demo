import flax.linen as nn
from typing import Tuple

class MyBlock(nn.Module):
    """Very simple Block for MySuperModel"""
    features: int

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=self.features, kernel_size=(3, 3))(x)
        x = nn.BatchNorm(use_running_average=False)(x)
        x = nn.relu(x)
        x = nn.Conv(features=self.features, kernel_size=(3, 3))(x)
        x = nn.BatchNorm(use_running_average=False)(x)
        x = nn.relu(x)
        return x

class MySuperModel(nn.Module):
    """Very simple model just for test purpose."""
    features: Tuple[int, ...] = (8, 16, 32, 64)

    def setup(self):
        self.blocks = [MyBlock(f) for f in self.features]

        self.linear_1 = nn.Dense(10)

    @nn.compact
    def __call__(self, x):
        y = x
        for b in self.blocks:
            y = b(y)

        y = nn.avg_pool(y, (32, 32))
        y = y.reshape((y.shape[0], y.shape[-1]))
        y = self.linear_1(y)

        return nn.softmax(y)
