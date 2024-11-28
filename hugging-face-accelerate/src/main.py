import os
import time
import argparse

import s3fs

import evaluate
import torch
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup, set_seed

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration

from utils import get_dataset, get_tokenizer, get_model, save_final_model, TENSORBOARD_PATH, AWS_ENDPOINT_URL

SEED = 42

LOCAL_PROJECT_DIR: str = "logs"

def training_function(args: argparse.Namespace):
    # Initialize accelerator
    accelerator = Accelerator(
        cpu=False,
        mixed_precision=args.mixed_precision,
        project_dir=LOCAL_PROJECT_DIR,
        project_config=ProjectConfiguration(
            logging_dir=os.environ.get(TENSORBOARD_PATH),
        ),
        log_with="tensorboard",
    )

    # We need to initialize the trackers we use, and also store our configuration
    run = os.path.split(__file__)[-1].split(".")[0]
    accelerator.init_trackers(run, {"lr": args.learning_rate, "num_epochs": args.num_epochs, "seed": SEED, "batch_size": args.batch_size})

    s3 = s3fs.S3FileSystem(endpoint_url=os.environ.get(AWS_ENDPOINT_URL))

    tokenizer = get_tokenizer(accelerator=accelerator, s3=s3, model_name=args.model)
    datasets = get_dataset(accelerator=accelerator, s3=s3)
    metric = evaluate.load("glue", "mrpc")

    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        # max_length=None => use the model max length (it's actually the default)
        outputs = tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, max_length=None)
        return outputs

    # Apply the method we just defined to all the examples in all the splits of the dataset
    # starting with the main process first:
    with accelerator.main_process_first():
        tokenized_datasets = datasets.map(
            tokenize_function,
            batched=True,
            remove_columns=["idx", "sentence1", "sentence2"],
        )

    # We also rename the 'label' column to 'labels' which is the expected name for labels by the models of the
    # transformers library
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

    def collate_fn(examples):
        # When using mixed precision we want round multiples of 8/16
        if accelerator.mixed_precision == "fp8":
            pad_to_multiple_of = 16
        elif accelerator.mixed_precision != "no":
            pad_to_multiple_of = 8
        else:
            pad_to_multiple_of = None

        return tokenizer.pad(
            examples,
            padding="longest",
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors="pt",
        )

    # Instantiate dataloaders.
    train_dataloader = DataLoader(
        tokenized_datasets["train"], shuffle=True, collate_fn=collate_fn, batch_size=args.batch_size
    )
    eval_dataloader = DataLoader(
        tokenized_datasets["validation"], shuffle=False, collate_fn=collate_fn, batch_size=args.batch_size
    )

    set_seed(SEED)

    # Instantiate the model (we build the model here so that the seed also control new weights initialization)
    model = get_model(accelerator=accelerator, s3=s3, model_name=args.model)
    model.config.pad_token_id = model.config.eos_token_id

    # We could avoid this line since the accelerator is set with `device_placement=True` (default value).
    # Note that if you are placing tensors on devices manually, this line absolutely needs to be before the optimizer
    # creation otherwise training will not work on TPU (`accelerate` will kindly throw an error to make us aware of that).
    model = model.to(accelerator.device)

    # Instantiate optimizer
    optimizer = AdamW(params=model.parameters(), lr=args.learning_rate)

    # Instantiate scheduler
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=(len(train_dataloader) * args.num_epochs),
    )

    # Prepare everything
    # There is no specific order to remember, we just need to unpack the objects in the same order we gave them to the
    # prepare method.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # We need to keep track of how many total steps we have iterated over
    overall_step = 0

    # Now we train the model
    if accelerator.is_main_process:
        print("Start training")
        start_time = time.time()

    for epoch in range(0, args.num_epochs):
        model.train()
        total_loss = 0
        for step, batch in enumerate(train_dataloader):
            # We could avoid this line since we set the accelerator with `device_placement=True`.
            batch.to(accelerator.device)
            outputs = model(**batch)
            loss = outputs.loss
            loss = loss
            # We keep track of the loss at each epoch
            total_loss += loss.detach().float()
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            overall_step += 1

        model.eval()
        for step, batch in enumerate(eval_dataloader):
            # We could avoid this line since we set the accelerator with `device_placement=True`.
            batch.to(accelerator.device)
            with torch.no_grad():
                outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            predictions, references = accelerator.gather_for_metrics((predictions, batch["labels"]))
            metric.add_batch(
                predictions=predictions,
                references=references,
            )

        eval_metric = metric.compute()
        # Use accelerator.print to print only on the main process.
        accelerator.print(f"epoch {epoch}:", eval_metric)
        accelerator.log(
            {
                "accuracy": eval_metric["accuracy"],
                "f1": eval_metric["f1"],
                "train_loss": total_loss.item() / len(train_dataloader),
                "epoch": epoch,
            },
            step=epoch,
        )

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        # display execution time
        execution_time = time.time() - start_time
        minutes = int(execution_time // 60)
        seconds = int(execution_time % 60)
        print(f"Training is done. Execution time: {minutes}m{seconds}s.")

    save_final_model(accelerator=accelerator, model=model, s3=s3)
    accelerator.end_training()
    accelerator.clear()

def main():
    parser = argparse.ArgumentParser(description="HuggingFace Accelerate training example.")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16", "fp8"],
        help="Whether to use mixed precision. Choose"
        "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
        "and an Nvidia Ampere GPU.",
    )
    parser.add_argument("--model", type=str, default="TinyLlama/TinyLlama_v1.1")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size. Adjust depending on GPU memory available")
    parser.add_argument("--num_epochs", type=int, default=12)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    args = parser.parse_args()

    training_function(args)

if __name__ == "__main__":
    main()