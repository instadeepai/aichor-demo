import os
import shutil

from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast, AutoModelForSequenceClassification
from accelerate import Accelerator
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict

from slugify import slugify
from s3fs import S3FileSystem

from constant import AICHOR_OUTPUT_BUCKET_NAME

# Save
def save_checkpoint(accelerator: Accelerator, epoch: int, checkpoint_dir: str, s3: S3FileSystem):
    if s3 is None:
        accelerator.save_state()
    save_checkpoint_s3(accelerator=accelerator, epoch=epoch, checkpoint_dir=checkpoint_dir, s3=s3)

def save_checkpoint_s3(accelerator: Accelerator, epoch: int, checkpoint_dir: str, s3: S3FileSystem):
    output_path = f"s3://{os.environ.get(AICHOR_OUTPUT_BUCKET_NAME)}/{checkpoint_dir}/checkpoint_epoch_{epoch}"
    path = accelerator.save_state()
    if accelerator.is_main_process:
        s3.put(path, output_path, recursive=True)
        # saving a "valid" file to make sure that checkpoint was fully saved.
        with s3.open(f"{output_path}/valid", "w") as f:
            f.write("1")
            f.flush()
        print(f"Checkpoint saved at {output_path}")
    accelerator.wait_for_everyone()
    
# Load
def load_checkpoint(accelerator: Accelerator, s3: S3FileSystem, args) -> int:
    checkpoint_path = args.load_checkpoint_name

    # S3 not initialized, using local checkpoint
    epoch = 0
    if s3 is None:
        if args.load_checkpoint_name is not None:
            accelerator.load_state(checkpoint_path)
            return get_epoch_from_path(checkpoint_path)
        else:
            accelerator.load_state()
        return epoch

    # retrieve from s3
    if args.load_checkpoint_name == None:
        checkpoint_path = get_last_checkpoint_path(checkpoint_dir=args.checkpoint_dir, s3=s3)
    if checkpoint_path != None:
        return load_checkpoint_s3(accelerator=accelerator, checkpoint_path=checkpoint_path, s3=s3)
    return epoch

def load_checkpoint_s3(accelerator: Accelerator, checkpoint_path: str, s3: S3FileSystem):
    checkpoint_local_path = "tmp_checkpoint"
    if accelerator.is_local_main_process:
        print(f"Loading checkpoint from {checkpoint_path}")
        s3.get(checkpoint_path, checkpoint_local_path, recursive=True)
    accelerator.wait_for_everyone()
    accelerator.load_state(checkpoint_local_path)
    accelerator.wait_for_everyone() # wait for every process to finish loading
    if accelerator.is_local_main_process:
        shutil.rmtree(checkpoint_local_path)

    # get epoch from checkpoint name
    return get_epoch_from_path(checkpoint_path)

def get_epoch_from_path(path: str) -> int:
    checkpoint_name = path.split('/')[-1]
    return int(checkpoint_name.replace("checkpoint_epoch_", "")) + 1

def get_last_checkpoint_path(checkpoint_dir: str, s3: S3FileSystem):
    checkpoint_dir_full = f"s3://{os.environ.get(AICHOR_OUTPUT_BUCKET_NAME)}/{checkpoint_dir}"
    try:
        dirs = s3.listdir(checkpoint_dir_full)
    except FileNotFoundError:
        print(f"Couldn't find checkpoint at {checkpoint_dir_full}, starting from epoch 0")
        return None
    sorted_dirs = sorted(dirs, key=lambda x: int(x['Key'].split('checkpoint_epoch_')[-1]), reverse=True)
    for directory in sorted_dirs:
        directory_key = directory['Key']
        files_in_dir = s3.listdir(f"s3://{directory_key}")
        for file in files_in_dir:
            if file['Key'].endswith('/valid'):
                return f"s3://{directory['Key']}"

    return None