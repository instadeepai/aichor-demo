import os

from accelerate import Accelerator
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict

from s3fs import S3FileSystem

from constant import AICHOR_INPUT_PATH

def get_dataset(accelerator: Accelerator, s3: S3FileSystem) -> (Dataset | DatasetDict):
    if s3 is None:
        return get_dataset_from_remote()
    return get_dataset_s3(accelerator=accelerator, s3=s3)

def get_dataset_from_remote() -> (Dataset | DatasetDict):
    return load_dataset("glue", "mrpc")

def get_dataset_s3(accelerator: Accelerator, s3: S3FileSystem) -> (Dataset | DatasetDict):
    s3_path = os.environ.get(AICHOR_INPUT_PATH) + "glue-mrpc"
    dataset: Dataset | DatasetDict

    if s3.exists(s3_path):
        dataset = load_from_disk(s3_path) # accepts S3 paths
    else:
        dataset = get_dataset_from_remote()
        if accelerator.is_main_process:
            dataset.save_to_disk(s3_path) # accepts S3 paths
        accelerator.wait_for_everyone()

    return dataset