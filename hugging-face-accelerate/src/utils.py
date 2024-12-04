import os
import shutil

from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast, AutoModelForSequenceClassification
from accelerate import Accelerator
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict

from slugify import slugify
from s3fs import S3FileSystem

AWS_ENDPOINT_URL: str = "AWS_ENDPOINT_URL"
AICHOR_INPUT_PATH: str = "AICHOR_INPUT_PATH"
AICHOR_OUTPUT_PATH: str = "AICHOR_OUTPUT_PATH"
TENSORBOARD_PATH: str = "AICHOR_TENSORBOARD_PATH"

HF_TOKEN: str = "HF_TOKEN"

def get_tokenizer(accelerator: Accelerator, s3: S3FileSystem, model_name: str) -> (PreTrainedTokenizer | PreTrainedTokenizerFast):
    model_slug = f"{slugify(model_name)}-tokenizer"
    local_path = model_slug
    load_from = ""
    should_save_to_s3 = False
    s3_path = os.environ.get(AICHOR_INPUT_PATH) + model_slug

    # download model from S3 if present
    if s3.exists(s3_path):
        # only main process should download from s3
        if accelerator.is_local_main_process:
            s3.get(s3_path, local_path, recursive=True)
        load_from = local_path
    else: # download from HuggingFace
        load_from = model_name
        should_save_to_s3 = True

    accelerator.wait_for_everyone() # wait for local main process to finish downloading the tokenizer from s3
    # tokenizer = AutoTokenizer.from_pretrained(load_from, token=os.environ.get(HF_TOKEN))
    tokenizer = AutoTokenizer.from_pretrained(load_from)

    accelerator.wait_for_everyone() # wait for all tokenizer loaded on all processes

    # cleanup downloaded model from S3
    if (not should_save_to_s3) and accelerator.is_local_main_process:
        shutil.rmtree(local_path)

    # save downloaded model from HuggingFace to S3
    if should_save_to_s3 and accelerator.is_main_process:
        tokenizer.save_pretrained(local_path)
        s3.put(local_path, s3_path, recursive=True)
        shutil.rmtree(local_path)

    accelerator.wait_for_everyone() # wait cleanup tasks to end
    return tokenizer

def get_model(accelerator: Accelerator, s3: S3FileSystem, model_name: str):
    model_slug = f"{slugify(model_name)}-model"
    local_path = model_slug
    load_from = ""
    should_save_to_s3 = False
    s3_path = os.environ.get(AICHOR_INPUT_PATH) + model_slug

    # download model from S3 if present
    if s3.exists(s3_path):
        # only main process should download from s3
        if accelerator.is_local_main_process:
            s3.get(s3_path, local_path, recursive=True)
        load_from = local_path
    else: # download from HuggingFace
        load_from = model_name
        should_save_to_s3 = True

    accelerator.wait_for_everyone() # wait for local main process to finish downloading the tokenizer from s3
    # model = AutoModelForSequenceClassification.from_pretrained(load_from, token=os.environ.get(HF_TOKEN))
    model = AutoModelForSequenceClassification.from_pretrained(load_from)

    # cleanup downloaded model from S3 from local main process
    if (not should_save_to_s3) and accelerator.is_local_main_process:
        shutil.rmtree(local_path)

    accelerator.wait_for_everyone() # wait for all model loaded on all processes
    if should_save_to_s3 and accelerator.is_main_process:
        model.save_pretrained(local_path)
        s3.put(local_path, s3_path, recursive=True)
        shutil.rmtree(local_path)

    accelerator.wait_for_everyone() # wait for local main process to finish cleaning directory
    return model

def get_dataset(accelerator: Accelerator, s3: S3FileSystem) -> (Dataset | DatasetDict):
    s3_path = os.environ.get(AICHOR_INPUT_PATH) + "glue-mrpc"
    dataset: Dataset | DatasetDict

    if s3.exists(s3_path):
        dataset = load_from_disk(s3_path) # accepts S3 paths
    else:
        dataset = load_dataset("glue", "mrpc")
        if accelerator.is_main_process:
            dataset.save_to_disk(s3_path) # accepts S3 paths
        accelerator.wait_for_everyone()

    return dataset

def save_final_model(accelerator: Accelerator, model, s3: S3FileSystem):
    local_path = "final_model"
    output_path = os.environ.get(AICHOR_OUTPUT_PATH)

    if accelerator.is_main_process:
        print(f"Saving trained model at: {output_path} from main process")
        accelerator.save_model(model, local_path)
        s3.put(local_path, output_path, recursive=True)
        shutil.rmtree(local_path)
        print("Uploaded")

    accelerator.wait_for_everyone()