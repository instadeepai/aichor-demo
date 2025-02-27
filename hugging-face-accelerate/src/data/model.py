import os
import shutil

from transformers import AutoModelForSequenceClassification
from accelerate import Accelerator

from slugify import slugify
from s3fs import S3FileSystem

from constant import HF_TOKEN, AICHOR_INPUT_PATH

def get_model(accelerator: Accelerator, s3: S3FileSystem, model_name: str):
    if s3 is None:
        return get_model_from(model_name)
    return get_model_s3(accelerator=accelerator, s3=s3, model_name=model_name)

def get_model_from(load_from: str):
    return AutoModelForSequenceClassification.from_pretrained(load_from, token=os.environ.get(HF_TOKEN))

def get_model_s3(accelerator: Accelerator, s3: S3FileSystem, model_name: str):
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
    model = AutoModelForSequenceClassification.from_pretrained(load_from, token=os.environ.get(HF_TOKEN))

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