import os
import logging

import pickle
from s3fs import S3FileSystem

logger = logging.getLogger(__name__)

def build_s3_client():
    s3_endpoint = os.environ['S3_ENDPOINT']
    s3_key = os.environ['AWS_ACCESS_KEY_ID']
    s3_secret_key = os.environ['AWS_SECRET_ACCESS_KEY']

    return S3FileSystem(client_kwargs={"endpoint_url": s3_endpoint},
        key=s3_key,
        secret=s3_secret_key
    )

def dump_model(model):
    s3client = build_s3_client()
    filename = os.environ["AICHOR_OUTPUT_PATH"] + "model-new.pickle"
    logger.info(f"save model to: {filename}")
    with s3client.open(filename, "wb") as f:
        pickle.dump(model, f)