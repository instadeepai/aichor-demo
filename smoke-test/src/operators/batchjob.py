import os
from src.utils.tensorboard import dummy_tb_write

def batchjobop(tb_write: bool):
    job_index = os.environ.get("JOB_COMPLETION_INDEX")

    print("job index: ", job_index)

    if tb_write:
        dummy_tb_write(f"From job_index {job_index}")