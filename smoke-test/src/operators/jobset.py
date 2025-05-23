import os
from src.utils.tensorboard import dummy_tb_write

def jobsetop(tb_write):
    print_jobset_env(tb_write)

def print_jobset_env(tb_write):
    print("job completion index: ", os.environ.get("JOB_COMPLETION_INDEX"))
    if tb_write:
        dummy_tb_write(f"From job_index {os.environ.get("JOB_COMPLETION_INDEX")}")