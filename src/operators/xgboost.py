import os
from src.utils.tensorboard import dummy_tb_write

def xgboostop(tb_write: bool):
    rank = os.environ.get("RANK")

    print("MASTER_PORT: ", os.environ.get("MASTER_PORT"))
    print("MASTER_ADDR: ", os.environ.get("MASTER_ADDR"))
    print("WORLD_SIZE: ", os.environ.get("WORLD_SIZE"))
    print("RANK: ", rank)
    print("WORKER_PORT: ", os.environ.get("WORKER_PORT"))
    print("WORKER_ADDRS: ", os.environ.get("WORKER_ADDRS"))

    if tb_write:
        dummy_tb_write(f"from rank {rank}")