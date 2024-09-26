import os

def xgboostop():
    print_xgboost_env()

def print_xgboost_env():
    print("MASTER_PORT: ", os.environ.get("MASTER_PORT"))
    print("MASTER_ADDR: ", os.environ.get("MASTER_ADDR"))
    print("WORLD_SIZE: ", os.environ.get("WORLD_SIZE"))
    print("RANK: ", os.environ.get("RANK"))
    print("WORKER_PORT: ", os.environ.get("WORKER_PORT"))
    print("WORKER_ADDRS: ", os.environ.get("WORKER_ADDRS"))