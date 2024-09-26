import os

def pytorchop():
    print_pytorch_env()

def print_pytorch_env():
    print("MASTER_PORT: ", os.environ.get("MASTER_PORT"))
    print("MASTER_ADDR: ", os.environ.get("MASTER_ADDR"))
    print("WORLD_SIZE: ", os.environ.get("WORLD_SIZE"))
    print("RANK: ", os.environ.get("RANK"))