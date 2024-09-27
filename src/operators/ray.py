import os
from src.utils.tensorboard import dummy_tb_write

def rayop(tb_write: bool):
    # importing here because in case of non-ray experiment, pkg isn't installed
    import ray

    ray.init(address=os.environ.get("RAY_SERVER", "auto"))

    nodes = ray.nodes()
    print("connected nodes: ", nodes)

    if tb_write:
        dummy_tb_write(None)