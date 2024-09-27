import os
from src.utils.tensorboard import dummy_tb_write

def tfop(tb_write: bool):
    tf_config_raw = os.environ.get("TF_CONFIG")
    print("tf_config: ", tf_config_raw)

    if tf_config_raw == None:
        print("tf_config is None because worker count = 1")

    if tb_write:
        dummy_tb_write(None)