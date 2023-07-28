import os

def tfop():
    print_tf_env()

def print_tf_env():
    tf_config_raw = os.environ.get("TF_CONFIG")
    print("tf_config: ", tf_config_raw)

    if tf_config_raw == None:
        print("tf_config is None because worker count = 1")