import json
import os
import time

def get_rank() -> int:
    tf_config_raw = os.environ.get("TF_CONFIG")
    if tf_config_raw == None:
        return 0

    tf_config = json.loads(tf_config_raw)
    return int(tf_config["task"]["index"])

if __name__ == '__main__':
    rank = get_rank()

    if rank == 0:
        print("hey i am rank 0, sleeping for 30s")
        time.sleep(30)
    elif rank == 1:
        print("hey i am rank 1, sleeping for 40s")
        time.sleep(40)
    else:
        print(f"hey i am rank {rank} sleeping for 10s")
        time.sleep(10)