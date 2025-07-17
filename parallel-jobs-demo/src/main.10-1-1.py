import json
import os
import time

def get_rank() -> int:
    jobset_global_index = os.environ.get("JOB_GLOBAL_INDEX") # here JOB_INDEX could also be used in this scenario
    if jobset_global_index == None:
        return 0

if __name__ == '__main__':
    rank = get_rank()

    if rank == 0:
        print("actions 1 to 100")
        time.sleep(30)
    elif rank == 1:
        print("actions 101 to 200")
        time.sleep(40)
    # .
    # .
    # .
    # rest of the if conditions
    else:
        print(f"actions 901 to 1000")
        time.sleep(10)