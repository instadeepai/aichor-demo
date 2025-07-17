import json
import os
import time

def get_index() -> int:
    jobset_type_index = os.environ.get("JOB_INDEX") # here JOB_GLOBAL_INDEX could also be used in this scenario
    if jobset_type_index == None:
        return 0
def get_completion() -> int:
    jobset_completion_index = os.environ.get("JOB_COMPLETION_INDEX")
    if jobset_completion_index == None:
        return 0

if __name__ == '__main__':
    rank = get_completion()
    index = get_index()
    if index == 0:
      if rank == 0:
        print("actions 1 to 10")
      elif rank ==1:
        print("actions 11 to 20")
    # rest of the if conditions
    if index == 1:
      if rank == 0:
        print("actions 201 to 210")
      elif rank ==1:
        print("actions 211 to 220")
    # rest of the if conditions