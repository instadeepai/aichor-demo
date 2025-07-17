import json
import os
import time

def get_index() -> int:
    jobset_type_index = os.environ.get("JOB_INDEX") # here JOB_GLOBAL_INDEX could NOT be used in this scenario
    if jobset_type_index == None:
        return 0
def get_type() -> str:
    jobset_type = os.environ.get("REPLICATED_JOB_NAME")
    if jobset_type == None:
        return 0

if __name__ == '__main__':
    type_of_job = get_type()
    rank = get_index()
    if type_of_job == "worker-heavy":
      print("actions 1 to 300")
    elif type_of_job == "worker":
      if rank == 0:
          print("actions 301 to 400")
          time.sleep(30)
      elif rank == 1:
          print("actions 401 to 500")
          time.sleep(40)
    #   .
    #   .
    #   .
    # rest of if conditions

      else:
          print(f"actions 901 to 1000")
          time.sleep(10)