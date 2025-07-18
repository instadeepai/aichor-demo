import time
import os

def jobsetop():
    job_completion_index = os.environ.get("JOB_COMPLETION_INDEX")
    job_index = os.environ.get("JOB_INDEX")
    global_replicas = os.environ.get("GLOBAL_REPLICAS")
    job_global_index = os.environ.get("JOB_GLOBAL_INDEX")
    replicated_job_name = os.environ.get("REPLICATED_JOB_NAME")
    replicated_job_replicas = os.environ.get("REPLICATED_JOB_REPLICAS")

    print("JOB_COMPLETION_INDEX ", job_completion_index)
    print("JOB_INDEX ", job_index)
    print("GLOBAL_REPLICAS ", global_replicas)
    print("JOB_GLOBAL_INDEX ", job_global_index)
    print("REPLICATED_JOB_NAME ", replicated_job_name)
    print("REPLICATED_JOB_REPLICAS ", replicated_job_replicas)

def get_rank() -> int:
    jobset_global_index = os.environ.get("JOB_INDEX") # here JOB_GLOBAL_INDEX can NOT be used in this scenario
    if jobset_global_index == None:
        return 0
    return jobset_global_index

def get_type() -> str:
    jobset_type = os.environ.get("REPLICATED_JOB_NAME")
    if jobset_type == None:
        return 0
    return jobset_type

# Now we can use the variables to define different tasks
if __name__ == '__main__':
    print(f"here are the rest of the jobset variables in this container:")
    jobsetop()

    type_of_job = get_type()
    rank = get_rank()
    if type_of_job == "worker-heavy":
      print("actions 1 to 300")
    elif type_of_job == "worker":
      if rank == "0":
          print("actions 301 to 400")
          time.sleep(30)
      elif rank == "1":
          print("actions 401 to 500")
          time.sleep(40)
    #   .
    #   .
    #   .

      else:
          print(f"actions 901 to 1000")
          time.sleep(10)