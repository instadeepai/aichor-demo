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

def get_index() -> int:
    jobset_type_index = os.environ.get("JOB_INDEX") # here JOB_GLOBAL_INDEX could also be used in this scenario
    if jobset_type_index == None:
        return 0
    return jobset_type_index

def get_completion() -> int:
    jobset_completion_index = os.environ.get("JOB_COMPLETION_INDEX")
    if jobset_completion_index == None:
        return 0
    return jobset_completion_index

# Now we can use the variables to define different tasks
if __name__ == '__main__':
    print(f"here are the rest of the jobset variables in this container:")
    jobsetop()

    rank = get_completion()
    index = get_index()
    if index == "0":
      if rank == "0":
        print("actions 1 to 10")
        time.sleep(10)
      elif rank =="1":
        print("actions 11 to 20")
        time.sleep(10)
    # .
    # .
    # .
    if index == "1":
      if rank == "0":
        print("actions 201 to 210")
        time.sleep(10)
      elif rank =="1":
        print("actions 211 to 220")
        time.sleep(10)
    # .
    # .
    # .