import os
import time

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
    jobset_global_index = os.environ.get("JOB_GLOBAL_INDEX") # here JOB_INDEX could also be used in this scenario
    if jobset_global_index == None:
        return 0
    return jobset_global_index


# Now we can use the ranks to define different tasks
if __name__ == '__main__':

    print(f"here are the rest of the jobset variables in this container:")
    jobsetop()

    rank = get_rank()

    print(f"here is the rank of this worker: {rank}")

    print("actions for this worker")
    if rank == "0":
        print("actions 1 to 100")
        time.sleep(30)
    elif rank == "1":
        print("actions 101 to 200")
        time.sleep(40)
    # .
    # .
    # .
    # rest of the if conditions
    else:
        print("actions 901 to 1000")
        time.sleep(10)