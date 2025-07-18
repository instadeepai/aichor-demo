You might also have a case where you have large slices of actions that are similar but want these slices to be processed separately, for example actions 0 to 200 are similar but different from actions 200 to 400 and so on. but you don't want these sections of 200 actions to all be proccessed by one single container (scenario 3):

```
worker 0: 1   ->  200  ->  worker 0-0: 1    ->    10
                           worker 0-1: 11   ->    20
                           worker 0-2: 21   ->    30
                           ...
worker 1: 201 ->  400  ->  worker 1-0: 201  ->    210
                           worker 1-1: 211  ->    220
                           worker 1-2: 221  ->    230
                           ...
...

```

For this scenario we need the 2D container array that jobset provides and we can acheive that using the `completions` in the manifest like this (please note that `parallelisms` must also be set to the same number as `completions` if you want all the pods to run at the same time):
***Please note that `JOB_INDEX` and `JOB_GLOBAL_INDEX` only scale on the count number, meaning that all the completion in each job replica (i.e. the ones like worker 0-0, worker 0-1 and worker 0-3 and ... all share the same `JOB_INDEX`)

```yaml
kind: AIchorManifest
apiVersion: 0.2.1

builder:
  image: jobset-multi-jobs
  dockerfile: ./Dockerfile
  context: .

spec:
  operator: jobset
  image: jobset-multi-jobs
  command: "python -u src/main.10-10-10.py"

  types:
    worker:
      count: 10
      completions: 10   # optional, defaults to 1
      parallelisms: 10  # optional, defaults to 1
      resources:
        cpus: 2
        ramRatio: 2
        shmSizeGB: 0

```
now the code could look like this (you can be more imaginative with using the env variables of course so you don't have to write 100 different if cases)

```python
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
```

Below you can see the pods in this scenario succeed

![alt text](/screenshots/scenario3-pods.png)

and now the logs of a few of the pods

worker-0-1:

![alt text](/screenshots/scenario3-0-1.png)

worker-1-1:

![alt text](/screenshots/scenario3-1-1.png)
