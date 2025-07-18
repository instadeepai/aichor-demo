# multi-jobs-demo

This is a demo project for a multijobs experiment, it will contain the JobSet operator

## Goal

The goal of this demo project is to concurrently run multiple jobs under a single AIchor experiment. We will provide an example and several ways of approaching it.

Example:
You need to process a dataset of size 1000 with completely independent actions with 200 CPUs and 400GB of memory, that's a lot of resources for a single container, your experiment might stay `pending` forever. The solution here is to divide the workload by the number of workers:

By controling the number of workers you can decide how your job will be divided, for example you can have 1000 different actions that pretty much take the same amount of resources and you don't really need any specific sort of distribution so you can set up 10 workers all with equal loads like this (scenario 1):

```
worker 0: 1   ->  100
worker 1: 101 ->  200
worker 2: 201 ->  300
...
worker 9: 901 -> 1000
```

In another case you might want to distribute the actions in a more uneven way because of a specific requirement for some of the actions, for example the first 300 actions must be done together and need more CPUs because they have a heavier load, but the rest can be divided equally (scenario 2):

```
worker-heavy  0: 1   ->  300
worker        0: 301 ->  400
worker        1: 401 ->  500
...
worker        6: 901 -> 1000
```

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

Now all the cases mean that all your containers will want to start at the same time and this could mean a very big job all at once, but you might not have an issue with the pods getting deployed a few at a time. This will mean that the job might take more time but since it's not all at once, it has a much higher probability of actually being scheduled on the cluster, especially during times of high load when the cluster might be too full to accomodate your entire job all at once. For example containers being deployed 2 at a time (scenario 4):
```
                                  start of exp          after the two containers finish ...
worker 0: 1   ->  200  ->  worker 0-0: 1    ->    10  |  worker 0-2: 21   ->    30      ...
                           worker 0-1: 11   ->    20  |  worker 0-3: 31   ->    40      ...
                           ...
worker 1: 201 ->  400  ->  worker 1-0: 201  ->    210 |  worker 1-2: 221  ->    230     ...
                           worker 1-1: 211  ->    220 |  worker 1-3: 231  ->    240     ...
                           ...
...

```

## How

Here the implementation of every scenario provided above will be shown, but first, explaining the manifest for the JobSet operator in the most simple case.

###  First let's inspect the `manifest.yaml`

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
  command: "python -u src/main.py"

  types:
    worker:
      count: 5
      resources:
        cpus: 1
        ramRatio: 1
        shmSizeGB: 0
```

Requirements:
- Use the `jobset` operator.
- Set multiple Worker (defined at `spec.types.worker.count: 5`)

This manifest will create an experiment with 5 containers, each one of them will have the same resource requirements (1 CPU, 1G of memory). Each container will execute the command: `spec.command: "python -u src/main.py"`

### Inside the code

By selecting multiple workers with the jobset operator, it will schedule 5 containers and will, concurrently, execute `python main.py`. Workers should read the `JOB_GLOBAL_INDEX` environment variable to know what their rank is.
Please note that as long as we don't have different types (i.e. we have only worker or only master) `JOB_INDEX` will be equal to `JOB_GLOBAL_INDEX`
This variable is injected by AIchor components:

```python
def get_rank() -> int:
    jobset_global_index = os.environ.get("JOB_GLOBAL_INDEX") # here JOB_INDEX could also be used in this scenario
    if jobset_global_index == None:
        return 0

```

Then when your worker is aware of its rank you can assign them different tasks:
```python
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
```

In this example:
- Container with rank `0` will print `hey i am rank 0, sleeping for 30s` and then sleep for 30s before exiting.
- Container with rank `1` will print `hey i am rank 1, sleeping for 40s` and then sleep for 40s before exiting.
- Container with ranks `2`, `3` and `4` will print `hey i am rank {rank} sleeping for 10s`  and then sleep for 10s before exiting.

Now what you saw above was the most basic setup, now we can go through the scenarios described earlier and how we can implement them:

Scenario 1:
This would be very similar to the simple case:

[Scenario 1](./scenarios/scenario-1.md)

Scenario 2:

For this since we need separate specifications in terms of the cpu requirements we need different types:

[Scenario 2](./scenarios/scenario-2.md)

Scenario 3:


for this scenario we need the 2D container array that jobset provides and we can acheive that using the `completions` in the manifest like this (please note that `parallelisms` must also be set to the same number as `completions` if you want all the pods to run at the same time):
***Please note that `JOB_INDEX` and `JOB_GLOBAL_INDEX` only scale on the count number, meaning that all the completion in each job replica (i.e. the ones like worker 0-0, worker 0-1 and worker 0-3 and ...) all share the same `JOB_INDEX`

[Scenario 3](./scenarios/scenario-3.md)

Scenario 4:

For this scenario we must operate using the `parallelisms` and set that to the number of pods we want running at the same time:

[Scenario 4](./scenarios/scenario-4.md)
