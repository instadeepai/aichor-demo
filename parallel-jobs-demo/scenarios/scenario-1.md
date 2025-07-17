By controling the number of workers you can decide how your job will be divided, for example you can have 1000 different actions that pretty much take the same amount of resources and you don't really need any specific sort of distribution so you can set up 10 workers all with equal loads like this:

```
worker 0: 1   ->  100
worker 1: 101 ->  200
worker 2: 201 ->  300
...
worker 9: 901 -> 1000
```

This would be very similar to the simple case:

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
      count: 10
      resources:
        cpus: 20
        ramRatio: 2
        shmSizeGB: 0
```
and then the following in the code:

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
        print("actions 1 to 100")
        time.sleep(30)
    elif rank == 1:
        print("actions 101 to 200")
        time.sleep(40)
    .
    .
    .

    else:
        print(f"actions 901 to 1000")
        time.sleep(10)
```