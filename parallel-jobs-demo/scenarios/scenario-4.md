All the previous cases mean that all your containers will want to start at the same time and this could mean a very big job all at once, but you might not have an issue with the pods getting deployed a few at a time. This will mean that the job might take more time but since it's not all at once, it has a much higher probability of actually being scheduled on the cluster, especially during times of high load when the cluster might be too full to accomodate your entire job all at once. For example containers being deployed 2 at a time (scenario 4):
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

For this scenario we must operate using the `parallelisms` and set that to the number of pods we want running at the same time:

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
  command: "python -u src/main.10-2-10.py"

  types:
    worker:
      count: 10
      completions: 10   # optional, defaults to 1
      parallelisms: 2   # optional, defaults to 1
      resources:
        cpus: 2
        ramRatio: 2
        shmSizeGB: 0

```
The code for this scenario would be identical to scenario 3 and the resulting containers and logs will also be the same, just that they will be deployed 2 at a time per replica
