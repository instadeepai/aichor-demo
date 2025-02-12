# KubeRay Multi Host TPU Demo

This demo project is inpired from [here](https://github.com/GoogleCloudPlatform/ai-on-gke/blob/main/ray-on-gke/examples/notebooks/jax-tpu.ipynb).

This example project is expected demonstrate how to run a multi host TPU experiment in AIchor.
Check the `manifests/manifest.yaml` file to learn more about the setup.

## Dependencies

This example has the minimum requirements to run a multi host TPU experiment with Ray and Jax. See `build/requiements.txt`.

## Script

The script in `main.py` will spawn 4 actors (1 actor by TPU VM) each requesting 4 TPU chips. The job should print:
```
starting 4 remote functions
['Index [0]: Global TPU Count=16', 'Index [1]: Global TPU Count=16', 'Index [2]: Global TPU Count=16', 'Index [3]: Global TPU Count=16']
Done! exiting
```

## Learn more about TPUs, topologies, ...

- AIchor documention soon
- [topology diagrams](https://cloud.google.com/kubernetes-engine/docs/concepts/tpus#type-node-pool)
- [TPU zones and region](https://cloud.google.com/tpu/docs/regions-zones#europe)