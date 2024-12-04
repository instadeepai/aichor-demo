### Demo project for demonstrating platform pipeline

### ⚠️ Warnings:
- This project it not indented to be good from the machine learning point of view. It it just intended to show how to use kuberay.

## With GPU

Make sure to:
- Have the Dockerfile based on cuda image (rayproject/ray:2.36.1-py39-gpu works).
- Set `num_gpus` to `1` in Tuner config in `main.py`
- Add a GPU in the `Head` in `manifest.yaml`

## Without GPU

Make sure to:
- Set `num_gpus` to `0` in Tuner config in `main.py`
- Remove or comment GPU part in `manifest.yaml`

Not required but you can also change the base image to reduce its size (using rayproject/ray:2.36.1-cpu for example).

## Reduce the size of the experiment

With the current config the experiment will attempt to spawn 32 environments across all of the experiment's workers.
The total number of CPUs in the Workers should be >= the number of environments.

To reduce the size of the experiment you will need to
- Reduce number and size of Worker.
- Reduce `num_workers` accordingly.

Also you might want to reduce the stop condition at `timesteps_total`.