# AIchor Pytorch demo

This is a demo project for the PyTorch operator on the AIchor platform

### ⚠️ Warnings:
- This project donwloads the dataset from the internet at every run to make it self contained. Don't do that, you should put your dataset in AICHOR provided S3 bucket.
- This project it not indented to be good from the machine learning point of view. It it just intended to show how to use pytorch distributed.

## Run localy
```bash
python -m virtualenv venv # create a virtual env
source venv/bin/activate # you can source other file depending on your shell
pip install -r requirements.txt # install the dependencies 
python src/main.py
```
When running on local, it only has 1 process with the device on the machine

## Running with AIchor
You can tweak the `manifest.yaml` file at the repository root

## Environment variables injected by the operator

Environment variables are injected to setup the distribution between the different containers.

```bash
MASTER_PORT:       23456
MASTER_ADDR:       pytorch-dist-cifar-master-0
WORLD_SIZE:        3 # number of container
RANK:              1 # rank of container (from 0 to $WORLD_SIZE - 1)
```

# Notes

The operator that manages PyTorch experiments is the Kubeflow Training operator
- [operator docs](https://www.kubeflow.org/docs/components/training/pytorch/)
- [pytorch docs](https://pytorch.org/tutorials/beginner/dist_overview.html)
- [pytorch docs about communication](https://pytorch.org/docs/stable/distributed.html)
- [examples](https://github.com/kubeflow/training-operator/tree/master/examples/pytorch)

This project is based on [this](https://github.com/kubeflow/training-operator/tree/master/examples/pytorch/mnist) example from the official Kubeflow training operator repo even if most of the code have been removed / rewritten.