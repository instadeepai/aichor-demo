# XGBOOST Demo

This is a demo project for the XGBoost operator on the AIChor platform

### ⚠️ Warnings:
- This project it not indented to be good from the machine learning point of view. It it just intended to show how to use xboost operator.

# Using it as a base for other project

You can easly reuse the project for your own needs:
## Load you data:
rewrite the `read_train_data` function in `src/train_data.py` to load your data

## Customize XGBoost Params
In edit `xgboost_kwargs` in `main.py` to your needs. 

## Environment variables injected by the operator

Environment variables are injected to setup the distribution between the different containers.

```bash
MASTER_PORT:       9999
MASTER_ADDR:       xgboost-dist-demo-master-0
WORLD_SIZE:        3 # number of container
RANK:              1 # rank of container (from 0 to $WORLD_SIZE - 1)
WORKER_PORT:       9999
WORKER_ADDRS:      xgboost-dist-demo-worker-0,xgboost-dist-demo-worker-1 # coma separated values
```

# Notes

The operator that manages XGBoost experiments is the Kubeflow Training operator
- [docs](https://www.kubeflow.org/docs/components/training/xgboost/)
- [more docs](https://xgboost.readthedocs.io/en/stable/tutorials/kubernetes.html)
- [examples](https://github.com/kubeflow/training-operator/tree/master/examples/xgboost)

This project is based on [this](https://github.com/kubeflow/training-operator/tree/master/examples/xgboost/xgboost-dist) example from the official Kubeflow training operator repo even if most of the code have been removed / rewrote.