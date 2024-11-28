# AIChor Jax example

## Info

This repository shows how to use the jax operator on aichor to benefit from multiple host on jax.
This repository trains a model with SPMD(Single Srogram Multiple Data) strategy.

### ⚠️ Warnings:
- This project donwloads the dataset from the internet at every run to make it self contained. Don't do that, you should put your dataset in AICHOR provided S3 bucket.
- This project it not indented to be good from the machine learning point of view. It it just intended to show how to use jax distributed.

## Running with AIChor
You can tweak the `manifest.yaml` file at the repository root