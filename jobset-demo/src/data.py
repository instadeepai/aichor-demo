import jax.numpy as jnp
import numpy as np
import flax.jax_utils as jax_utils
import tensorflow as tf
import tensorflow_datasets as tfds
import functools
import jax

def get_dataset(batch_size: int, split: str="train", download=True):
    dsb = tfds.image_classification.Cifar10()
    if download:
        dsb.download_and_prepare()
    ds = dsb.as_dataset(split, shuffle_files=True)

    return ds.batch(batch_size, drop_remainder=True)

def _prepare_data(x, local_device_count):
    x = x._numpy()

    return x.reshape((local_device_count, -1) + x.shape[1:])

def prepare_data(batch, local_device_count):
    batch = {
        "image": batch["image"],
        "label": batch["label"],
    }
    return jax.tree_map(functools.partial(_prepare_data, local_device_count=local_device_count), batch)
