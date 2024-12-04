import argparse
import functools
import os
import time
import sys
from typing import Any, Tuple, Generator, Dict, Callable

import numpy as np

import jax
import jax.numpy as jnp
from jax.random import PRNGKey, KeyArray
from tensorboardX import SummaryWriter

from flax.training import train_state, checkpoints
import flax.jax_utils as jax_utils
import flax.linen as nn
from flax.linen import FrozenDict

import optax
from s3fs import S3FileSystem

from tqdm import tqdm
import tensorflow as tf

import data
from model import MySuperModel

BATCH_SIZE = 64


# maybe one day get this from a lib
AICHOR_TENSORBOARD_PATH = "AICHOR_TENSORBOARD_PATH"
AICHOR_OUTPUT_PATH = "AICHOR_OUTPUT_PATH"

class TrainState(train_state.TrainState):
    batch_stats: Any

def create_train_state(rng: KeyArray, sample_input: jnp.ndarray) -> TrainState:
    """
        Create initial training state.

        Init the model and create the optimizer.

        Args:
            rng (KeyArray): a jax PRNGKey to init the model
            sample_input (jnp.ndarray): a sample input to init the model 

        Returns:
            TrainState: our trainning state
    """
    model = MySuperModel()

    # jiting make init lazy
    init = jax.jit(model.init)
    params = init(rng, sample_input)

    tx = optax.sgd(0.01)

    return TrainState.create(
        apply_fn=model.apply,
        params=params["params"],
        tx=tx,
        batch_stats=params["batch_stats"],
    )


def loss_fn(logits, labels):
    """Our Loss function
    Simply use cross entropy loss
    """
    return -jnp.mean(labels * logits)


def compute_loss(apply_fn: Callable,
                 params: FrozenDict,
                 batch_stats: FrozenDict,
                 batch=Dict[str, jnp.ndarray]):
    """Compute the loss for 1 batch

    Args:
        apply_fn (Callable): apply function of the model
        params: (FrozenDict): parameters of the model
        batch_stats: (FrozenDict): batch stats for batch normalization layers
        batch: (Dict[str, jnp.ndarray]): a dict container x and y
    """
    logits, new_state = apply_fn(
        {
            "params": params,
            "batch_stats": batch_stats,
        },
        batch["x"],  # type: ignore
        mutable=['batch_stats'])
    loss = loss_fn(logits, batch["y"])  # type: ignore
    return loss, (new_state, {
        "accuracy":
        jnp.mean(jnp.argmax(logits, -1) == jnp.argmax(batch["y"], -1)),  # type: ignore
    })


# create a function to get the grad over parameters
compute_loss_grad = jax.value_and_grad(compute_loss, argnums=1, has_aux=True)

# utils functions to apply pmean over the batch axis
pmean_batch = functools.partial(jax.lax.pmean, axis_name="batch")


def train_step(state: TrainState, batch: Dict[str, jnp.ndarray]):
    """Compute loss and grad for 1 batch and apply it using the optimizer
    
    Returns:
        Tuple[TrainState, Dict[str, Any]]
        The new updated state and metrics about this batch 
    """
    res, grads = compute_loss_grad(state.apply_fn, state.params,
                                   state.batch_stats, batch)

    grads = jax.tree_map(pmean_batch, grads)

    new_state, metrics = res[1]
    metrics = jax.tree_map(pmean_batch, metrics)

    state = state.apply_gradients(
        grads=grads,
        batch_stats=new_state["batch_stats"],
    )

    return state, metrics


def train_epoch(state: TrainState,
                train_loader: Generator,
                writer: SummaryWriter,
                epoch: int,
                step_per_epoch: int,
                silence: bool = False):
    """Train for a single epoch.

    iterate 1 time on the train_loader
    """
    acc = 0.0
    bar = tqdm(train_loader,
               total=step_per_epoch) if not silence else train_loader
    p_train_step = jax.pmap(
        train_step,
        axis_name="batch",
    )

    for i, batch in enumerate(bar):
        batch = {
            "x": jnp.array(batch["image"]),
            "y": jax.nn.one_hot(batch["label"], 10),
        }
        state, train_metrics = p_train_step(state, batch)

        acc += train_metrics["accuracy"]

        writer.add_scalar("train/accuracy", train_metrics["accuracy"], epoch * step_per_epoch + i)
        if not silence:
            bar.set_postfix({"mean_acc": acc / (i + 1)})  # type: ignore

    return state


def main():
    rng = PRNGKey(int(time.time()))
    local_device_count = jax.local_device_count()

    def get_dataloader() -> Tuple[Generator, int]:
        # here you could load diffenre part of your dataset depending on the process id
        dataset = data.get_dataset(BATCH_SIZE)
        dataset_len = len(dataset)
        train_loader = map(
            functools.partial(data.prepare_data,
                              local_device_count=local_device_count), dataset)
        train_loader = jax_utils.prefetch_to_device(train_loader, 2)
        return train_loader, dataset_len

    state = create_train_state(rng, jnp.ones((BATCH_SIZE, 32, 32, 3)))

    state = jax_utils.replicate(state)
    writer = SummaryWriter(os.environ.get("AICHOR_TENSORBOARD_PATH"))

    for epoch in range(10):
        dl, dl_len = get_dataloader()
    
        state = train_epoch(state, dl, writer, epoch, dl_len)

        # add test here

        # only coordinator save model
        if jax.process_index() == 0:
            checkpoints.save_checkpoint(ckpt_dir="/checkpoint", target=state, step=epoch, keep_every_n_steps=3)  # type: ignore
            s3 = S3FileSystem(endpoint_url=os.environ.get("AWS_ENDPOINT_URL"))
            s3.put("/checkpoint", os.environ.get(AICHOR_OUTPUT_PATH), recursive=True)
    
    writer.close()


if __name__ == "__main__":
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    # Make sure tensorflow don't take the GPUs since we only use it for datasets
    tf.config.experimental.set_visible_devices([], 'GPU')

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--coordinator_address",
        type=str,
        default=os.environ.get("JAXOPERATOR_COORDINATOR_ADDRESS", "localhost:5000"),
        help=
        "Set the address of the coordinator, if this process is the coordinator then it will be the address that it will bind for the other to connect"
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=os.environ.get("JAXOPERATOR_NUM_PROCESSES", 1),
        help="Set total number of process for this jax distributed")
    parser.add_argument(
        "--process_id",
        type=int,
        default=os.environ.get("JAXOPERATOR_PROCESS_ID", 0),
        help=
        "Set the id of this process (O <= process_id < num_processes), if set to 0 then this process is the coordinator"
    )
    args = parser.parse_args()

    print("Starting JAX distributed")
    jax.distributed.initialize(
        coordinator_address=args.coordinator_address,
        num_processes=args.num_processes,
        process_id=args.process_id,
    )

    if jax.process_index() == 0:
        print("Coordinator process")
    print("JAX devices:", jax.devices())
    print("JAX backend:", jax.lib.xla_bridge.get_backend().platform)  # type: ignore
    print("JAX host_id:", jax.lib.xla_bridge.get_backend().host_id(), " ", jax.process_index())  # type: ignore
    print("Jax process count:", jax.process_count())

    print(f"local: {jax.local_devices()}")
    print(f"all: {jax.devices()}")
    main()
