from __future__ import annotations

import os, sys, time, functools
from typing import Any, Callable, Dict, Tuple, Generator

import jax
import jax.numpy as jnp
from jax.random import PRNGKey, KeyArray

import flax.jax_utils as fjx
import flax.linen as nn
from flax.training import train_state, checkpoints
from flax.linen import FrozenDict

import optax
from tensorboardX import SummaryWriter
from tqdm import tqdm
import tensorflow as tf
from s3fs import S3FileSystem

import data
from model import MySuperModel


def _require(name: str) -> str:
    val = os.getenv(name)
    if val is None:
        print(f"required env-var '{name}' is not set", file=sys.stderr)
        sys.exit(1)
    return val

def infer_process_info() -> tuple[str, int, int]:

    rank        = int(_require("JOB_COMPLETION_INDEX"))
    world_size  = int(_require("WORLD_SIZE"))
    host        = _require("COORDINATOR_SERVICE_HOST")
    port        = _require("COORDINATOR_SERVICE_PORT")
    coordinator = f"{host}:{port}"

    return coordinator, world_size, rank

BATCH_SIZE = 64

class TrainState(train_state.TrainState):
    batch_stats: Any

def create_train_state(rng: KeyArray, sample_input: jnp.ndarray) -> TrainState:
    model   = MySuperModel()
    params  = jax.jit(model.init)(rng, sample_input)
    tx      = optax.sgd(0.01)
    return TrainState.create(
        apply_fn   = model.apply,
        params     = params["params"],
        tx         = tx,
        batch_stats= params["batch_stats"],
    )

def loss_fn(logits, labels):
    return -jnp.mean(labels * logits)

def compute_loss(apply_fn: Callable,
                 params: FrozenDict,
                 batch_stats: FrozenDict,
                 batch: Dict[str, jnp.ndarray]):
    logits, new_state = apply_fn({"params": params, "batch_stats": batch_stats},
                                 batch["x"], mutable=["batch_stats"])
    loss = loss_fn(logits, batch["y"])
    acc  = jnp.mean(jnp.argmax(logits, -1) == jnp.argmax(batch["y"], -1))
    return loss, (new_state, {"accuracy": acc})

compute_loss_grad = jax.value_and_grad(compute_loss, argnums=1, has_aux=True)
pmean = functools.partial(jax.lax.pmean, axis_name="batch")

def train_step(state: TrainState, batch: Dict[str, jnp.ndarray]):
    (res, grads) = compute_loss_grad(state.apply_fn, state.params,
                                     state.batch_stats, batch)
    grads        = jax.tree_map(pmean, grads)
    new_state, metrics = res[1]
    metrics      = jax.tree_map(pmean, metrics)
    state        = state.apply_gradients(
                       grads=grads,
                       batch_stats=new_state["batch_stats"])
    return state, metrics

def train_epoch(state: TrainState, loader: Generator, writer: SummaryWriter,
                epoch: int, steps: int, silent: bool=False):
    acc = 0.0
    bar = tqdm(loader, total=steps) if not silent else loader
    p_train = jax.pmap(train_step, axis_name="batch")

    for i, batch in enumerate(bar):
        batch = {"x": jnp.array(batch["image"]),
                 "y": jax.nn.one_hot(batch["label"], 10)}
        state, m = p_train(state, batch)
        acc += m["accuracy"]
        writer.add_scalar("train/accuracy", m["accuracy"], epoch*steps + i)
        if not silent:
            bar.set_postfix(mean_acc=float(acc/(i+1)))
    return state

def main():
    coord, world, rank = infer_process_info()

    tf.config.experimental.set_visible_devices([], "GPU")
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

    print(f"rank {rank}/{world-1} connecting to {coord}")
    jax.distributed.initialize(coordinator_address=coord,
                               num_processes=world,
                               process_id=rank)

    rng            = PRNGKey(int(time.time()))
    local_devices  = jax.local_device_count()

    def make_loader() -> Tuple[Generator, int]:
        ds = data.get_dataset(BATCH_SIZE)
        gen = map(functools.partial(data.prepare_data,
                                    local_device_count=local_devices), ds)
        return fjx.prefetch_to_device(gen, 2), len(ds)

    state  = create_train_state(rng, jnp.ones((BATCH_SIZE, 32, 32, 3)))
    state  = fjx.replicate(state)
    writer = SummaryWriter(os.getenv("AICHOR_TENSORBOARD_PATH", "/tmp/tb"))

    for epoch in range(10):
        loader, steps = make_loader()
        state = train_epoch(state, loader, writer, epoch, steps)
        if jax.process_index() == 0:
            checkpoints.save_checkpoint("/checkpoint", state, epoch,
                                         keep_every_n_steps=3)
            s3 = S3FileSystem(endpoint_url=os.getenv("AWS_ENDPOINT_URL"))
            s3.put("/checkpoint", os.getenv("AICHOR_OUTPUT_PATH", "/tmp/ckpt"),
                   recursive=True)

    writer.close()
    print(f"training complete on rank {rank}")

if __name__ == "__main__":
    main()
