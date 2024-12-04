import pyarrow.fs
import os
import ray
from ray import tune, train

if __name__ == "__main__":
    fs = pyarrow.fs.S3FileSystem(endpoint_override=os.environ["AWS_ENDPOINT_URL"])
    tensorboard_path = os.getenv('AICHOR_TENSORBOARD_PATH')[5:] # Get path 

    # ray.init(local_mode=True)
    ray.init(address=os.environ.get("RAY_ADDRESS", "auto"))

    tuner = tune.Tuner(
        "PPO",
        tune_config=tune.TuneConfig(
            # metric="env_runners/episode_return_mean",
            mode="max",
            num_samples=1,
        ),
        param_space={
            "env": "ALE/Pong-v5",
            "framework": "torch",
            "env_config": {
                "frameskip": 1,
                "full_action_space": False,
                "repeat_action_probability": 0.0,
            },
            "lambda": 0.95,
            "kl_coeff": 0.5,
            "clip_rewards": True,
            "clip_param": 0.1,
            "vf_clip_param": 10.0,
            "entropy_coeff": 0.01,
            "train_batch_size": 4096,
            "rollout_fragment_length": "auto",
            "sgd_minibatch_size": 500,
            "num_sgd_iter": 10,
            "num_workers": 32,
            "num_envs_per_worker": 8,
            "batch_mode": "truncate_episodes",
            "observation_filter": "NoFilter",
            "num_gpus": 1,
            "model": {
                "dim": 42,
                "vf_share_layers": True
            }
        },
        run_config=train.RunConfig(
            stop={
                "env_runners/episode_return_mean": 18,
                # "timesteps_total": 5000000
            },
            storage_filesystem=fs,
            storage_path=tensorboard_path,
        ),
    )
    results = tuner.fit()