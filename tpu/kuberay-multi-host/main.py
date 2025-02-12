import ray
import jax
import os

@ray.remote(resources={"TPU": 4})
def tpu_cores(index: int):
    device_count = jax.device_count()
    return f"Index [{index}]: Global TPU Count={device_count}"

if __name__ == "__main__":
    ray.init(address=os.environ.get("RAY_ADDRESS", "auto"), log_to_driver=True)

    num_workers = 4 # number of pods in 4x4 topology on v5e
    print(f"starting {num_workers} remote functions")

    result = [tpu_cores.remote(i) for i in range(num_workers)]
    print(ray.get(result))

    print("Done! exiting")
    ray.shutdown()