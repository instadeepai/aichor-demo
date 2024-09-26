import os
import ray

def rayop():
    ray.init(address=os.environ.get("RAY_SERVER", "auto"))

    nodes = ray.nodes()
    print("connected nodes: ", nodes)