import os

def rayop():
    try:
        import ray
    except:
        print("cannot import ray")
        return

    ray.init(address=os.environ.get("RAY_SERVER", "auto"))

    nodes = ray.nodes()
    print("connected nodes: ", nodes)