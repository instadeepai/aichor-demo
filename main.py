import argparse
import time

from src.operators.jax import jaxop
from src.operators.ray import rayop
from src.operators.tf import tfop

from src.utils.tensorboard import dummy_tb_write

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='AIchor Smoke test on any operator')
    parser.add_argument("--operator", type=str, default="tf", choices=["ray", "jax", "tf"],help="operator name")
    parser.add_argument("--sleep", type=int, default="0", help="sleep time in seconds")
    parser.add_argument("--tb-write", type=bool, default=False, help="test write to tensorboard")

    args = parser.parse_args()

    print(f"using {args.operator} operator")

    if args.operator == "ray":
        rayop()
    elif args.operator == "jax":
        jaxop()
    elif args.operator == "tf":
        tfop()
    
    if args.tb_write:
        dummy_tb_write()
    
    if args.sleep > 0:
        print(f"sleeping for {args.sleep}s before exiting")
        time.sleep(args.sleep)