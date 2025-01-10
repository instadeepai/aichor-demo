import argparse
import time

from src.operators.jax import jaxop
from src.operators.ray import rayop
from src.operators.tf import tfop
from src.operators.pytorch import pytorchop
from src.operators.xgboost import xgboostop
from src.operators.batchjob import batchjobop

OPERATOR_TABLE = {
    "ray": rayop,
    "kuberay": rayop,
    "tf": tfop,
    "jax": jaxop,
    "pytorch": pytorchop,
    "xgboost": xgboostop,
    "batchjob": batchjobop
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='AIchor Smoke test on any operator')
    parser.add_argument("--operator", type=str, default="tf", choices=OPERATOR_TABLE.keys(),help="operator name")
    parser.add_argument("--sleep", type=int, default="0", help="sleep time in seconds")
    parser.add_argument("--tb-write", type=bool, default=False, help="test write to tensorboard")

    args = parser.parse_args()

    print(f"using {args.operator} operator")
    OPERATOR_TABLE[args.operator](args.tb_write)

    if args.sleep > 0:
        print(f"sleeping for {args.sleep}s before exiting")
        time.sleep(args.sleep)