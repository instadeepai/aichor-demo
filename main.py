import argparse
import time

from src.operators.jax import jaxop
from src.operators.ray import rayop
from src.operators.tf import tfop
from src.operators.pytorch import pytorchop
from src.operators.xgboost import xgboostop


def generate_logs(size, unit='B', batch_size=1024):
    # Conversion of size to bytes
    units = {
        'B': 1,
        'KiB': 1024,
        'MiB': 1024**2,
        'GiB': 1024**3,
        'TiB': 1024**4
    }
    total_size_bytes = size * units[unit]
    log_entry = "beyrem : This is a log entry.\n"
    log_entry_size = len(log_entry)
    # Generate logs in small batches
    bytes_written = 0
    batch = []
    while bytes_written < total_size_bytes:
        # Calculate how many entries to generate in this batch
        entries_in_batch = min(batch_size, (total_size_bytes - bytes_written) // log_entry_size)
        # Create the batch of log entries
        batch = [log_entry] * entries_in_batch
        # Print out the batch
        print("".join(batch), end='')
        # Update the number of bytes written    
        bytes_written += log_entry_size * entries_in_batch

OPERATOR_TABLE = {
    "ray": rayop,
    "kuberay": rayop,
    "tf": tfop,
    "jax": jaxop,
    "pytorch": pytorchop,
    "xgboost": xgboostop
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='AIchor Z S   moke test on any operator')
    parser.add_argument("--operator", type=str, default="tf", choices=OPERATOR_TABLE.keys(),help="operator name")
    parser.add_argument("--sleep", type=int, default="0", help="sleep time in seconds")
    parser.add_argument("--tb-write", type=bool, default=False, help="test write to tensorboard")

    args = parser.parse_args()

    print(f"using {args.operator} operator")
    OPERATOR_TABLE[args.operator](args.tb_write)
    generate_logs(1, unit='MiB')
    print("Beyrem: Good bey")
