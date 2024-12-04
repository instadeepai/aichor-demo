import os
from src.utils.tensorboard import dummy_tb_write

def jaxop(tb_write: bool):
    p_id = os.environ.get("JAXOPERATOR_PROCESS_ID")

    print("coordinator address: ", os.environ.get("JAXOPERATOR_COORDINATOR_ADDRESS"))
    print("num processes: ", os.environ.get("JAXOPERATOR_NUM_PROCESSES"))
    print("process id: ", p_id)

    if tb_write:
        dummy_tb_write(f"From p_id {p_id}")