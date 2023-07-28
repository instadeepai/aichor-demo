import os

def jaxop():
    print_jax_env()

def print_jax_env():
    print("coordinator address: ", os.environ.get("JAXOPERATOR_COORDINATOR_ADDRESS"))
    print("num processes: ", os.environ.get("JAXOPERATOR_NUM_PROCESSES"))
    print("process id: ", os.environ.get("JAXOPERATOR_PROCESS_ID"))