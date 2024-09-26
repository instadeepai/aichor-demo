import os
from tensorboardX import SummaryWriter

def dummy_tb_write(message: str):
    log_path = os.environ.get("AICHOR_LOGS_PATH")
    if log_path == None:
        print("\"AICHOR_LOGS_PATH\" env var not found")
        return

    aichor_message = os.environ.get("AICHOR_EXPERIMENT_MESSAGE")

    if message == None:
        message = aichor_message
    else:
        message =  f"{message} - {aichor_message}"

    writer = SummaryWriter(log_path)
    writer.add_text("testing text", message, 0)
    writer.flush()
    writer.close()
