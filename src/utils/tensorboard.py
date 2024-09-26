import os
from tensorboardX import SummaryWriter

def dummy_tb_write():
    log_path = os.environ.get("AICHOR_LOGS_PATH")
    if log_path == None:
        print("\"AICHOR_LOGS_PATH\" env var not found")
        return
    
    writer = SummaryWriter(log_path)
    message = os.environ.get("VCS_COMMIT_MESSAGE")
    if message == None:
        message = "VCS_COMMIT_MESSAGE env var not found"
    writer.add_text("testing text", message, 0)
    writer.flush()
    writer.close()
