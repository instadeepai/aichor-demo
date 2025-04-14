import os
import shutil

from accelerate import Accelerator

from s3fs import S3FileSystem

from constant import AICHOR_OUTPUT_PATH

def save_final_model(accelerator: Accelerator, model, s3: S3FileSystem):
    local_path = "final_model"
    output_path = os.environ.get(AICHOR_OUTPUT_PATH)

    if accelerator.is_main_process:
        print(f"Saving trained model at: {output_path} from main process")
        accelerator.save_model(model, local_path)
        print(f"Model saved at {local_path}")
        if s3 is not None:
            print(f"Uploading model to {output_path}")
            s3.put(local_path, output_path, recursive=True)
            shutil.rmtree(local_path)
            print("Uploaded")

    accelerator.wait_for_everyone()