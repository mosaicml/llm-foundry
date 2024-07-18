import argparse
import os

from composer.utils.object_store.mlflow_object_store import MLFlowObjectStore


def upload_checkpoint(checkpoint_dir: str, mlflow_path: str):
    # Convert to absolute path if not already
    checkpoint_dir = os.path.abspath(checkpoint_dir)
    
    # Check if the directory exists
    if not os.path.isdir(checkpoint_dir):
        print(f"Error: The checkpoint directory '{checkpoint_dir}' does not exist or is not accessible.")
        return

    object_store = MLFlowObjectStore(path=mlflow_path)
    print("object_store:", object_store)

    print(f"Walking through the checkpoint directory: {checkpoint_dir}")
    
    for root, _, files in os.walk(checkpoint_dir):
        print(f"Current directory: {root}")
        print(f"Files: {files}")
        
        for file in files:
            file_path = os.path.join(root, file)
            print(f"Processing file: {file_path}")
            
            artifact_path = os.path.relpath(file_path, checkpoint_dir)  # Make the path relative to the checkpoint_dir
            print(f"Uploading {file_path} as {artifact_path}")
            
            object_store.upload_object(object_name=artifact_path, filename=file_path)
            print(f"Uploaded {file_path} as {artifact_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Upload Composer checkpoint to MLflow artifact store.')
    parser.add_argument('--checkpoint_dir', type=str, required=True, help='Path to the Composer checkpoint directory')
    parser.add_argument('--mlflow_path', type=str, required=True, help='MLflow path for the artifact store (e.g., databricks/mlflow-tracking/{mlflow_experiment_id}/{mlflow_run_id}/artifacts/{path})')

    args = parser.parse_args()

    print(f"Current working directory: {os.getcwd()}")
    print(f"Checkpoint directory: {args.checkpoint_dir}")
    print(f"MLflow path: {args.mlflow_path}")

    upload_checkpoint(args.checkpoint_dir, args.mlflow_path)

