import argparse
import os

from composer.utils.object_store.mlflow_object_store import (
    MLFlowObjectStore,
)


def upload_checkpoint(checkpoint_dir: str, mlflow_path: str):
    object_store = MLFlowObjectStore(path=mlflow_path)

    for root, _, files in os.walk(checkpoint_dir):
        for file in files:
            file_path = os.path.join(root, file)
            artifact_path = os.path.relpath(
                file_path, checkpoint_dir
            )  # Make the path relative to the checkpoint_dir
            object_store.upload_object(
                object_name=artifact_path, filename=file_path
            )
            print(f"Uploaded {file_path} as {artifact_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Upload Composer checkpoint to MLflow artifact store.'
    )
    parser.add_argument(
        'checkpoint_dir',
        type=str,
        help='Path to the Composer checkpoint directory'
    )
    parser.add_argument(
        'mlflow_path',
        type=str,
        help=
        'MLflow path for the artifact store (e.g., databricks/mlflow-tracking/{mlflow_experiment_id}/{mlflow_run_id}/artifacts/{path})'
    )

    args = parser.parse_args()

    upload_checkpoint(args.checkpoint_dir, args.mlflow_path)
