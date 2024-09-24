cat << 'EOF' > upload_to_uc.py
import argparse
import os
from pathlib import Path

import mlflow
import pandas as pd
from mlflow import MlflowClient
from mlflow.models.signature import ModelSignature
from mlflow.types import ColSpec, DataType, Schema
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmfoundry.models.utils import init_empty_weights


def download_artifacts(
    experiment_path: str, run_name: str, local_dir: str
) -> Path:
    # Create the local directory if it doesn't exist
    os.makedirs(local_dir, exist_ok=True)

    # Get the MLflow experiment by the given experiment path
    experiment = mlflow.get_experiment_by_name(name=experiment_path)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_path}' not found.")

    # Get the experiment run id given the experiment run name
    client = MlflowClient()
    runs = client.search_runs(experiment_ids=[experiment.experiment_id])
    run_id = None
    for run in runs:
        if run.info.run_name == run_name:
            run_id = run.info.run_id
            break
    if run_id is None:
        raise ValueError(
            f"Run '{run_name}' not found in experiment '{experiment_path}'."
        )

    print(f"Found MLflow run with run_id: {run_id}")

    print("Start downloading artifacts to local directory...")
    # Download all artifacts
    all_artifacts = mlflow.artifacts.list_artifacts(run_id=run_id)
    for artifact in all_artifacts:
        if not os.path.basename(artifact.path
                               ).startswith('_') and not os.path.basename(
                                   artifact.path
                               ).startswith('.'):
            print(f"Downloading artifact at path: {artifact.path}")
            mlflow.artifacts.download_artifacts(
                run_id=run_id, artifact_path=artifact.path, dst_path=local_dir
            )

    # Print the contents of the local directory for debugging
    print(f"Contents of {local_dir}:")
    for root, dirs, files in os.walk(local_dir):
        for name in files:
            print(os.path.join(root, name))
        for name in dirs:
            print(os.path.join(root, name))

    return Path(local_dir)


def load_model_from_mlflow_and_log_to_uc(
    experiment_path: str, run_name: str, model_name: str, local_dir: str
):
    # Given the experiment path, download all the artifacts associated with the experiment
    # parent_dir = download_artifacts(experiment_path, run_name, local_dir)
    # print(f"Finished downloading files to {parent_dir}. Now loading model...")

    # # Check if essential model files are present
    # config_path = parent_dir / 'config.json'
    # if not config_path.exists():
    #     raise FileNotFoundError(f"config.json not found in {parent_dir}")
    parent_dir = '/test'

    # Now that all files are downloaded, you can load the model and tokenizer from the local directory
    tokenizer = AutoTokenizer.from_pretrained(
        parent_dir, trust_remote_code=True
    )
    print("Tokenizer loaded...")
    print("loading model")
    model = None
    with init_empty_weights():
        model = AutoModelForCausalLM.from_pretrained(
            parent_dir, trust_remote_code=True
        )
    print("Model loaded...")

    # Load model, we will then log the model in MLflow ensuring to add the metadata to the mlflow.transformers.log_model.
    print("Creating Experiment")
    mlflow.set_experiment(experiment_path)
    print("Experiment created")

    # Log model to MLflow - This will take approx. 5mins to complete.
    # Define input and output schema
    input_schema = Schema([
        ColSpec(DataType.string, "prompt"),
        ColSpec(DataType.double, "temperature", optional=True),
        ColSpec(DataType.long, "max_tokens", optional=True)
    ])

    output_schema = Schema([ColSpec(DataType.string)])
    signature = ModelSignature(inputs=input_schema, outputs=output_schema)

    print("Logging model with MLflow.")
    with mlflow.start_run() as mlflow_run:
        components = {
            "model": model,
            "tokenizer": tokenizer,
        }

        mlflow.transformers.log_model(
            transformers_model=components,
            artifact_path="model",
            input_example=pd.DataFrame({
                "prompt": ["what is mlflow?"],  # This input example is just an example
                "temperature": [0.1],
                "max_tokens": [256]
            }),
            signature=signature,
            task='text-generation',
            metadata={'task': 'text-generation'},  # This metadata is currently needed for optimized serving
            registered_model_name=model_name,
        )

        return mlflow_run


def main():
    parser = argparse.ArgumentParser(
        description="Load model from MLflow and log to Unity Catalog."
    )
    parser.add_argument(
        "--experiment_path",
        type=str,
        required=True,
        help="Path to the MLflow experiment."
    )
    parser.add_argument(
        "--run_name", type=str, required=True, help="Name of the MLflow run."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name to register the model under in Unity Catalog."
    )
    parser.add_argument(
        "--local_dir",
        type=str,
        required=True,
        help="Local directory to download artifacts to."
    )

    args = parser.parse_args()

    mlflow.set_tracking_uri('databricks')
    mlflow.set_registry_uri('databricks-uc')

    mlflow_run = load_model_from_mlflow_and_log_to_uc(
        experiment_path=args.experiment_path,
        run_name=args.run_name,
        model_name=args.model_name,
        local_dir=args.local_dir
    )
    print(f"MLflow Run ID: {mlflow_run.info.run_id}")


if __name__ == "__main__":
    main()
EOF
