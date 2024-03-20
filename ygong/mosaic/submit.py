from ygong.mosaic.scaling_config import ScalingConfig
from ygong.mosaic.mpt125mConfig import MPT125MConfig
from mcli import wait_for_run_status, Run, RunConfig, RunStatus, create_run
import pandas as pd
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML
import mlflow
import os
from typing import Optional
import time
import base64
import json
import hashlib
from mcli.api.engine.engine import MAPIConnection
from mcli.config import MCLIConfig

def _set_up_environment(content):
     data = json.loads(base64.b64decode(content).decode('utf-8'))
     workspace_url = data.get("workspace_url", None)
     token = data.get("token", None)
     mosaic_token = data.get("mosaic_token", None)
     
     if token is None:
        from databricks.sdk import WorkspaceClient
        wc = WorkspaceClient()
        ctx = wc.dbutils.entry_point.getDbutils().notebook().getContext()
        token = ctx.apiToken().get()

     if workspace_url is None or mosaic_token is None:
        raise ValueError("workspace_url and token must be provided")
     os.environ['WORKSPACE_URL'] = workspace_url
     os.environ['MLFLOW_TRACKING_TOKEN'] = token
     
     # set up the mosaic token
     conf = MCLIConfig.load_config()
     conf.api_key = mosaic_token
     conf.save_config()
     MAPIConnection.reset_connection()

     
     hash = hashlib.sha256(f"{workspace_url}-{token}-{mosaic_token}".encode()).hexdigest()[:8]
     databricks_secret_name = f"databricks-{hash}"
     
     # clean up the old secret. MosaicML doesn't support multiple databricks secrets
     # would have to clean up the old secret if it exists
     from mcli.api.secrets.api_get_secrets import get_secrets
     from mcli.api.secrets.api_delete_secrets import delete_secrets
     from mcli.models.mcli_secret import SecretType
     s = get_secrets(secret_types=[SecretType.databricks])
     if len(s) == 1:
        if s[0].name != databricks_secret_name:
            delete_secrets(s)
        else:
            print("databricks secret already exists")
            return
     from mcli.objects.secrets.create.databricks import DatabricksSecretCreator
     from mcli.api.secrets.api_create_secret import create_secret
     s = DatabricksSecretCreator().create(name=databricks_secret_name, host=workspace_url, token=token)
     print(f"successfully created databricks secret: {databricks_secret_name}")
     create_secret(s)

     

def get_experiment_run_url(tracking_uri: Optional[str], experiment_name: str, run_name: str):
      if tracking_uri is None:
          raise ValueError("tracking_uri must be provided")
      mlflow.set_tracking_uri(tracking_uri)
      tracking_uri = tracking_uri.rstrip("/")
      experiment = mlflow.get_experiment_by_name(name=experiment_name)
      if experiment is None:
          raise ValueError(f"experiment {experiment_name} does not exist")
      experiment_id = experiment.experiment_id
      runs = mlflow.search_runs(experiment_ids=[experiment_id],
                                                   filter_string=f'tags.composer_run_name = "{run_name}"',
                                                   output_format='list')
      if len(runs) == 0:
            raise ValueError(f"run {run_name} does not exist in experiment {experiment_name}")
      elif len(runs) > 1:
            raise ValueError(f"multiple runs {run_name} exist in experiment {experiment_name}")
      else:
            run_id = runs[0].info.run_id
            return f"{tracking_uri}/ml/experiments/{experiment_id}/runs/{run_id}"
      
def _get_run_summary(run: Run, experiment_name: Optional[str] = None):
    url = None
    if run.status == RunStatus.RUNNING and experiment_name is not None:
          url = get_experiment_run_url(os.environ.get('WORKSPACE_URL'), experiment_name, run.name)
    
    df = pd.DataFrame({
         'Run Name': [run.name],
         'Run ID': [run.run_uid],
         "Status": [str(run.status)],
         'Experiment Run': [f'<a href="{url}">Link</a>' if url is not None else ""],
    })
    return df

def _display_run_summary(summary: pd.DataFrame, cancel_button: Optional[widgets.Button]):
    clear_output(wait=True)
    if cancel_button is not None:
        display(cancel_button)
    display(HTML(summary.to_html(escape=False)))



def submit(model, config: any, scalingConfig: ScalingConfig):
    mlflow_experiment_name = None
    if model == "mpt125m":
        if not isinstance(config, MPT125MConfig):
            raise ValueError("config must be an instance of MPT125MConfig")
        mlflow_experiment_name = config.mlflow_experimentName
        runConfig = config.toRunConfig(scalingConfig)
    else:
        raise ValueError(f"model {model} is not supported")
    
    
    run = create_run(runConfig)
    # Create a button
    button = widgets.Button(description="cancel the run")
    def on_button_clicked(b):
        clear_output(wait=False)
        run.stop()
        wait_for_run_status(run, RunStatus.TERMINATING)
        summary = _get_run_summary(run, mlflow_experiment_name)
        display(HTML(summary.to_html(escape=False)))
    button.on_click(on_button_clicked)

    _display_run_summary(_get_run_summary(run, mlflow_experiment_name), button)
    
    wait_for_run_status(run, RunStatus.RUNNING)
    # setting mlflow_experiment_name to be None, since its very likely mlflow run isn't ready yet
    # when the run just starts running
    _display_run_summary(_get_run_summary(run, None), button)
    

    try_count = 0
    while try_count < 10:
        try_count += 1
        time.sleep(20)
        try:
            summary = _get_run_summary(run, mlflow_experiment_name)
            _display_run_summary(summary, button)
            break
        except ValueError:
             print("waiting for the run to be ready...")
             pass