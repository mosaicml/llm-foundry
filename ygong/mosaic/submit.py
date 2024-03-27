from ygong.mosaic.scaling_config import ScalingConfig
from ygong.mosaic.mpt125mConfig import MPT125MConfig

from databricks.sdk import WorkspaceClient
from mcli import config, Run, RunStatus, create_run
from mcli.api.runs.api_get_runs import get_run
from mcli.cli.m_get.runs import RunDisplayItem
from IPython.display import display, clear_output, HTML
import ipywidgets as widgets
import mlflow
import pandas as pd

from typing import Optional
import base64
import time
import json
import logging
import os
import sys
from mcli import RunConfig
import hashlib
from mcli.config import MCLIConfig
from mcli.api.engine.engine import MAPIConnection

logger = logging.getLogger('ygong.mosaic.submit')

def _set_up_environment(content: str):
    os.environ['CREDENTIALS'] = content

     
def _init_connection():
     def _is_local():
        if os.environ.get('CREDENTIALS') is not None:
            return True 
        try:
            wc = WorkspaceClient()
            wc.dbutils.entry_point.getDbutils().notebook().getContext()
            return False
        except:
            return True
        
     if _is_local():
        logger.debug("init_connection in local mode")
        if os.environ.get('CREDENTIALS') is None:
            raise ValueError("_set_up_environment must be manually called to configure credentials for local runs")
        data = json.loads(base64.b64decode(os.environ.get('CREDENTIALS')).decode('utf-8'))
        workspace_url = data.get("workspace_url", None)
        token = data.get("token", None)
        mosaic_token = data.get("mosaic_token", None)
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
     else:
        logger.debug("init_connection in databricks environment")
        wc = WorkspaceClient()
        ctx = wc.dbutils.entry_point.getDbutils().notebook().getContext()
        token = ctx.apiToken().get()
        api_url = ctx.apiUrl().get()
        endpoint = f'{api_url}/api/2.0/genai-mapi/graphql'
        workspace_url = api_url
        os.environ[config.MOSAICML_API_KEY_ENV] = f'Bearer {token}'
        os.environ[config.MOSAICML_API_ENDPOINT_ENV] = endpoint
        try:
            jobs_id = ctx.jobId().get()
            os.environ['JOB_ID'] = jobs_id
        except:
            pass

     # needed to set up the MLFlow query for experiment runs   
     os.environ['WORKSPACE_URL'] = workspace_url
     os.environ['MLFLOW_TRACKING_TOKEN'] = token
     logger.debug(f"init_connection token: {os.environ['MLFLOW_TRACKING_TOKEN']}, workspace: {os.environ['WORKSPACE_URL']}, is_jobs: {os.environ.get('JOB_ID')}")
        

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
        
    run_rows = []

    # Copy pasted from mcli to display the the resumption status of the run.
    for row_raw in RunDisplayItem.from_run(run, [], True):
        row = row_raw.to_dict()
        if row['Status'].startswith('Running') and experiment_name is not None:
            url = get_experiment_run_url(os.environ.get('WORKSPACE_URL'), experiment_name, run.name)
        row['Experiment Run'] =f'<a href="{url}">Link</a>' if url is not None else ""
        run_rows.append(row)
    
    df = pd.DataFrame(run_rows)
    return df

def _display_run_summary(summary: pd.DataFrame, cancel_button: Optional[widgets.Button]):
    clear_output(wait=True)
    if cancel_button is not None:
        display(cancel_button)
    display(HTML(summary.to_html(escape=False)))

def _wait_for_run_status(run: Run, status: RunStatus, inclusive: bool = True):
    run_name = run.name
    while not run.status.after(status, inclusive=inclusive):
        time.sleep(5)
        run =  get_run(run_name)
        logger.debug(f"run status {run.status}, expected status {status}")
    logger.debug(f"finish waiting run reached expected status {status}")
    return run

def submit(config: any, scalingConfig: ScalingConfig, sync: bool = False, debug: bool = False):
    if debug:
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(logging.DEBUG)  # Set minimum log level for the handler
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        stdout_handler.setFormatter(formatter)

        # Add the handler to the logger
        logger.addHandler(stdout_handler)
        logger.setLevel(logging.DEBUG)
        
        logger.info("set the logger to debug mode")
        
    # MTC + AWS Dogfood
    _init_connection()
    mlflow_experiment_name = None
    if isinstance(config, MPT125MConfig):
        mlflow_experiment_name = config.mlflow_experimentName
        runConfig = config.toRunConfig(scalingConfig)
    elif isinstance(config, RunConfig):
        runConfig = config
        mlflow_experiment_name = runConfig.name
    else:
        raise ValueError(f"config type {type(config)} is not supported")
    
    run = create_run(runConfig)
    run_name = run.name
    # Create a button
    if os.environ.get('JOB_ID') is not None:
        # running in jobs workflow, no need to cancel the run and doesn't support widgets
        button = None
    else:
        button = widgets.Button(description="cancel the run")
        def on_button_clicked(b):
            logger.debug(f"cancel button clicked")
            clear_output(wait=False)
            run = get_run(run_name)
            run.stop()
            logger.debug(f"run {run_name} is cancelled")
            run = _wait_for_run_status(run, RunStatus.TERMINATING)
            summary = _get_run_summary(run, mlflow_experiment_name)
            display(HTML(summary.to_html(escape=False)))
        button.on_click(on_button_clicked)
    _display_run_summary(_get_run_summary(run, mlflow_experiment_name), button)
    run = _wait_for_run_status(run, RunStatus.RUNNING)

    try_count = 0
    while try_count < 10:
        try_count += 1
        time.sleep(20)
        try:
            run = get_run(run)
            if run.status.is_terminal():
                logger.debug(f"run {run_name} is in terminal state. Status {run.status}")
                break
            summary = _get_run_summary(run, mlflow_experiment_name)
            _display_run_summary(summary, button)
            break
        except ValueError:
             logger.debug(f"waiting for the MLFLow experiment run to be ready, run status{run.status}")
             pass

    if sync:
        logger.debug(f"synchronously waiting for the run to finish.")
        run = _wait_for_run_status(run, RunStatus.TERMINATING, inclusive=False)
        _display_run_summary(_get_run_summary(run, mlflow_experiment_name), None)
    
    return run