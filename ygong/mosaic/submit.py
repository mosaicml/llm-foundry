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

logger = logging.getLogger('ygong.mosaic.submit')

        
def _set_up_environment(content: str):
    os.environ['CREDENTIALS'] = content

     
def _init_connection():
     def _is_local():
        try:
            wc = WorkspaceClient()
            wc.dbutils.entry_point.getDbutils().notebook().getContext()
            return False
        except:
            return True
        
     if _is_local():
        if os.environ.get('CREDENTIALS') is None:
            raise ValueError("_set_up_environment must be manually called to configure credentials for local runs")
        data = json.loads(base64.b64decode(os.environ.get('CREDENTIALS')).decode('utf-8'))
        workspace_url = data.get("workspace_url", None)
        token = data.get("token", None)
        # set up the mosaic token
        os.environ[config.MCLI_MODE_ENV] = config.MCLIMode.DBX_AWS_STAGING.value
        os.environ[config.MOSAICML_ACCESS_TOKEN_FILE_ENV] = "/home/shitao.li/e2_token"
     else:
        wc = WorkspaceClient()
        import mlflow.utils.databricks_utils as databricks_utils
        workspace_url = databricks_utils.get_workspace_info_from_dbutils()[0]
        ctx = wc.dbutils.entry_point.getDbutils().notebook().getContext()
        token = ctx.apiToken().get()
        api_url = ctx.apiUrl().get()
        endpoint = f'{api_url}/api/2.0/genai-mapi/graphql'
        os.environ[config.MOSAICML_API_KEY_ENV] = f'Bearer {token}'
        os.environ[config.MOSAICML_API_ENDPOINT_ENV] = endpoint

     # needed to set up the MLFlow query for experiment runs   
     os.environ['WORKSPACE_URL'] = workspace_url
     os.environ['MLFLOW_TRACKING_TOKEN'] = token
     

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

def submit(model, config: any, scalingConfig: ScalingConfig, sync: bool = False, debug: bool = False):
    if debug:
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(logging.DEBUG)  # Set minimum log level for the handler
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        stdout_handler.setFormatter(formatter)

        # Add the handler to the logger
        logger.addHandler(stdout_handler)
        logger.setLevel(logging.DEBUG)
        
        logger.info("set the logger to debug mode")

    _init_connection()
    mlflow_experiment_name = None
    if model == "mpt125m":
        if not isinstance(config, MPT125MConfig):
            raise ValueError("config must be an instance of MPT125MConfig")
        mlflow_experiment_name = config.mlflow_experimentName
        runConfig = config.toRunConfig(scalingConfig)
    else:
        raise ValueError(f"model {model} is not supported")
    
    
    run = create_run(runConfig)
    run_name = run.name
    # Create a button
    button = widgets.Button(description="cancel the run")
    def on_button_clicked(b):
        clear_output(wait=False)
        run = get_run(run_name)
        run.stop()
        logger.debug(f"run {run_name} is cancelled")
        run = _wait_for_run_status(run, RunStatus.TERMINATING)
        summary = _get_run_summary(run, mlflow_experiment_name)
        display(HTML(summary.to_html(escape=False)))
    button.on_click(on_button_clicked)

    def _wait_for_run_status(run: Run, status: RunStatus, inclusive: bool = True):
        run_name = run.name
        while not run.status.after(status, inclusive=inclusive) and not run.status.is_terminal():
            run =  get_run(run_name)
            # setting mlflow_experiment_name to be None, since its very likely mlflow run isn't ready yet
            # when the run just starts running
            _display_run_summary(_get_run_summary(run, None), button)
            time.sleep(5)
        logger.debug(f"finish waiting run reached expected status {status}")
        return run

    def _wait_for_run_finish(run: Run):
        run_name = run.name
        while not run.status.is_terminal():
            run =  get_run(run_name)
            _display_run_summary(_get_run_summary(run, mlflow_experiment_name), button)
            time.sleep(5)
        logger.debug(f"finish waiting run reached terminal")
        return run

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
        run = _wait_for_run_finish(run)
        _display_run_summary(_get_run_summary(run, mlflow_experiment_name), None)
    
    return run