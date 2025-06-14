import torch
import pytest

from pathlib import Path
from composer.loggers import MLFlowLogger

from llmfoundry.callbacks.hf_checkpointer import _move_mlflow_logger_to_cpu

@pytest.mark.gpu
def test_move_mlflow_logger_to_cpu(
    tmp_path: Path,
    
):
    logger = MLFlowLogger(
        tracking_uri=tmp_path / Path('my-test-mlflow-uri'),
    )

    metric_tensor = torch.tensor(1.0, device='cuda')
    metrics_dict = {'test_metric': metric_tensor}

    logger.log_metrics(metrics_dict)

    # Tensor is expected to be on GPU intially
    assert logger._metrics_cache['test_metric'][0].device.type == 'cuda'

    # Move the logger's metrics to CPU
    _move_mlflow_logger_to_cpu(logger)

    # Check that the tensor is now on CPU
    assert logger._metrics_cache['test_metric'][0].device.type == 'cpu'