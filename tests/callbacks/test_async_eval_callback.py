from unittest.mock import patch

import pytest

from llmfoundry.callbacks import AsyncEval
from llmfoundry.callbacks.async_eval_callback import get_run_name
from mcli import Run, RunConfig, RunStatus

RUN_NAME = 'foo_bar'


def test_get_run_name():
    a = get_run_name('foo', 0)
    assert a == 'eval0-foo'

    b = get_run_name(50 * 'foo', 1)
    assert b == 'eval1-foofoofoofoofoofoofoofoofoofoofoofoofoof'


@pytest.fixture(autouse=True, scope='module')
def set_os_env_vars():
    with patch.dict('os.environ', {
            'MOSAICML_PLATFORM': 'true',
            'RUN_NAME': RUN_NAME
    }):
        yield


def test_fails_when_not_on_platform():
    with patch.dict('os.environ', {'MOSAICML_PLATFORM': 'false'}):
        with pytest.raises(
                Exception,
                match=
                'AsyncEval callback is only supported when running on the MosaicML platform'
        ):
            AsyncEval(interval='2ba')


def test_fails_when_no_run_name():
    with patch.dict('os.environ', {
            'MOSAICML_PLATFORM': 'true',
            'RUN_NAME': ''
    }):
        with pytest.raises(
                Exception,
                match=
                'RUN_NAME environment variable must be set to use the AsyncEval callback'
        ):
            AsyncEval(interval='2ba')


def test_get_eval_parameters():
    with pytest.raises(
            Exception,
            match='Missing the following required parameters for async eval:'):
        AsyncEval.get_eval_parameters(None, {}, RUN_NAME)

    # minimal example
    params = AsyncEval.get_eval_parameters(
        None, {
            'device_eval_batch_size': 2,
            'icl_tasks': 'icl_task_example',
            'max_seq_len': 3,
            'model': {
                'model_example': 'model_example'
            },
            'save_folder': 'save_folder_example',
        }, RUN_NAME)
    assert params == {
        'device_eval_batch_size': 2,
        'icl_tasks': 'icl_task_example',
        'max_seq_len': 3,
        'load_path': 'save_folder_example/latest-rank0.pt',
        'run_name': 'eval0-foo_bar',
        'models': [{
            'model_example': 'model_example'
        }],
    }

    # maximal example
    params2 = AsyncEval.get_eval_parameters(
        None,
        {
            # required
            'device_eval_batch_size': 2,
            'icl_tasks': 'icl_task_example',
            'max_seq_len': 3,
            'model': {
                'model_example': 'model_example'
            },
            'save_folder': 'save_folder_example',
            # optional
            'dist_timeout': 1,
            'eval_gauntlet': 'eval_gauntlet_example',
            'fsdp_config': {
                'fsdp_cfg_example': 'fsdp_cfg_example'
            },
            'icl_subset_num_batches': 4,
            'loggers': {
                'loggers_example': 'loggers_example'
            },
            'precision': 'precision_example',
            'python_log_level': 'debug',
            'seed': 5,
            # ignore this
            'ignore_this': 'ignore_this',
        },
        RUN_NAME)
    assert params2 == {
        'device_eval_batch_size': 2,
        'icl_tasks': 'icl_task_example',
        'max_seq_len': 3,
        'run_name': 'eval0-foo_bar',
        'dist_timeout': 1,
        'models': [{
            'model_example': 'model_example'
        }],
        'eval_gauntlet': 'eval_gauntlet_example',
        'fsdp_dict_cfg': {
            'fsdp_cfg_example': 'fsdp_cfg_example'
        },
        'icl_subset_num_batches': 4,
        'loggers': {
            'loggers_example': 'loggers_example'
        },
        'precision': 'precision_example',
        'python_log_level': 'debug',
        'seed': 5,
        'load_path': 'save_folder_example/latest-rank0.pt'
    }


@patch('llmfoundry.callbacks.async_eval_callback.get_run',
       return_value=Run(
           run_uid='123',
           name=RUN_NAME,
           status=RunStatus.RUNNING,
           created_at='2021-01-01',
           updated_at='2021-01-01',
           created_by='me',
           priority='low',
           preemptible=False,
           retry_on_system_failure=True,
           cluster='c1z2',
           gpu_type="a100",
           gpus=16,
           cpus=0,
           node_count=2,
           latest_resumption=None,
           submitted_config=RunConfig(
               parameters={
                   'device_eval_batch_size': 2,
                   'icl_tasks': 'icl_task_example',
                   'max_seq_len': 3,
                   'model': 'model_example',
                   'save_folder': 'save_folder_example',
               }),
       ))
@patch('llmfoundry.callbacks.async_eval_callback.create_run', return_value=None)
def test_async_eval_callback_minimal(mock_get_run, mock_create_run):
    callback = AsyncEval(interval='2ba')
    assert callback.current_run.name == RUN_NAME
    # todo
