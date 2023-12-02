# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import datetime
from unittest.mock import MagicMock, patch

import pytest

from llmfoundry.callbacks.async_eval_callback import (AsyncEval,
                                                      get_eval_parameters,
                                                      get_run_name)
from mcli import Run, RunConfig, RunStatus

RUN_NAME = 'foo_bar-1234'
BASIC_PARAMS = {
    'device_eval_batch_size': 2,
    'icl_tasks': 'icl_task_example',
    'max_seq_len': 3,
    'model': {
        'name': 'model_example',
        'config_overrides': {
            'attn_config': {
                'foo': 'bar'
            }
        }
    },
    'tokenizer': {
        'tokenizer_example': 'tokenizer_example',
    },
}


def test_get_run_name():
    a = get_run_name('foo-1234', '1ba')
    assert a == 'eval-1ba-foo'

    # Run name should be truncated
    b = get_run_name(50 * 'foo' + '-1234', '1ba')
    assert b == 'eval-1ba-foofoofoofoofoofoofoofoofoofoofoofoofoofoofoof'


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
            AsyncEval(BASIC_PARAMS, interval='2ba')


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
            AsyncEval(BASIC_PARAMS, interval='2ba')


def test_get_eval_parameters():
    with pytest.raises(
            Exception,
            match='Missing the following required parameters for async eval:'):
        get_eval_parameters({}, 'checkpoints/file', RUN_NAME)

    # minimal example
    params = get_eval_parameters(BASIC_PARAMS, 'checkpoints/file', RUN_NAME)
    assert params == {
        'device_eval_batch_size':
            2,
        'icl_tasks':
            'icl_task_example',
        'max_seq_len':
            3,
        'load_path':
            'checkpoints/file',
        'models': [{
            'model_name': 'model_example',
            'model': {
                'name': 'model_example',
                'config_overrides': {
                    'attn_config': {
                        'foo': 'bar'
                    },
                },
            },
            'tokenizer': {
                'tokenizer_example': 'tokenizer_example'
            },
        }],
    }

    # maximal example
    params2 = get_eval_parameters(
        {
            # required
            **BASIC_PARAMS,
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
        'checkpoints/file',
        RUN_NAME,
    )
    assert params2 == {
        'device_eval_batch_size': 2,
        'icl_tasks': 'icl_task_example',
        'max_seq_len': 3,
        'dist_timeout': 1,
        'models': [{
            'model_name': 'model_example',
            'model': {
                'name': 'model_example',
                'config_overrides': {
                    'attn_config': {
                        'foo': 'bar'
                    },
                },
            },
            'tokenizer': {
                'tokenizer_example': 'tokenizer_example'
            },
        }],
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
        'load_path': 'checkpoints/file',
    }


FAKE_RUN = Run(
    run_uid='123',
    name=RUN_NAME,
    image='fake-image',
    status=RunStatus.RUNNING,
    created_at=datetime.datetime(2021, 1, 1),
    updated_at=datetime.datetime(2021, 1, 1),
    created_by='me',
    priority='low',
    preemptible=False,
    retry_on_system_failure=True,
    cluster='c1z2',
    gpu_type='a100',
    gpus=16,
    cpus=0,
    node_count=2,
    latest_resumption=None,  # type: ignore
    submitted_config=RunConfig(
        name=RUN_NAME,
        image='fake-image',
        command='echo hi',
        parameters={},
    ),
)


@patch('llmfoundry.callbacks.async_eval_callback.get_run',
       return_value=FAKE_RUN)
@patch('llmfoundry.callbacks.async_eval_callback.create_run',
       return_value=FAKE_RUN)
def test_async_eval_callback_minimal(mock_create_run: MagicMock,
                                     mock_get_run: MagicMock):
    callback = AsyncEval(BASIC_PARAMS,
                         interval='2ba',
                         compute={
                             'cluster': 'c2z3',
                             'nodes': 2,
                         })
    assert callback.current_run.name == RUN_NAME
    assert mock_get_run.call_count == 1
    assert mock_get_run.call_args[0][0] == RUN_NAME

    callback.launch_run('checkpoint/path', '1ba')
    assert mock_create_run.call_count == 1

    run_config_created = mock_create_run.call_args[0][0]
    assert run_config_created.name == 'eval-1ba-foo_bar'
    assert run_config_created.image == 'fake-image'
    assert run_config_created.command

    compute = run_config_created.compute
    assert compute['cluster'] == 'c2z3'
    assert compute['nodes'] == 2

    parameters = run_config_created.parameters
    assert parameters['device_eval_batch_size'] == 2
    assert parameters['icl_tasks'] == 'icl_task_example'
    assert parameters['max_seq_len'] == 3
    assert parameters['load_path'] == 'checkpoint/path'
    assert parameters['models'] == [{
        'model_name': 'model_example',
        'model': {
            'name': 'model_example',
            'config_overrides': {
                'attn_config': {
                    'foo': 'bar'
                },
            },
        },
        'tokenizer': {
            'tokenizer_example': 'tokenizer_example'
        }
    }]
    assert parameters['run_name'] == 'eval-1ba-foo_bar'  # original run
