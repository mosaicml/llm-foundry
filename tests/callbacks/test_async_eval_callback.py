# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import datetime
from copy import deepcopy
from unittest.mock import MagicMock, patch

import pytest
from composer.core import Time, Timestamp, TimeUnit

from llmfoundry.callbacks.async_eval_callback import (AsyncEval,
                                                      get_eval_parameters,
                                                      get_run_name,
                                                      validate_eval_run_config,
                                                      validate_interval)
from mcli import Run, RunConfig, RunStatus

RUN_NAME = 'foo_bar-1234'
BASIC_PARAMS = {
    'save_interval': '1ba',
    'save_folder': 'foobar',
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
            'load_path': 'checkpoints/file',
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
                'wandb': {
                    'init_kwargs': {
                        'fee': 'bee'
                    }
                }
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
            'load_path': 'checkpoints/file',
        }],
        'eval_gauntlet': 'eval_gauntlet_example',
        'fsdp_config': {
            'fsdp_cfg_example': 'fsdp_cfg_example'
        },
        'icl_subset_num_batches': 4,
        'loggers': {
            'wandb': {
                'group': 'foo_bar-1234',
                'init_kwargs': {
                    'fee': 'bee'
                },
            }
        },
        'precision': 'precision_example',
        'python_log_level': 'debug',
        'seed': 5,
    }


def test_validate_interval():
    with pytest.raises(ValueError):
        validate_interval('1ba', '1ep')  # different units
    with pytest.raises(ValueError):
        validate_interval('1ba', '2ba')  # checkpointing happens less often
    with pytest.raises(ValueError):
        validate_interval('3ba', '2ba')  # not a multiple

    assert validate_interval('2ba', '1ba') == Time(2, TimeUnit.BATCH)
    two_epochs = Time(2, TimeUnit.EPOCH)
    assert validate_interval(2, 2) == two_epochs
    assert validate_interval(two_epochs, two_epochs) == two_epochs
    assert validate_interval('2ep', two_epochs) == two_epochs


def test_validate_eval_run_config():
    assert validate_eval_run_config(None) == {}
    assert validate_eval_run_config({}) == {}

    with pytest.raises(ValueError):
        validate_eval_run_config({'foo': 'bar'})

    valid_config = {
        'image': 'example_image',
        'command': 'example_command',
        'compute': {
            'gpus': 1,
            'cluster': 'example_cluster',
        },
        'scheduling': {
            'priority': 'high',
            'preemptible': True,
        },
    }
    res = validate_eval_run_config(valid_config)
    assert res == valid_config


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
    callback = AsyncEval(
        BASIC_PARAMS,
        interval='2ba',
        eval_run_config={
            'compute': {
                'cluster': 'c2z3',
                'nodes': 2,
            },
        },
    )
    assert callback.current_run.name == RUN_NAME
    assert mock_get_run.call_count == 1
    assert mock_get_run.call_args[0][0] == RUN_NAME

    launch_time = Time(1, TimeUnit.BATCH)
    callback.launch_run('checkpoint/path', launch_time)
    assert mock_create_run.call_count == 1

    run_config_created = mock_create_run.call_args[0][0]
    assert run_config_created.name == 'eval-1ba-foo_bar'
    assert run_config_created.image == 'fake-image'

    metadata = run_config_created.metadata
    assert 'eval_timestamp' in metadata
    assert isinstance(metadata['eval_timestamp'], int)
    assert metadata['eval_timestamp'] == launch_time.value

    assert 'eval_timestamp_unit' in metadata
    assert isinstance(metadata['eval_timestamp_unit'], str)
    assert metadata['eval_timestamp_unit'] == launch_time.unit.value

    assert 'cd llm-foundry/scripts' in run_config_created.command

    integrations = run_config_created.integrations
    assert len(integrations) == 1
    assert integrations[0]['integration_type'] == 'git_repo'
    assert integrations[0]['git_repo'] == 'mosaicml/llm-foundry'
    assert integrations[0]['git_branch'].startswith('v')

    compute = run_config_created.compute
    assert compute['cluster'] == 'c2z3'
    assert compute['nodes'] == 2

    parameters = run_config_created.parameters
    assert parameters['device_eval_batch_size'] == 2
    assert parameters['icl_tasks'] == 'icl_task_example'
    assert parameters['max_seq_len'] == 3
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
        },
        'load_path': 'checkpoint/path',
    }]
    assert parameters['run_name'] == 'eval-1ba-foo_bar'  # original run


@patch('llmfoundry.callbacks.async_eval_callback.get_run',
       return_value=FAKE_RUN)
def test_async_eval_state(mock_create_run: MagicMock):
    callback = AsyncEval(BASIC_PARAMS, interval='2ba')

    assert not callback.checkpoints_evaled

    state_dict = callback.state_dict()
    assert state_dict['checkpoints_evaled'] == []

    callback.load_state_dict(state_dict)
    assert not callback.checkpoints_evaled

    callback.checkpoints_evaled = {
        Time(1, TimeUnit.BATCH): ('checkpoint/path', 'run-name'),
    }
    state_dict = callback.state_dict()
    assert state_dict['checkpoints_evaled'] == [
        (
            {
                'value': 1,
                'unit': 'ba',
            },
            'checkpoint/path',
            'run-name',
        ),
    ]

    callback.checkpoints_evaled = {}
    callback.load_state_dict(state_dict)
    assert callback.checkpoints_evaled == {
        Time(1, TimeUnit.BATCH): ('checkpoint/path', 'run-name'),
    }


INTEGRATION_GIT_LLMFOUNDRY = {
    'integration_type': 'git_repo',
    'git_repo': 'mosaicml/llm-foundry',
    'git_branch': 'custom_branch',
    'path': 'custom/llm-foundry',
    'pip_install': '-e .[gpu]',
    'ssh_clone': False,
}
INTEGRATION_GIT_RANDOM = {
    'integration_type': 'git_repo',
    'git_repo': 'another-repo',
    'git_branch': 'foobar',
}

FAKE_RUN_WITH_INTEGRATIONS = deepcopy(FAKE_RUN)
FAKE_RUN_WITH_INTEGRATIONS.submitted_config.integrations = [
    INTEGRATION_GIT_LLMFOUNDRY, INTEGRATION_GIT_RANDOM
]


@patch('llmfoundry.callbacks.async_eval_callback.get_run',
       return_value=FAKE_RUN_WITH_INTEGRATIONS)
@patch('llmfoundry.callbacks.async_eval_callback.create_run',
       return_value=FAKE_RUN_WITH_INTEGRATIONS)
def test_async_eval_callback_integrations(mock_create_run: MagicMock,
                                          mock_get_run: MagicMock):
    callback = AsyncEval(
        BASIC_PARAMS,
        interval='2ba',
        eval_run_config={'compute': {
            'cluster': 'c2z3',
            'nodes': 2,
        }})
    assert mock_get_run.call_count == 1

    callback.launch_run('checkpoint/path', Time(1, TimeUnit.BATCH))
    assert mock_create_run.call_count == 1
    run_config_created = mock_create_run.call_args[0][0]

    assert len(run_config_created.integrations) == 2
    # order should be retained
    assert run_config_created.integrations[0] == INTEGRATION_GIT_LLMFOUNDRY
    assert run_config_created.integrations[1] == INTEGRATION_GIT_RANDOM

    custom_path = run_config_created.integrations[0]['path']
    assert f'cd {custom_path}/scripts' in run_config_created.command


@patch('llmfoundry.callbacks.async_eval_callback.dist.get_world_size',
       return_value=4)
def test_get_ready_sharded_checkpoints(mocked_get_world_size: MagicMock):
    assert not AsyncEval._get_ready_sharded_checkpoints({}, [])
    assert not AsyncEval._get_ready_sharded_checkpoints(
        {'save_folder/ep0-ba1/__0_0.distcp': Timestamp(epoch=0, batch=1)},
        [],
    )
    assert not AsyncEval._get_ready_sharded_checkpoints(
        {},
        ['save_folder/ep0-ba1/__0_0.distcp'],
    )

    checkpointer_checkpoints = {
        'save_folder/ep0-ba1/__0_0.distcp': Timestamp(epoch=0, batch=1),
        'save_folder/ep0-ba2/__0_0.distcp': Timestamp(epoch=0, batch=2),
        'save_folder/ep0-ba3/__0_0.distcp': Timestamp(epoch=0, batch=3),
    }
    remote_files = [
        # ba1 is ready
        'save_folder/ep0-ba1/__0_0.distcp',
        'save_folder/ep0-ba1/__1_0.distcp',
        'save_folder/ep0-ba1/__2_0.distcp',
        'save_folder/ep0-ba1/__3_0.distcp',
        'save_folder/ep0-ba1/.metadata',
        # ba2 is missing shard 2
        'save_folder/ep0-ba2/__0_0.distcp',
        'save_folder/ep0-ba2/__1_0.distcp',
        'save_folder/ep0-ba2/__3_0.distcp',
        'save_folder/ep0-ba2/.metadata',
        # ba3 is missing metadata
        'save_folder/ep0-ba3/__0_0.distcp',
        'save_folder/ep0-ba3/__1_0.distcp',
        'save_folder/ep0-ba3/__2_0.distcp',
        'save_folder/ep0-ba3/__3_0.distcp',
    ]
    res = AsyncEval._get_ready_sharded_checkpoints(
        checkpointer_checkpoints,
        remote_files,
    )
    assert res == {'ep0-ba1': Timestamp(epoch=0, batch=1)}


def test_get_ready_single_checkpoints():
    assert not AsyncEval._get_ready_single_checkpoints({}, [])
    assert not AsyncEval._get_ready_single_checkpoints(
        {'save_folder/ep0-ba1-rank0.pt': Timestamp(epoch=0, batch=1)},
        [],
    )
    assert not AsyncEval._get_ready_single_checkpoints(
        {},
        ['save_folder/ep0-ba1-rank0.pt'],
    )

    checkpointer_checkpoints = {
        'save_folder/ep0-ba1-rank0.pt': Timestamp(epoch=0, batch=1),
        'save_folder/ep0-ba2-rank0.pt': Timestamp(epoch=0, batch=2),
    }
    remote_checkpoints = [
        'save_folder/ep0-ba1-rank0.pt',
    ]

    res = AsyncEval._get_ready_single_checkpoints(
        checkpointer_checkpoints,
        remote_checkpoints,
    )
    assert res == {'ep0-ba1-rank0.pt': Timestamp(epoch=0, batch=1)}
