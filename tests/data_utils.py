# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import json
import os
import shutil
from argparse import Namespace
from pathlib import Path
from typing import Dict, List, Optional

from omegaconf import DictConfig
from omegaconf import OmegaConf as om

from llmfoundry.data_prep import convert_dataset_json  # noqa: E402
from scripts.data_prep.convert_dataset_hf import main as main_hf  # noqa: E402


def make_tiny_ft_dataset(
    path: str,
    size: int = 4,
    add_bad_data_dropped: bool = False,
    add_invalid_prompt_type: bool = False,
    add_invalid_response_type: bool = False,
    add_unknown_example_type: bool = False,
    add_just_bos_eos_pad: bool = False,
    add_too_many_example_keys: bool = False,
    pad_token: Optional[str] = None,
    start_token: Optional[str] = None,
    end_token: Optional[str] = None,
):
    if Path(path).suffix != '.jsonl':
        raise ValueError(f'Path {path} must be a jsonl file.')
    good_sample = {'prompt': 'hello', 'response': 'goodbye'}
    samples = [good_sample] * size
    if add_bad_data_dropped:
        if pad_token is None:
            raise ValueError(
                'pad_token, start_token, and end_token must be specified if add_bad_data is True',
            )
        # empty prompt
        samples.append({'prompt': '', 'response': 'goodbye'})
        # empty response
        samples.append({'prompt': 'hello', 'response': ''})

    if add_invalid_prompt_type:
        # prompt just None
        samples.append({
            'prompt': None,
            'response': 'goodbye',
        })  # type: ignore (intentional test)

    if add_invalid_response_type:
        # response just None
        samples.append({
            'prompt': 'hello',
            'response': None,
        })  # type: ignore (intentional test)

    if add_too_many_example_keys:
        # too many keys
        samples.append({
            'prompt': 'hello',
            'response': 'goodbye',
            'completion': 'bar',
        })

    if add_just_bos_eos_pad:
        if pad_token is None or start_token is None or end_token is None:
            raise ValueError(
                'pad_token, start_token, and end_token must be specified if add_just_bos_eos is True',
            )
        # prompt just start
        samples.append({'prompt': start_token, 'response': 'goodbye'})
        # response just start
        samples.append({'prompt': 'hello', 'response': start_token})
        # prompt just end
        samples.append({'prompt': end_token, 'response': 'goodbye'})
        # response just end
        samples.append({'prompt': 'hello', 'response': end_token})
        # prompt just pad
        samples.append({'prompt': pad_token, 'response': 'goodbye'})
    if add_unknown_example_type:
        # unknown example type
        samples = [{'foo': 'yee', 'bar': 'haw'}]

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as _f:
        for sample in samples:
            _f.write(json.dumps(sample))
            _f.write('\n')


def make_tiny_conversation_ft_dataset(
    size: int,
    path: str,
    add_invalid_last_chat_message: bool = False,
    add_invalid_message_key_quantity: bool = False,
    add_invalid_content_type: bool = False,
    add_invalid_role: bool = False,
    add_not_alternating_roles: bool = False,
    use_messages_format: bool = True,
):
    if Path(path).suffix != '.jsonl':
        raise ValueError(f'Path {path} must be a jsonl file.')
    good_sample = {
        'messages': [{
            'role': 'system',
            'content': 'A conversation between a user and a helpful assistant.',
        }, {
            'role': 'user',
            'content': "Hi there. What's the capital of the moon?",
        }, {
            'role': 'assistant',
            'content': "This question doesn't make sense.",
        }],
    }

    samples = [good_sample] * size

    if add_invalid_last_chat_message:
        # invalid last chat message
        samples.append({
            'messages': [{
                'role':
                    'system',
                'content':
                    'A conversation between a user and a helpful assistant.',
            }, {
                'role': 'user',
                'content': "Hi there. What's the capital of the moon?",
            }, {
                'role': 'system',
                'content': "This question doesn't make sense.",
            }],
        })

    if add_invalid_message_key_quantity:
        # invalid message key quantity
        samples.append({
            'messages': [{
                'role':
                    'system',
                'content':
                    'A conversation between a user and a helpful assistant.',
                'extra_key':
                    'extra value',
            }],
        })

    if add_invalid_role:
        # invalid role
        samples.append({
            'messages': [{
                'role':
                    'system',
                'content':
                    'A conversation between a user and a helpful assistant.',
            }, {
                'role': 'foo',
                'content': "Hi there. What's the capital of the moon?",
            }, {
                'role': 'assistant',
                'content': "This question doesn't make sense.",
            }],
        })

    if add_invalid_content_type:
        # invalid conversation type
        samples.append({
            'messages': [{
                'role':
                    'system',
                'content':
                    'A conversation between a user and a helpful assistant.',
            }, {
                'role': 'user',
                'content': "Hi there. What's the capital of the moon?",
            }, {
                'role': 'assistant',
                'content': None,
            }],
        })  # type: ignore (intentional test)

    if add_not_alternating_roles:
        # not alternating roles
        samples.append({
            'messages': [{
                'role':
                    'system',
                'content':
                    'A conversation between a user and a helpful assistant.',
            }, {
                'role': 'assistant',
                'content': "Hi there. What's the capital of the moon?",
            }, {
                'role': 'assistant',
                'content': "This question doesn't make sense.",
            }],
        })

    def messages_to_conversation(sample: Dict):
        assert 'messages' in sample
        messages = sample['messages']

        role_map = {
            'user': 'human',
            'assistant': 'gpt',
        }
        conversations: List[Dict[str, str]] = []
        for message in messages:
            role: str = role_map.get(message['role'], message['role'])
            content: str = message['content']
            conversations.append({'from': role, 'value': content})
        return {'conversations': conversations}

    if not use_messages_format:
        samples = [messages_to_conversation(sample) for sample in samples]

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as _f:
        for sample in samples:
            _f.write(json.dumps(sample))
            _f.write('\n')


def create_c4_dataset_xxsmall(path: Path) -> str:
    """Creates a small mocked version of the C4 dataset."""
    c4_dir = os.path.join(path, f'my-copy-c4')
    downloaded_split = 'val_xxsmall'  # very fast to convert

    # Hyperparameters from https://github.com/mosaicml/llm-foundry/blob/340a56658560ebceb2a3aa69d6e37813e415acd0/README.md#L188
    main_hf(
        Namespace(
            **{
                'dataset': 'c4',
                'data_subset': 'en',
                'splits': [downloaded_split],
                'out_root': c4_dir,
                'compression': None,
                'concat_tokens': 2048,
                'tokenizer': 'EleutherAI/gpt-neox-20b',
                'tokenizer_kwargs': {},
                'bos_text': '',
                'eos_text': '<|endoftext|>',
                'no_wrap': False,
                'num_workers': 8,
            },
        ),
    )

    # copy the small downloaded_split to other c4 splits for mocking purposes
    mocked_splits = ['train', 'val']
    for mocked_split in mocked_splits:
        shutil.copytree(
            os.path.join(c4_dir, 'val_xxsmall'),
            os.path.join(c4_dir, mocked_split),
        )
    assert os.path.exists(c4_dir)
    return c4_dir


def create_arxiv_dataset(path: Path) -> str:
    """Creates an arxiv dataset."""
    arxiv_dir = os.path.join(path, f'my-copy-arxiv')
    downloaded_split = 'train'

    arxiv_path = 'data_prep/example_data/arxiv.jsonl'
    if not os.getcwd().endswith('scripts'):
        arxiv_path = os.path.join('scripts', arxiv_path)

    convert_dataset_json(
        Namespace(
            **{
                'path': arxiv_path,
                'out_root': arxiv_dir,
                'compression': None,
                'split': downloaded_split,
                'concat_tokens': None,
                'bos_text': None,
                'eos_text': None,
                'no_wrap': False,
                'num_workers': None,
            },
        ),
    )

    return arxiv_dir


def gpt_tiny_cfg(dataset_name: str, device: str):
    """Create gpt tiny cfg."""
    from tests.fixtures.autouse import REPO_DIR
    conf_path: str = os.path.join(
        REPO_DIR,
        'scripts/train/yamls/pretrain/testing.yaml',
    )
    with open(conf_path) as f:
        test_cfg = om.load(f)
    assert isinstance(test_cfg, DictConfig)

    test_cfg.variables.data_local = dataset_name
    test_cfg.global_train_batch_size = 8
    test_cfg.device_eval_batch_size = 4
    test_cfg.device_train_microbatch_size = 4
    test_cfg.max_duration = '4ba'
    test_cfg.eval_interval = '4ba'
    test_cfg.variables.run_name = 'gpt-mini-integration-test'

    if device == 'cpu':
        test_cfg.model.init_device = 'cpu'
        test_cfg.fsdp_config = None
        test_cfg.model.attn_config.attn_impl = 'torch'
        test_cfg.model.loss_fn = 'torch_crossentropy'
        test_cfg.precision = 'fp32'

    return test_cfg
