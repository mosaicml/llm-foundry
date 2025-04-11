# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from omegaconf import OmegaConf as om
from transformers import AutoTokenizer

from llmfoundry.data.finetuning.tasks import _DEFAULT_CHAT_TEMPLATE
from llmfoundry.tokenizers.utils import get_date_string


def get_config(conf_path: str = 'scripts/train/yamls/pretrain/mpt-125m.yaml'):
    with open(conf_path) as f:
        test_cfg = om.load(f)
    return test_cfg


@pytest.mark.parametrize(
    'tokenizer_name',
    [
        'EleutherAI/gpt-neox-20b',
        'meta-llama/Meta-Llama-3-8B-Instruct',
        'meta-llama/Meta-Llama-3.1-8B-Instruct',
        'meta-llama/Meta-Llama-3.1-70B-Instruct',
        'meta-llama/Meta-Llama-3.1-405B-Instruct',
    ],
)
@pytest.mark.parametrize('use_date_string', [True, False])
def test_tokenizer_date_string(tokenizer_name: str, use_date_string: bool):
    if 'meta-llama' in tokenizer_name:
        pytest.skip('Model is gated. Skipping test.')

    is_llama_3_1_instruct = 'Meta-Llama-3.1' in tokenizer_name and 'Instruct' in tokenizer_name
    if is_llama_3_1_instruct and use_date_string:
        pytest.skip(
            'Llama 3.1 Instruct models use date_string in chat template already. Skipping test.',
        )

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    messages = [{'role': 'system', 'content': ''}]
    date_string = get_date_string()

    # Manually set a chat template to test if the date_string is being used.
    if use_date_string:
        tokenizer.chat_template = "{%- if not date_string is defined %}\n    {%- set date_string = \"26 Jul 2024\" %}\n{%- endif %}\n{{- \"Today Date: \" + date_string }}\n"

    if not tokenizer.chat_template:
        tokenizer.chat_template = _DEFAULT_CHAT_TEMPLATE

    token_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors='pt',
        date_string=date_string,
    )

    assert isinstance(token_ids, torch.Tensor)
    decoded_text = tokenizer.decode(token_ids.flatten())

    # Only Llama 3.1 instruct family models should use the current date in their chat templates.
    if is_llama_3_1_instruct or use_date_string:
        assert date_string in decoded_text
    else:
        assert date_string not in decoded_text
