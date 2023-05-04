# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
from pathlib import Path

import pytest
from omegaconf import OmegaConf as om
from transformers import AutoTokenizer

from llmfoundry.utils.builders import build_icl_evaluators

TMP_FOLDER = 'tmp_data'


def load_icl_config(conf_path='tests/test_tasks.yaml'):
    with open(conf_path) as f:
        test_cfg = om.load(f)
    return test_cfg


@pytest.fixture(autouse=True)
def cleanup():
    dirpath = Path(TMP_FOLDER)
    if dirpath.exists() and dirpath.is_dir():
        shutil.rmtree(dirpath)
    os.mkdir(TMP_FOLDER)
    yield
    dirpath = Path(TMP_FOLDER)
    if dirpath.exists() and dirpath.is_dir():
        shutil.rmtree(dirpath)


def test_icl_task_loading_gpt2_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    task_cfg = load_icl_config()
    evaluators, _ = build_icl_evaluators(
        task_cfg.icl_tasks,
        tokenizer,
        1024,
        8,
        destination_dir=f'{os.getcwd()}/{TMP_FOLDER}')

    for e in evaluators:
        batch = next(e.dataloader.dataloader.__iter__())

        inputs = batch['input_ids'][0]
        if 'continuation_indices' in batch:
            continuation_indices = list(batch['continuation_indices'][0])
            full_example = tokenizer.decode(inputs[0:continuation_indices[-1]])
            answer = tokenizer.decode(
                inputs[continuation_indices[0]:continuation_indices[-1]])
        else:
            start_idx = (inputs == tokenizer.eos_token_id).tolist().index(False)
            full_example = tokenizer.decode(inputs[start_idx:])
            answer = batch['labels'][0][0]

        if e.label == 'jeopardy/0-shot/american_history':
            assert full_example == 'AMERICAN HISTORY: On May 29, 1765 Patrick Henrys Stamp Act protest was interrupted with this one word\nAnswer: Treason'
            assert answer == ' Treason'
        elif e.label == 'jeopardy/1-shot/american_history':
            assert full_example == 'AMERICAN HISTORY: Witchcraft trials held in this town in 1692 led to the hangings of 19 people\nAnswer: Salem\nAMERICAN HISTORY: On May 29, 1765 Patrick Henrys Stamp Act protest was interrupted with this one word\nAnswer: Treason'
            assert answer == ' Treason'
        elif e.label == 'triviaqa/0-shot':
            assert full_example == 'Question: Who was the man behind The Chipmunks?\nAnswer:'
            assert answer == 'David Seville'
        elif e.label == 'triviaqa/1-shot':
            assert full_example == 'Question: High Willhays is the highest point of what National Park?\nAnswer: DARTMOOR\nQuestion: Who was the man behind The Chipmunks?\nAnswer:'
            assert answer == 'David Seville'
        elif e.label == 'copa/0-shot':
            assert full_example == 'The man turned on the faucet, therefore the toilet filled with water'
            assert answer == ' the toilet filled with water'
        elif e.label == 'copa/1-shot':
            assert full_example == 'The woman was in a bad mood, therefore she told her friend to leave her alone.\nThe man turned on the faucet, therefore the toilet filled with water'
            assert answer == ' the toilet filled with water'
        elif e.label == 'winograd/0-shot':
            assert full_example == 'The city councilmen refused the demonstrators a permit because the city councilmen feared violence'
            assert answer == ' feared violence'
        elif e.label == 'winograd/1-shot':
            assert full_example == "Tom gave Ralph a lift to school so Ralph wouldn't have to walk.\nThe city councilmen refused the demonstrators a permit because the city councilmen feared violence"
            assert answer == ' feared violence'


# def test_icl_task_loading_gptj_tokenizer():
#     tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-j-6b')
#     task_cfg = load_icl_config()
#     evaluators, _ = build_icl_evaluators(task_cfg, tokenizer)
#     breakpoint()

# def test_icl_task_loading_opt_tokenizer():
#     tokenizer = AutoTokenizer.from_pretrained('facebook/opt-6.7b')
#     task_cfg = load_icl_config()
#     evaluators, _ = build_icl_evaluators(task_cfg, tokenizer)
#     breakpoint()

# def test_icl_task_loading_sentencepiece_tokenizer():
#     tokenizer = AutoTokenizer.from_pretrained('huggyllama/llama-7b')
#     task_cfg = load_icl_config()
#     evaluators, _ = build_icl_evaluators(task_cfg, tokenizer)
#     breakpoint()

# def test_icl_task_loading_gptneox_tokenizer():
#     tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')
#     task_cfg = load_icl_config()
#     evaluators, _ = build_icl_evaluators(task_cfg, tokenizer)
#     breakpoint()
