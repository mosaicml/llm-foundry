# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import pathlib

import pytest
from omegaconf import OmegaConf as om
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from llmfoundry.utils.builders import build_icl_evaluators
from llmfoundry.utils.config_utils import to_list_container

EXPECTED_FIRST_DATALOADER_LEN = 52  # scripts/eval/local_data/world_knowledge/jeopardy_all.jsonl


def load_icl_config(conf_path: str = 'tests/data/test_tasks.yaml'):
    with open(conf_path) as f:
        test_cfg = om.load(f)
    return test_cfg


def run_test(
    dir: pathlib.Path,
    tokenizer: PreTrainedTokenizerBase,
    bos_tok: str = '',
    eval_drop_last: bool = False,
):
    task_cfg = load_icl_config()
    evaluators, _ = build_icl_evaluators(
        to_list_container(task_cfg.icl_tasks),
        tokenizer,
        1024,
        default_batch_size=8,
        destination_dir=str(dir),
        eval_drop_last=eval_drop_last,
    )
    if eval_drop_last:
        # Drop the evaluator for eval/local_data/symbolic_problem_solving/bigbench_operators.jsonl as it won't evenly divide even 1 batch.
        assert len(evaluators) == 17
    else:
        assert len(evaluators) == 17

    for i, e in enumerate(evaluators):
        batch = next(e.dataloader.dataloader.__iter__())
        # Check that the dataloader is the correct length for the first task.
        if i == 0:
            if eval_drop_last:
                assert len(
                    e.dataloader.dataloader,
                ) == EXPECTED_FIRST_DATALOADER_LEN - 1  # pyright: ignore
            else:
                assert len(
                    e.dataloader.dataloader,
                ) == EXPECTED_FIRST_DATALOADER_LEN  # pyright: ignore

        inputs = batch['input_ids'][0]
        if 'continuation_indices' in batch:
            continuation_indices = list(batch['continuation_indices'][0])
            full_example = tokenizer.decode(inputs[0:continuation_indices[-1]])
            answer = tokenizer.decode(
                inputs[continuation_indices[0]:continuation_indices[-1]],
            )
        else:
            if tokenizer.pad_token_id is not None:
                start_idx = (inputs == tokenizer.pad_token_id
                            ).tolist().index(False)
            else:
                start_idx = (inputs == tokenizer.eos_token_id
                            ).tolist().index(False)
            full_example = tokenizer.decode(inputs[start_idx:])
            answer = batch['labels'][0][0]

        if e.label == 'jeopardy/0-shot/american_history':
            assert full_example == bos_tok + 'AMERICAN HISTORY: On May 29, 1765 Patrick Henrys Stamp Act protest was interrupted with this one word\nAnswer: Treason'
            assert answer == ' Treason'
        elif e.label == 'jeopardy/1-shot/american_history':
            assert full_example == bos_tok + 'AMERICAN HISTORY: Witchcraft trials held in this town in 1692 led to the hangings of 19 people\nAnswer: Salem\nAMERICAN HISTORY: On May 29, 1765 Patrick Henrys Stamp Act protest was interrupted with this one word\nAnswer: Treason'
            assert answer == ' Treason'
        elif e.label == 'triviaqa/0-shot':
            assert full_example == bos_tok + 'Question: Who was the man behind The Chipmunks?\nAnswer:'
            assert answer == 'David Seville'
        elif e.label == 'triviaqa/1-shot':
            assert full_example == bos_tok + 'Question: Which was the only eastern bloc country to participate in the 1984 LA Olympics?\nAnswer: Rumania\nQuestion: Who was the man behind The Chipmunks?\nAnswer:'
            assert answer == 'David Seville'
        elif e.label == 'copa/0-shot':
            assert full_example == bos_tok + 'The man turned on the faucet, therefore the toilet filled with water'
            assert answer == ' the toilet filled with water'
        elif e.label == 'copa/1-shot':
            assert full_example == bos_tok + 'The woman was in a bad mood, therefore she told her friend to leave her alone.\nThe man turned on the faucet, therefore the toilet filled with water'
            assert answer == ' the toilet filled with water'
        elif e.label == 'winograd/0-shot':
            assert full_example == bos_tok + 'The city councilmen refused the demonstrators a permit because the city councilmen feared violence'
            assert answer == ' feared violence'
        elif e.label == 'winograd/1-shot':
            assert full_example == bos_tok + "Tom gave Ralph a lift to school so Ralph wouldn't have to walk.\nThe city councilmen refused the demonstrators a permit because the city councilmen feared violence"
            assert answer == ' feared violence'


@pytest.mark.parametrize(
    'tokenizer_name,bos_token,eval_drop_last',
    [('facebook/opt-6.7b', '</s>', True),
     ('EleutherAI/gpt-neox-20b', '', False)],
)
def test_icl_task_tokenizer_and_dataloader(
    tmp_path: pathlib.Path,
    tokenizer_name: str,
    bos_token: str,
    eval_drop_last: bool,
):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    run_test(tmp_path, tokenizer, bos_token, eval_drop_last)
