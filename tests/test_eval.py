# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import os
import sys
from typing import Any

import omegaconf as om
import pytest

# Add repo root to path so we can import scripts and test it
repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(repo_dir)

from scripts.eval.eval import main  # noqa: E402


@pytest.fixture(autouse=True)
def set_correct_cwd():
    if not os.getcwd().endswith('llm-foundry/scripts'):
        os.chdir('scripts')

    yield

    if os.getcwd().endswith('llm-foundry/scripts'):
        os.chdir('..')


def test_icl_eval(capfd: Any):
    test_cfg = om.OmegaConf.create("""
        max_seq_len: 1024
        seed: 1
        precision: fp32
        models:
        -
            model_name: tiny_mpt
            model:
                name: mpt_causal_lm
                init_device: meta
                d_model: 128
                n_heads: 2
                n_layers: 2
                expansion_ratio: 4
                max_seq_len: ${max_seq_len}
                vocab_size: 50368
                attn_config:
                    attn_impl: torch
                loss_fn: torch_crossentropy
            # Tokenizer
            tokenizer:
                name: EleutherAI/gpt-neox-20b
                kwargs:
                    model_max_length: ${max_seq_len}

        device_eval_batch_size: 4
        icl_subset_num_batches: 1
        icl_tasks:
        -
            label: lambada_openai
            dataset_uri: eval/local_data/language_understanding/lambada_openai.jsonl
            num_fewshot: [0]
            icl_task_type: language_modeling
        eval_gauntlet:
            weighting: EQUAL
            subtract_random_baseline: true
            rescale_accuracy: true
            categories:
            - name: language_understanding_lite
              benchmarks:
                - name: lambada_openai
                  num_fewshot: 0
                  random_baseline: 0.0
        """)
    assert isinstance(test_cfg, om.DictConfig)
    main(test_cfg)
    out, _ = capfd.readouterr()
    expected_results = '| Category                    | Benchmark      | Subtask   |   Accuracy | Number few shot   | Model    |\n|:----------------------------|:---------------|:----------|-----------:|:------------------|:---------|\n| language_understanding_lite | lambada_openai |           |          0 | 0-shot            | tiny_mpt '
    assert expected_results in out
    expected_results = '| model_name   |   average |   language_understanding_lite |\n|:-------------|----------:|------------------------------:|\n| tiny_mpt     |         0 |                             0 |'
    assert expected_results in out
