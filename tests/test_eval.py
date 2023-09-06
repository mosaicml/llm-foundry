# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import os
import sys
from typing import Any

import omegaconf as om
import pytest
from composer import Trainer

from llmfoundry import COMPOSER_MODEL_REGISTRY
from llmfoundry.utils import build_tokenizer

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


@pytest.fixture()
def mock_saved_model_path():
    # load the eval and model config
    with open('eval/yamls/test_eval.yaml', 'r', encoding='utf-8') as f:
        eval_cfg = om.OmegaConf.load(f)
    model_cfg = eval_cfg.models[0]
    # set device to cpu
    device = 'cpu'
    model_cfg.model.init_device = device
    # build tokenizer
    tokenizer = build_tokenizer(model_cfg.tokenizer.name,
                                model_cfg.tokenizer.get('kwargs', {}))
    # build model
    model = COMPOSER_MODEL_REGISTRY[model_cfg.model.name](model_cfg.model,
                                                          tokenizer)
    # create mocked save checkpoint
    trainer = Trainer(model=model, device=device)
    saved_model_path = os.path.join(os.getcwd(), 'test_model.pt')
    trainer.save_checkpoint(saved_model_path)
    yield saved_model_path

    # clean up the mocked save checkpoint
    os.remove(saved_model_path)


def test_icl_eval(capfd: Any, mock_saved_model_path: Any):
    with open('eval/yamls/test_eval.yaml', 'r', encoding='utf-8') as f:
        test_cfg = om.OmegaConf.load(f)
    test_cfg.models[0].load_path = mock_saved_model_path
    assert isinstance(test_cfg, om.DictConfig)
    main(test_cfg)
    out, _ = capfd.readouterr()
    expected_results = '| Category                    | Benchmark      | Subtask   |   Accuracy | Number few shot   | Model    |\n|:----------------------------|:---------------|:----------|-----------:|:------------------|:---------|\n| language_understanding_lite | lambada_openai |           |          0 | 0-shot            | tiny_mpt '
    assert expected_results in out
    expected_results = '| model_name   |   average |   language_understanding_lite |\n|:-------------|----------:|------------------------------:|\n| tiny_mpt     |         0 |                             0 |'
    assert expected_results in out
