# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import copy
import os
import pathlib
from typing import Any, Union

import omegaconf as om
import pytest
from composer import Trainer
from composer.loggers import InMemoryLogger

from llmfoundry.utils import build_tokenizer
from llmfoundry.utils.builders import build_composer_model
from llmfoundry.utils.config_utils import to_str_dict
from scripts.eval.eval import main  # noqa: E402
from tests.data_utils import create_c4_dataset_xxsmall, gpt_tiny_cfg


@pytest.fixture(autouse=True)
def set_correct_cwd():
    if not os.getcwd().endswith('llm-foundry/scripts'):
        os.chdir('scripts')

    yield

    if os.getcwd().endswith('llm-foundry/scripts'):
        os.chdir('..')


@pytest.fixture
def eval_cfg(foundry_dir: str) -> Union[om.ListConfig, om.DictConfig]:
    yaml_path = os.path.join(foundry_dir, 'scripts/eval/yamls/test_eval.yaml')
    with open(yaml_path, 'r', encoding='utf-8') as f:
        eval_cfg = om.OmegaConf.load(f)
    return eval_cfg


@pytest.fixture()
def mock_saved_model_path(eval_cfg: Union[om.ListConfig, om.DictConfig]):
    eval_cfg = copy.deepcopy(eval_cfg)  # copy config before modifying
    model_cfg = eval_cfg.models[0]
    # set device to cpu
    device = 'cpu'
    model_cfg.model.init_device = device
    # build tokenizer
    tokenizer = build_tokenizer(model_cfg.tokenizer.name,
                                model_cfg.tokenizer.get('kwargs', {}))
    # build model
    name = model_cfg.model.pop('name')
    model = build_composer_model(name=name,
                                 tokenizer=tokenizer,
                                 cfg=to_str_dict(model_cfg.model))

    # create mocked save checkpoint
    trainer = Trainer(model=model, device=device)
    saved_model_path = os.path.join(os.getcwd(), 'test_model.pt')
    trainer.save_checkpoint(saved_model_path)
    yield saved_model_path

    # clean up the mocked save checkpoint
    os.remove(saved_model_path)


def test_icl_eval(eval_cfg: Union[om.ListConfig, om.DictConfig], capfd: Any,
                  mock_saved_model_path: Any):
    eval_cfg = copy.deepcopy(eval_cfg)
    eval_cfg.models[0].load_path = mock_saved_model_path
    assert isinstance(eval_cfg, om.DictConfig)
    main(eval_cfg)
    out, _ = capfd.readouterr()
    expected_results = '| Category                    | Benchmark      | Subtask   |   Accuracy | Number few shot   | Model    |\n|:----------------------------|:---------------|:----------|-----------:|:------------------|:---------|\n| language_understanding_lite | lambada_openai |           |          0 | 0-shot            | tiny_mpt |'
    assert expected_results in out
    expected_results = '| model_name   |   default_average |   language_understanding_lite |\n|:-------------|------------------:|------------------------------:|\n| tiny_mpt     |                 0 |                             0 |'
    assert expected_results in out


def test_loader_eval(capfd: Any, mock_saved_model_path: Any,
                     tmp_path: pathlib.Path):

    c4_dataset_name = create_c4_dataset_xxsmall(tmp_path)

    # Use a training config that already has eval loader configured
    test_cfg = gpt_tiny_cfg(c4_dataset_name, 'cpu')

    # define icl eval task
    test_cfg.icl_tasks = om.ListConfig([
        om.DictConfig({
            'label':
                'lambada_openai',
            'dataset_uri':
                'eval/local_data/language_understanding/lambada_openai_small.jsonl',
            'num_fewshot': [0],
            'icl_task_type':
                'language_modeling'
        })
    ])

    # convert the model from a training to eval model
    model = test_cfg.pop('model')
    eval_model = {
        'model_name': model.get('name'),
        'model': model,
        'load_path': mock_saved_model_path
    }

    tokenizer = test_cfg.pop('tokenizer')
    eval_model['tokenizer'] = tokenizer
    test_cfg.models = [eval_model]

    # Set up multiple eval dataloaders
    first_eval_loader = test_cfg.eval_loader
    first_eval_loader.label = 'c4'
    # Create second eval dataloader using the arxiv dataset.
    second_eval_loader = copy.deepcopy(first_eval_loader)
    second_eval_loader.label = 'arxiv'
    test_cfg.eval_loader = om.OmegaConf.create(
        [first_eval_loader, second_eval_loader])

    test_cfg.max_duration = '1ba'
    test_cfg.eval_interval = '1ba'
    test_cfg.loggers = om.DictConfig({'inmemory': om.DictConfig({})})

    trainers, eval_gauntlet_df = main(test_cfg)

    assert eval_gauntlet_df is None
    assert len(trainers) == 1  # one per model
    trainer = trainers[0]

    assert isinstance(trainer.logger.destinations, tuple)

    assert len(trainer.logger.destinations) > 0
    inmemorylogger = trainer.logger.destinations[
        0]  # pyright: ignore [reportGeneralTypeIssues]
    assert isinstance(inmemorylogger, InMemoryLogger)
    print(inmemorylogger.data.keys())

    # Checks for first eval dataloader
    assert 'metrics/eval/c4/LanguageCrossEntropy' in inmemorylogger.data.keys()
    assert isinstance(
        inmemorylogger.data['metrics/eval/c4/LanguageCrossEntropy'], list)
    assert len(
        inmemorylogger.data['metrics/eval/c4/LanguageCrossEntropy'][-1]) > 0
    assert isinstance(
        inmemorylogger.data['metrics/eval/c4/LanguageCrossEntropy'][-1], tuple)

    # Checks for second eval dataloader
    assert 'metrics/eval/arxiv/LanguageCrossEntropy' in inmemorylogger.data.keys(
    )
    assert isinstance(
        inmemorylogger.data['metrics/eval/arxiv/LanguageCrossEntropy'], list)
    assert len(
        inmemorylogger.data['metrics/eval/arxiv/LanguageCrossEntropy'][-1]) > 0
    assert isinstance(
        inmemorylogger.data['metrics/eval/arxiv/LanguageCrossEntropy'][-1],
        tuple)
