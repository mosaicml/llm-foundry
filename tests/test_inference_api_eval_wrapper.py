

from llmfoundry.models.inference_api_wrapper import OpenAICausalLMEvalWrapper,OpenAIChatAPIEvalWrapper, OpenAITokenizerWrapper
import os
import random
import shutil
from pathlib import Path

import pytest
from omegaconf import OmegaConf as om


from llmfoundry.utils.builders import build_icl_evaluators


def load_icl_config(conf_path='tests/test_tasks.yaml'):
    with open(conf_path) as f:
        test_cfg = om.load(f)
    return test_cfg


@pytest.fixture(autouse=True, scope='function')
def tmp_dir():
    TMP_FOLDER = 'tmp_data' + str(random.randint(0, 100_000))
    dirpath = Path(TMP_FOLDER)
    if dirpath.exists() and dirpath.is_dir():
        shutil.rmtree(dirpath)
    os.mkdir(TMP_FOLDER)
    yield TMP_FOLDER
    dirpath = Path(TMP_FOLDER)
    if dirpath.exists() and dirpath.is_dir():
        shutil.rmtree(dirpath)


def test_openai_api_eval_wrapper(tmp_dir):
    model_name = 'davinci'
    tokenizer = OpenAITokenizerWrapper(model_name)
    model = OpenAICausalLMEvalWrapper(model_cfg={"version": model_name}, tokenizer=tokenizer)
    task_cfg = load_icl_config()
    evaluators, _ = build_icl_evaluators(task_cfg.icl_tasks,
                                         tokenizer,
                                         1024,
                                         8,
                                         destination_dir=f'{os.getcwd()}/{tmp_dir}')
    
    batch = next(evaluators[0].dataloader.dataloader.__iter__())
    result = model.eval_forward(batch)
    model.update_metric(
        batch,
        result,
        metric=model.get_metrics()['InContextLearningLMAccuracy']
    )
    acc = model.get_metrics()['InContextLearningLMAccuracy'].compute()
    assert acc > 0.0

def test_chat_api_eval_wrapper(tmp_dir):
    model_name = 'gpt-3.5-turbo'
    tokenizer = OpenAITokenizerWrapper(model_name)
    chatmodel = OpenAIChatAPIEvalWrapper(model_cfg={"version": model_name}, tokenizer=tokenizer)
    task_cfg = load_icl_config()
    evaluators, _ = build_icl_evaluators(task_cfg.icl_tasks,
                                         tokenizer,
                                         1024,
                                         8,
                                         destination_dir=f'{os.getcwd()}/{tmp_dir}')
    
    batch = next(evaluators[0].dataloader.dataloader.__iter__())
    result = chatmodel.eval_forward(batch)
    
    chatmodel.update_metric(
        batch,
        result,
        metric=chatmodel.get_metrics()['InContextLearningLMAccuracy']
    )
    acc = chatmodel.get_metrics()['InContextLearningLMAccuracy'].compute()
    assert acc > 0.0
