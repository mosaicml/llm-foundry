
from pathlib import Path
import shutil
from omegaconf import OmegaConf as om
from transformers import AutoTokenizer
from llmfoundry.common.builders import build_icl_evaluators
import os
import pytest

TMP_FOLDER = 'tmp_data'
EXPECTED_LABELS = ['jeopardy/0-shot/american_history', 'jeopardy/0-shot/literature', 'jeopardy/0-shot/science', 'jeopardy/0-shot/word_origins', 'jeopardy/0-shot/world_history', 'jeopardy/1-shot/american_history', 'jeopardy/1-shot/literature', 'jeopardy/1-shot/science', 'jeopardy/1-shot/word_origins', 'jeopardy/1-shot/world_history', 'jeopardy/5-shot/american_history', 'jeopardy/5-shot/literature', 'jeopardy/5-shot/science', 'jeopardy/5-shot/word_origins', 'jeopardy/5-shot/world_history', 'jeopardy/10-shot/american_history', 'jeopardy/10-shot/literature', 'jeopardy/10-shot/science', 'jeopardy/10-shot/word_origins', 'jeopardy/10-shot/world_history']

def load_icl_config(conf_path='llmfoundry/icl_eval/yamls/tasks.yaml'):
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
    evaluators, _ = build_icl_evaluators(task_cfg, tokenizer, destination_dir=f"{os.getcwd()}/{TMP_FOLDER}")

    labels = [e.label for e in evaluators]
    assert labels == EXPECTED_LABELS
    for e in evaluators:
        inputs = next(e.dataloader.dataloader.__iter__())['input_ids'][0]
        continuation_indices = list(next(e.dataloader.dataloader.__iter__())['continuation_indices'][0])
        full_example = tokenizer.decode(inputs[0:continuation_indices[-1]])
        answer = tokenizer.decode(inputs[continuation_indices[0]:continuation_indices[-1]])

        if e.label == 'jeopardy/0-shot/american_history':
            assert full_example == "AMERICAN HISTORY: On May 29, 1765 Patrick Henrys Stamp Act protest was interrupted with this one word\nAnswer: Treason"
            assert answer == " Treason"
        elif e.label == 'jeopardy/1-shot/american_history':
            assert full_example == "AMERICAN HISTORY: Witchcraft trials held in this town in 1692 led to the hangings of 19 people\nAnswer: Salem\nAMERICAN HISTORY: On May 29, 1765 Patrick Henrys Stamp Act protest was interrupted with this one word\nAnswer: Treason"
            assert answer == " Treason"
            
    
    
  

def test_icl_task_loading_gptj_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-j-6b')
    task_cfg = load_icl_config()
    evaluators, _ = build_icl_evaluators(task_cfg, tokenizer)
    breakpoint()

def test_icl_task_loading_opt_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained('facebook/opt-6.7b')
    task_cfg = load_icl_config()
    evaluators, _ = build_icl_evaluators(task_cfg, tokenizer)
    breakpoint()

def test_icl_task_loading_sentencepiece_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained('huggyllama/llama-7b')
    task_cfg = load_icl_config()
    evaluators, _ = build_icl_evaluators(task_cfg, tokenizer)
    breakpoint()

def test_icl_task_loading_gptneox_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')
    task_cfg = load_icl_config()
    evaluators, _ = build_icl_evaluators(task_cfg, tokenizer)
    breakpoint()
