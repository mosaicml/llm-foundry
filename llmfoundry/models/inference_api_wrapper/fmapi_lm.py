
from typing import Dict
import os

from transformers import AutoTokenizer

from llmfoundry.models.inference_api_wrapper import (OpenAICausalLMEvalWrapper,
                                                     OpenAIChatAPIEvalWrapper)
__all__ = [
    'FMAPICasualLMEvalWrapper',
]


def _get_base_url(model_cfg: Dict) -> str:
    is_local = model_cfg.get('local', False)

    if is_local:
        return os.environ.get('MOSAICML_MODEL_ENDPOINT', 'http://0.0.0.0:8080/v2')
    elif 'base_url' in model_cfg:
        return model_cfg['base_url']

    raise ValueError('Must specify base_url in model_cfg for FMAPIsEvalWrapper')


class FMAPICasualLMEvalWrapper(OpenAICausalLMEvalWrapper):
    """Databricks Foundational Model API wrapper for causal LM models
    """

    def __init__(self, model_cfg: Dict, tokenizer: AutoTokenizer):
        model_cfg['base_url'] = _get_base_url(model_cfg)
        # todo: check if endpoint is ready
        super().__init__(model_cfg, tokenizer)
