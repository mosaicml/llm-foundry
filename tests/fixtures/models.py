from typing import Any, Dict, Optional, Union
from omegaconf import DictConfig
from pytest import fixture
from composer.devices import Device

from llmfoundry.models.model_registry import COMPOSER_MODEL_REGISTRY
from transformers import PreTrainedTokenizerBase

from llmfoundry.utils.builders import build_tokenizer


def _build_model(config: DictConfig, tokenizer: PreTrainedTokenizerBase, device: Device):
    model = COMPOSER_MODEL_REGISTRY[config.name](config, tokenizer)
    model = device.module_to_device(model)
    return model

@fixture
def tokenizer():
    return build_tokenizer('EleutherAI/gpt-neox-20b', {})

@fixture
def build_mpt(tokenizer: PreTrainedTokenizerBase):
    def build(device: Optional[Union[str, Device]], **kwargs: Dict[str, Any]):
        config = DictConfig({
            'name': 'mpt_causal_lm',
            'd_model': 128,
            'n_heads': 4,
            'n_layers': 2,
            'expansion_ratio': 2,
        })
        config.update(kwargs)
        model = _build_model(config, tokenizer, device)
        return model
    return build

@fixture
def build_hf_mpt(tokenizer: PreTrainedTokenizerBase):
    def build(device: Optional[Union[str, Device]], **kwargs: Dict[str, Any]):
        config_overrides = {
                'd_model': 128,
                'n_heads': 4,
                'n_layers': 2,
                'expansion_ratio': 2,
        }
        config_overrides.update(kwargs)
        config = DictConfig({
            'name': 'hf_causal_lm',
            'pretrained_model_name_or_path': 'mosaicml/mpt-7b',
            'pretrained': False,
            'config_overrides': config_overrides,
        })
        model = _build_model(config, tokenizer, device)
        return model
    return build
