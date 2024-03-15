import transformers
import pytest

from llmfoundry.models.hf.hf_t5 import ComposerHFT5
from llmfoundry.utils.warnings import ExperimentalWarning

from omegaconf import OmegaConf

def test_experimental_hf_t5():
    cfg = OmegaConf.create({
        'pretrained_model_name_or_path': 't5-base',
        'config_overrides': {
            'num_layers': 2,
            'num_decoder_layers': 2,
        },
        'pretrained': False,
        'init_device': 'cpu',
        'z_loss': 0.0,
        'adapt_vocab_for_denoising': False
    })

    tokenizer = transformers.T5Tokenizer.from_pretrained('t5-base')

    with pytest.warns(ExperimentalWarning):
        _ = ComposerHFT5(cfg, tokenizer)