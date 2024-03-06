# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

# helper functions from https://github.com/CarperAI/trlx/blob/main/trlx/utils/modeling.py
# which is MIT licensed

import functools
from typing import TYPE_CHECKING, Any, Iterable, List, Optional, Union

from composer.models.huggingface import maybe_get_underlying_model
from transformers import PreTrainedModel
from transformers.models.opt.modeling_opt import OPTDecoder

if TYPE_CHECKING:
    from peft import PeftModel


# helper functions
def rhasattr(obj: Any, attr: str) -> bool:
    """A chain-able attribute version of hasattr.

    For example, to check if
    `obj` has the attribute `foo.bar.baz`, you can use:
        `rhasattr(obj, "foo.bar.baz")`
    Reference: https://stackoverflow.com/a/67303315
    """
    _nested_attrs = attr.split('.')
    _curr_obj = obj
    for _a in _nested_attrs[:-1]:
        if hasattr(_curr_obj, _a):
            _curr_obj = getattr(_curr_obj, _a)
        else:
            return False
    return hasattr(_curr_obj, _nested_attrs[-1])


def rgetattr(obj: Any, attr: str, *args: List[Any]) -> Any:
    """A chain-able attribute version of getattr.

    For example, to get the attribute `foo.bar.baz` from `obj`, you can use:
        `rgetattr(obj, "foo.bar.baz")`
    Reference: https://stackoverflow.com/a/31174427
    """

    def _getattr(obj: Any, attr: str):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split('.'))


def findattr(obj: Any, attrs: Iterable[str]) -> Optional[Any]:
    for attr in attrs:
        if rhasattr(obj, attr):
            return rgetattr(obj, attr)
    return None


def hf_get_causal_base_model(model: PreTrainedModel) -> Any:
    """Returns the causal decoder backbone of the specified HuggingFace model.

    Newer HF models have a `self.get_decoder()` method. Older models do not.

    NOTE: Different model configurations have different causal decoder attribute
    names.
        - transformer: (GPT2LMHeadModel, GPTJConfig)
        - model.decoder: (OPTConfig, BloomConfig)
        - gpt_neox: (GPTNeoXConfig)
    """
    if hasattr(model, 'get_decoder'):
        return model.get_decoder()

    decoder_attrs = ('transformer', 'model.decoder', 'gpt_neox', 'model.transformer')
    causal_base_model = findattr(model, decoder_attrs)
    if causal_base_model is None:
        raise ValueError(
            f'Unable to FSDP-wrap model {model}. Please open a github issue to add support.'
        )
    return causal_base_model


def hf_get_hidden_layers(model: PreTrainedModel) -> Any:
    """Returns the hidden layers of the specified model.

    Expects to receive the causal decoder backbone, not he XXForCausalLM wrapper.

    NOTE: Different model configurations have different hidden layer attribute names.
        - h: (BloomForCausalLM, GPT2LMHeadModel, GPTJForCausalLM)
        - decoder.layers: (OPTForCausalLM)
        - layers: (GPTNeoXForCausalLM, LlaMaForCausalLM)
        - blocks: (MPTForCausalLM)
    """
    hidden_layers_attrs = (
        'h',  # BLOOM, GPT2, GPTJ
        'decoder.layers',  # OPT
        'layers',  # GPTNeoX, Llama, ProphetNet, Marian (from encoder)
        'block',  # T5, BART, Pegasus (from encoder)
        'blocks',  # MPT
    )
    layers = findattr(model, hidden_layers_attrs)
    if layers is None:
        raise ValueError(
            f'Unable to find hidden layer for {model}. Model must have one of the following attributes: {hidden_layers_attrs}'
        )
    return layers


def hf_get_init_device(init_device: Optional[str]) -> Optional[str]:
    """Returns the appropriate device to initialize models."""
    from composer.utils import dist
    if init_device == 'mixed':
        if dist.get_local_rank() == 0:
            return 'cpu'
        return 'meta'
    return init_device


# /end helper functions


def prepare_hf_model_for_fsdp(model: PreTrainedModel,
                              init_device: Optional[str]) -> None:
    """FSDP wrap a HuggingFace model.

    Call specific functions
    """
    if model.config.is_encoder_decoder:
        prepare_hf_enc_dec_model_for_fsdp(model, init_device)
    else:
        # many common decoder-only model do not set the flag
        # model.config.is_decoder, so we can't trust it
        prepare_hf_causal_lm_model_for_fsdp(model, init_device)


def prepare_hf_causal_lm_model_for_fsdp(model: Union[PreTrainedModel,
                                                     'PeftModel'],
                                        init_device: Optional[str]) -> None:
    """FSDP wrap a HuggingFace decoder.

    Wrap any model for FSDP which follows one of the 3 existing conventions from
    HuggingFace for decoder-only LLMs.
    """
    causal_base_model = hf_get_causal_base_model(model)

    # OPT has an extra layer of wrapping, so special case here
    if isinstance(causal_base_model, OPTDecoder):
        underlying_model = maybe_get_underlying_model(model)
        underlying_model.model._fsdp_wrap = False
    model_block = hf_get_hidden_layers(causal_base_model)
    lm_head = model.get_output_embeddings()
    # Try to get input embeddings from the transformer backbone
    # and then from the XXXForCausalLM
    try:
        tied_embeddings = causal_base_model.get_input_embeddings()
    except:
        tied_embeddings = model.get_input_embeddings()

    modules = {
        'base_model': causal_base_model,
        'model_block': model_block,
        'lm_head': lm_head,
        'tied_embeddings': tied_embeddings
    }

    for mod_name, module in modules.items():
        if module is None:
            raise ValueError(
                f'Unable to FSDP-wrap this model! `{mod_name}` does not ' +
                'follow common layer/weight naming conventions.')
    block_type = type(model_block[0])

    # When using the HF LM models,
    # the weights of the self.lm_head and self.transformer.wte are tied.
    # This tying occurs inside the `self.post_init()` function.
    # This is a hurdle for FSDP because they need to be in the same FSDP block
    # These lines ensures that both modules stay together in the top-most block when
    # the model has this tying enabled (almost all do; this property defaults to True)
    if model.config.tie_word_embeddings:
        causal_base_model._fsdp_wrap = False
        tied_embeddings._fsdp_wrap = False
        lm_head._fsdp_wrap = False

    # PEFT layers should be individually wrapped
    # TODO: Revisit this if we enforce use_orig_params=True, which seems to support
    # mixed frozen/unfrozen FSDP modules
    if hasattr(model, 'peft_type') and model.peft_type is not None:
        peft_type = model.peft_type.lower()
        active_adapters = [adapter.lower() for adapter in model.active_adapters]
        for name, module in model.named_modules():
            if peft_type in name.lower() and any(
                    adapter in name.lower() for adapter in active_adapters):
                has_parameters = next(module.parameters(), None) is not None
                has_buffers = next(module.buffers(), None) is not None
                if has_parameters or has_buffers:
                    module._fsdp_wrap = True

    # FSDP Wrap and Activation Checkpoint every model block
    model.fsdp_wrap_fn = lambda module: isinstance(module, block_type)
    model.activation_checkpointing_fn = lambda module: isinstance(
        module, block_type)


def prepare_hf_enc_dec_model_for_fsdp(model: PreTrainedModel,
                                      init_device: Optional[str]) -> None:
    """Wrap an encoder/decoder HF model.

    This works for T5, BART, Pegasus, PegasusX, but not all enc/dec (ProphetNet)
    You have model.shared, model.encoder, model.decoder and model.lm_head, where
    model.shared are the embeddings which are tied to model.lm_head, and
    model.shared == model.encoder.embed_tokens and model.shared ==
    model.decoder.embed_tokens
    """
    tied_embeddings = model.get_input_embeddings()
    encoder = model.get_encoder()
    decoder = model.get_decoder()
    lm_head = model.get_output_embeddings()
    # some encoder/decoders have different layers for encoder vs decoder
    encoder_block = hf_get_hidden_layers(encoder)
    decoder_block = hf_get_hidden_layers(decoder)

    modules = {
        'encoder': encoder,
        'decoder': decoder,
        'encoder_block': encoder_block,
        'decoder_block': decoder_block,
        'lm_head': lm_head,
        'tied_embeddings': tied_embeddings
    }

    for mod_name, module in modules.items():
        if module is None:
            raise ValueError(
                f'Unable to FSDP-wrap this model! `{mod_name}` does not ' +
                'follow common layer/weight naming conventions.')
    decoder_block_type = type(decoder_block[0])
    encoder_block_type = type(encoder_block[0])

    if model.config.tie_word_embeddings:
        # it is possible to train an enc/dec without tied embeddings, hence the check
        tied_embeddings._fsdp_wrap = False
        encoder._fsdp_wrap = False
        decoder._fsdp_wrap = False
        lm_head._fsdp_wrap = False

    # FSDP Wrap and Activation Checkpoint every decoder block
    model.fsdp_wrap_fn = lambda module: isinstance(module, decoder_block_type)
    model.activation_checkpointing_fn = lambda module: isinstance(
        module, decoder_block_type)

    if encoder_block_type == decoder_block_type:
        return

    # need to wrap encoder blocks separately for ProhpetNet and Marian
    model.fsdp_wrap_fn = lambda module: isinstance(module, encoder_block_type)
    model.activation_checkpointing_fn = lambda module: isinstance(
        module, encoder_block_type)
