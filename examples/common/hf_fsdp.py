# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0
# helper functions from https://github.com/CarperAI/trlx/blob/main/trlx/utils/modeling.py
# which is MIT licensed

import functools
from typing import Any, Iterable, List

from transformers import PreTrainedModel
from transformers.models.opt.modeling_opt import OPTDecoder

# helper functions


def rhasattr(obj: Any, attr: str):
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


def rgetattr(obj: Any, attr: str, *args: List[Any]):
    """A chain-able attribute version of getattr.

    For example, to get the attribute `foo.bar.baz` from `obj`, you can use:
        `rgetattr(obj, "foo.bar.baz")`
    Reference: https://stackoverflow.com/a/31174427
    """

    def _getattr(obj: Any, attr: str):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split('.'))


def findattr(obj: Any, attrs: Iterable[str]):
    for attr in attrs:
        if rhasattr(obj, attr):
            return rgetattr(obj, attr)
    return None


def hf_get_causal_base_model(model: PreTrainedModel):
    """Returns the causal decoder backbone of the specified HuggingFace model.

    NOTE: Different model configurations have different causal decoder attribute
    names.
        - transformer: (GPT2LMHeadModel, GPTJConfig)
        - model.decoder: (OPTConfig, BloomConfig)
        - gpt_neox: (GPTNeoXConfig)
    """
    decoder_attrs = ('transformer', 'model.decoder', 'gpt_neox')
    return findattr(model, decoder_attrs)


def hf_get_hidden_layers(model: PreTrainedModel):
    """Returns the hidden layers of the specified model.

    NOTE: Different model configurations have different hidden layer attribute names.
        - transformer.h: (BloomForCausalLM, GPT2LMHeadModel, GPTJForCausalLM)
        - model.decoder.layers: (OPTForCausalLM)
        - gpt_neox.layers: (GPTNeoXForCausalLM)
    """
    hidden_layers_attrs = (
        'transformer.h',
        'model.decoder.layers',
        'gpt_neox.layers',
        'block',  # T5, BART, Pegasus (from encoder)
        'layers',  # ProphetNet, Marian (from encoder)
    )
    return findattr(model, hidden_layers_attrs)


# /end helper functions


def prepare_hf_model_for_fsdp(model: PreTrainedModel) -> None:
    """FSDP wrap a HuggingFace model.

    Call specific functions
    """
    if model.config.is_encoder_decoder:
        prepare_hf_enc_dec_model_for_fsdp(model)
    else:
        # many common decoder-only model do not set the flag
        # model.config.is_decoder, so we can't trust it
        prepare_hf_causal_lm_model_for_fsdp(model)


def prepare_hf_causal_lm_model_for_fsdp(model: PreTrainedModel) -> None:
    """FSDP wrap a HuggingFace decoder.

    Wrap any model for FSDP which follows one of the 3 existing conventions from
    HuggingFace for decoder-only LLMs.
    """
    causal_base_model = hf_get_causal_base_model(model)
    # OPT has an extra layer of wrapping, so special case here
    if isinstance(causal_base_model, OPTDecoder):
        model.model._fsdp_wrap = False
    model_block = hf_get_hidden_layers(model)  # type: ignore
    lm_head = model.get_output_embeddings()
    # some models (OPT) implement .get_input_embeddings for the causal subclass
    # but all of them implement it for the base model
    tied_embeddings = causal_base_model.get_input_embeddings()
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
    block_type = type(model_block[0])  # type: ignore
    # When using the HF LM models,
    # the weights of the self.lm_head and self.transformer.wte are tied.
    # This tying occurs inside the `self.post_init()` function.
    # This is a hurdle for FSDP because they need to be in the same FSDP block
    # These lines ensures that both modules stay together in the top-most block when
    # the model has this tying enabled (almost all do; this property defaults to True)
    if model.config.tie_word_embeddings:
        causal_base_model._fsdp_wrap = False  # type: ignore
        tied_embeddings._fsdp_wrap = False  # type: ignore
        lm_head._fsdp_wrap = False  # type: ignore

    # FSDP Wrap and Activation Checkpoint every model block
    model.fsdp_wrap_fn = lambda module: isinstance(module, block_type)
    model.activation_checkpointing_fn = lambda module: isinstance(
        module, block_type)


def prepare_hf_enc_dec_model_for_fsdp(model: PreTrainedModel) -> None:
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
    decoder_block_type = type(decoder_block[0])  # type: ignore
    encoder_block_type = type(encoder_block[0])  # type: ignore

    if model.config.tie_word_embeddings:
        # it is possible to train an enc/dec without tied embeddings, hence the check
        tied_embeddings._fsdp_wrap = False  # type: ignore
        encoder._fsdp_wrap = False  # type: ignore
        decoder._fsdp_wrap = False  # type: ignore
        lm_head._fsdp_wrap = False  # type: ignore

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
