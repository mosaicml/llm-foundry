# Copyright 2025 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0
#
# ===================================================================
# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This copyright notice applies to the following classes and methods:
# - TELlamaDecoderLayer
# - custom_prepare_te_modules_for_fsdp
# ===================================================================

import logging
from contextlib import contextmanager
from typing import Any, Optional, Union

import torch
import torch.nn as nn
import transformer_engine
import transformer_engine.pytorch as te
import transformers
from transformers import AutoModelForCausalLM, PreTrainedModel
from transformers.models.auto.auto_factory import _BaseAutoModelClass
from transformers.models.llama.modeling_llama import LlamaConfig

from llmfoundry.models.hf import ComposerHFCausalLM
from llmfoundry.registry import models

log = logging.getLogger(__name__)

# ===================================================================
# Transformer Engine FSDP Integration with ComposerModel
# ===================================================================
# This section monkey patches the TE's FSDP support to work with our
# specific model architecture
# (ComposerModel wrapped HF model with TE layers)
# ===================================================================

from transformer_engine.pytorch.distributed import prepare_te_modules_for_fsdp


def custom_prepare_te_modules_for_fsdp(fsdp_obj: torch.nn.Module) -> None:
    """Enhanced version of TE's prepare_te_modules_for_fsdp.

    This handles nested module structures. Specifically, this function:
    1. Calls the original TE implementation first
    2. Adds additional traversal to find TE modules that might be nested deeper in the model
       structure and weren't found by the original implementation
    3. Ensures FP8 FSDP sharding works correctly with ComposerModel wrapped HF models

    Args:
        fsdp_obj: FSDP-wrapped root module that may contain FSDP-wrapped TE modules.
    """
    log.info(
        'Using custom prepare_te_modules_for_fsdp for ComposerModel wrapped HF models',
    )

    # First, call the original implementation
    prepare_te_modules_for_fsdp(fsdp_obj)

    from torch.distributed.fsdp._traversal_utils import \
        _get_fsdp_states_with_modules

    fsdp_states, fsdp_modules = _get_fsdp_states_with_modules(fsdp_obj)

    # Perform additional traversal for nested TE modules
    for state, fsdp_module in zip(fsdp_states, fsdp_modules):

        if hasattr(fsdp_module.module, 'named_modules'):
            for _, submodule in fsdp_module.module.named_modules():
                if transformer_engine.pytorch.distributed._is_te_module(
                    submodule,
                ):
                    if hasattr(submodule, 'primary_weights_in_fp8'):
                        assert not submodule.primary_weights_in_fp8, (
                            'TE modules with primary weights in FP8 cannot be FSDP-wrapped. '
                            'Please initialize your model without the te.fp8_model_init(...) context.'
                        )

                    # Apply the process group to enable proper sharding
                    setattr(submodule, 'fsdp_group', state.process_group)


# Apply the monkey patch
transformer_engine.pytorch.distributed.prepare_te_modules_for_fsdp = custom_prepare_te_modules_for_fsdp

# ===================================================================
# TE-optimized ComposerHFCausalLM
# ===================================================================


@contextmanager
def replace_decoder(te_decoder_cls: te.TransformerLayer):
    """Replace `LlamaDecoderLayer` with custom `TELlamaDecoderLayer`."""
    original_llama_decoder_cls = (
        transformers.models.llama.modeling_llama.LlamaDecoderLayer
    )
    transformers.models.llama.modeling_llama.LlamaDecoderLayer = te_decoder_cls
    try:
        yield
    finally:
        transformers.models.llama.modeling_llama.LlamaDecoderLayer = (
            original_llama_decoder_cls
        )


class TELlamaDecoderLayer(te.TransformerLayer):
    """Wrapper class over TE's `TransformerLayer`.

    This makes the wrapper very similar to HF's `LlamaDecoderLayer` and easier to replace it in the code.

    Args:
        config: LlamaConfig
        args: positional args (for compatibility with `LlamaDecoderLayer`)
        kwargs: keyword args (for compatibility with `LlamaDecoderLayer`)
    """

    def __init__(
        self,
        config: LlamaConfig,
        *args: tuple[Any, ...],
        **kwargs: dict[str, Any],
    ):
        super().__init__(
            hidden_size=config.hidden_size,
            ffn_hidden_size=config.intermediate_size,
            num_attention_heads=config.num_attention_heads,
            bias=False,
            layernorm_epsilon=config.rms_norm_eps,
            hidden_dropout=0,
            attention_dropout=0,
            fuse_qkv_params=False,
            normalization='RMSNorm',
            activation='swiglu',
            attn_input_format='bshd',
            num_gqa_groups=config.num_key_value_heads,
        )
        te_rope = te.attention.RotaryPositionEmbedding(
            config.hidden_size // config.num_attention_heads,
        )
        self.te_rope_emb = te_rope(max_seq_len=config.max_position_embeddings,
                                  ).cuda()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *args: tuple[Any, ...],
        **kwargs: dict[str, Any],
    ):
        """Custom forward to match HF LlamaDecoderLayer.

        We only pass arguments similar to `TransformerLayer`.
        """
        return (
            super().forward(
                hidden_states,
                attention_mask=attention_mask,
                rotary_pos_emb=self.te_rope_emb,
            ),
        )


class TEAutoModelForCausalLM(AutoModelForCausalLM):
    """TE-optimized AutoModelForCausalLM.

    This class extends HuggingFace's AutoModelForCausalLM to:
    1. Replace standard LlamaDecoderLayer with the optimized TELlamaDecoderLayer
    2. Replace standard nn.Linear layers with optimized TE Linear layers
    """

    @classmethod
    def replace_linear_layers(
        cls,
        model: nn.Module,
        precision: torch.dtype = torch.bfloat16,
    ) -> None:
        """Replace all nn.Linear layers in the model with TE Linear layers.

        This function recursively traverses the model and replaces all nn.Linear layers with TE Linear layers.

        Args:
            model: The PyTorch model containing nn.Linear layers to replace
            precision: Data type for the TE Linear layers parameters (default: bfloat16)
        """

        def replace_linear(module: nn.Module, precision: torch.dtype):
            for name, child in module.named_children():
                if isinstance(child, nn.Linear):
                    # Create equivalent TE Linear layer with same dimensions and bias setting
                    te_linear = te.Linear(
                        child.in_features,
                        child.out_features,
                        bias=child.bias is not None,
                        params_dtype=precision,
                    )
                    setattr(module, name, te_linear)
                else:
                    # Recursively process non-Linear modules
                    replace_linear(child, precision)

        replace_linear(model, precision)

    @classmethod
    def from_config(cls, config: LlamaConfig, **kwargs: dict[str, Any]):
        """Creates a TE-optimized model instance from a configuration.

        This method:
        1. Replaces the LlamaDecoderLayers with TELlamaDecoderLayers
        2. Creates the model using the parent class method
        3. Replaces all linear layers with TE Linear layers

        Args:
            config: LlamaConfig object containing model architecture settings
            **kwargs: Additional arguments passed to the model constructor

        Returns:
            A TE-optimized model instance
        """
        # Replace LlamaDecoderLayers with TE-optimized versions and create the model
        with replace_decoder(te_decoder_cls=TELlamaDecoderLayer):
            model = super().from_config(config, **kwargs)

        # Replace standard linear layers with TE-optimized versions
        precision = kwargs.get('torch_dtype', torch.bfloat16)
        cls.replace_linear_layers(model, precision)

        return model


class ComposerHFCausalTELM(ComposerHFCausalLM):
    """TE optimized ComposerHFCausalLM.

    Configures a :class:`.HuggingFaceModel` around a Causal LM using
    TEAutoModelForCausalLM.
    """

    model_cls: Union[_BaseAutoModelClass,
                     PreTrainedModel] = TEAutoModelForCausalLM


models.register('hf_causal_lm_te', func=ComposerHFCausalTELM)
