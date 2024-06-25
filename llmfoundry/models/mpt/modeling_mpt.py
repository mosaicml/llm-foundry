# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""A simple, flexible implementation of a GPT model.

Inspired by https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
"""

from __future__ import annotations

import copy
import math
import warnings
from functools import cached_property
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Tuple,
    Type,
    Union,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from composer.models import HuggingFaceModel
from composer.utils import dist
from tabulate import tabulate

from llmfoundry.layers_registry import ffns_with_megablocks
from llmfoundry.models.layers.attention import is_flash_v2_installed

if is_flash_v2_installed():
    try:  # This try...except is needed because transformers requires it despite the 'if' statement above
        from flash_attn import bert_padding
        from flash_attn.layers.rotary import \
            RotaryEmbedding as DAILRotaryEmbedding
    except Exception as e:
        raise e

import logging

from transformers import PreTrainedModel, PreTrainedTokenizerBase
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.models.llama.modeling_llama import \
    LlamaDynamicNTKScalingRotaryEmbedding as HFDynamicNTKScalingRotaryEmbedding
from transformers.models.llama.modeling_llama import \
    LlamaLinearScalingRotaryEmbedding as HFLinearScalingRotaryEmbedding
from transformers.models.llama.modeling_llama import \
    LlamaRotaryEmbedding as HFRotaryEmbedding

from llmfoundry.layers_registry import norms, param_init_fns
from llmfoundry.models.layers.attention import (
    attn_bias_shape,
    build_attn_bias,
    gen_slopes,
)
from llmfoundry.models.layers.blocks import MPTBlock
from llmfoundry.models.layers.custom_embedding import SharedEmbedding
from llmfoundry.models.layers.layer_builders import build_norm
from llmfoundry.models.mpt.configuration_mpt import MPTConfig
from llmfoundry.models.utils.act_ckpt import (
    build_act_ckpt_mod_to_blocks,
    check_mapping_blocks_overlap,
    pass_on_block_idx,
)
from llmfoundry.models.utils.config_moe_args import config_moe_args
from llmfoundry.models.utils.mpt_param_count import (
    mpt_get_active_params,
    mpt_get_total_params,
)

# Import the fcs and param_init_fns here so that the recursive code creating the files for hf checkpoints can find them
# These are the exceptions because fc.py and param_init_fns.py are not imported in any other place in the import tree
# isort: off
from llmfoundry.models.layers.fc import fcs  #  type: ignore
from llmfoundry.models.utils.param_init_fns import generic_param_init_fn_  # type: ignore
from llmfoundry.models.layers.norm import LPLayerNorm  # type: ignore
# isort: on

log = logging.getLogger(__name__)


def gen_rotary_embedding(
    rope_head_dim: int,
    rope_impl: str,
    rope_theta: int,
    rope_dail_config: dict,
    rope_hf_config: dict,
    max_seq_len: int,
):
    if rope_impl == 'dail':
        return DAILRotaryEmbedding(
            dim=rope_head_dim,
            base=rope_theta,
            interleaved=False,
            scale_base=rope_dail_config['xpos_scale_base'] if
            (rope_dail_config['type'] == 'xpos') else None,
            pos_idx_in_fp32=rope_dail_config['pos_idx_in_fp32'],
            device=
            'cpu',  # FSDP does not materialize modules with meta buffers, hence device is set to cpu
        )
    elif rope_impl == 'hf':
        if rope_hf_config['type'] == 'no_scaling':
            return HFRotaryEmbedding(
                rope_head_dim,
                max_position_embeddings=max_seq_len,
                base=rope_theta,
                device=
                'cpu',  # FSDP does not materialize modules with meta buffers, hence device is set to cpu
            )
        elif rope_hf_config['type'] == 'linear':
            return HFLinearScalingRotaryEmbedding(
                rope_head_dim,
                max_position_embeddings=max_seq_len,
                base=rope_theta,
                scaling_factor=rope_hf_config['factor'],
                device=
                'cpu',  # FSDP does not materialize modules with meta buffers, hence device is set to cpu
            )
        elif rope_hf_config['type'] == 'dynamic':
            return HFDynamicNTKScalingRotaryEmbedding(
                rope_head_dim,
                max_position_embeddings=max_seq_len,
                base=rope_theta,
                scaling_factor=rope_hf_config['factor'],
                device=
                'cpu',  # FSDP does not materialize modules with meta buffers, hence device is set to cpu
            )
    raise ValueError('rope_impl needs to be either dail or hf')


def gen_attention_mask_in_length(
    sequence_id: Union[None, torch.Tensor],
    S: int,
    attn_uses_sequence_id: bool,
    attn_impl: str,
    attention_mask: Union[torch.Tensor, None],
):
    """Generates the attention mask used for sequence masking in FA v2.

    Only supports sequence id based sparse attention for no attention masking or attention masking with right padding.
    In case of left padding:
        1. Training with left padding is not supported in MPT (see https://github.com/mosaicml/llm-foundry/blob/1eecd4cb8e734499f77f6a35f657b8b20c0adfcb/llmfoundry/models/mpt/modeling_mpt.py#L407).
        2. For generation with left padding, we only have a single sequence id per sample, so we don't need sequence id based sparse attention.

    Args:
        sequence_id (Union[None, torch.Tensor]): Tensor containing the sequence id for each token. Shape (batch_size, seq_len).
        S (int): Sequence length
        attn_uses_sequence_id (bool): Whether the attention uses sequence id based masking.
        attn_impl (str): Attention implementation. This function is only creates attention_mask_in_length for flash attention.
        attention_mask (Union[torch.Tensor, None]): Attention mask tensor of shape (batch_size, seq_len)

    Returns:
        attention_mask_in_length: (batch, seqlen), int, a nonzero number (e.g., 1, 2, 3, etc.) means length of concatenated sequence in b-th batch, and 0 means none. For example, if batch = 3 and seqlen = 6, the attention_mask_in_length is:
            ```
            [
            [2, 3, 0, 0, 0, 0],
            [3, 2, 0, 0, 0, 0],
            [6, 0, 0, 0, 0, 0]
            ]
            ```
        , which refers to the 3D-attention mask:
            ```
            [
            [
                [1, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 1, 1, 0, 0],
                [0, 0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0, 1]
            ],
            [
                [1, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0],
                [1, 1, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 1, 1, 0],
                [0, 0, 0, 0, 0, 1]
            ],
            [
                [1, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0],
                [1, 1, 1, 0, 0, 0],
                [1, 1, 1, 1, 0, 0],
                [1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 1]
            ]
            ]
            ```.
            (The description above is taken verbatim from https://github.com/Dao-AILab/flash-attention/blob/9356a1c0389660d7e231ff3163c1ac17d9e3824a/flash_attn/bert_padding.py#L125 .)
    """
    attention_mask_in_length = None
    if (sequence_id
        is not None) and attn_uses_sequence_id and (attn_impl == 'flash'):
        # Check if sequence has left padding. If yes, raise an error.
        if (attention_mask is not None
           ) and (attention_mask[:, 0].sum() != attention_mask.shape[0]):
            raise NotImplementedError(
                'Left padding is not supported with flash attention when attn_uses_sequence_id is set to True.',
            )
        if S != sequence_id.shape[-1]:
            raise ValueError(
                f'Sequence length ({S}) does not match length of sequences in sequence_id ({sequence_id.shape[-1]}).',
            )
        if attention_mask is not None:
            # -1 is used to pad the sequence_id where attention mask is False (https://github.com/mosaicml/llm-foundry/blob/706ea7dd40ba60a98dea5f37695d143d91c98b6c/llmfoundry/data/packing.py#L249).
            # We replace those -1 with 0 to prevent `torch.nn.functional.one_hot(sequence_id)` in the next line from failing.
            # We apply the attention mask again after the one_hot operation.
            sequence_id = sequence_id.masked_fill(~attention_mask, 0)
        attention_mask_in_length = torch.nn.functional.one_hot(sequence_id)
        if attention_mask is not None:
            attention_mask_in_length = attention_mask_in_length.masked_fill(
                ~attention_mask.unsqueeze(-1),
                0,
            )
        attention_mask_in_length = attention_mask_in_length.sum(dim=1)
        attention_mask_in_length = torch.nn.functional.pad(
            attention_mask_in_length,
            (0, S - attention_mask_in_length.shape[-1]),
            mode='constant',
            value=0,
        )

    return attention_mask_in_length


def gen_flash_attn_padding_info(
    bsz: int,
    S: int,
    past_key_len: int,
    device: torch.device,
    attention_mask_in_length: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
):
    flash_attn_padding_info = {}
    if attention_mask_in_length is None:
        key_padding_mask = attention_mask
        if key_padding_mask is None:
            key_padding_mask = torch.ones((bsz, past_key_len + S),
                                          dtype=torch.bool,
                                          device=device)
        query_padding_mask = key_padding_mask[:, -S:]
        unpadding_function = bert_padding.unpad_input
    else:
        key_padding_mask = attention_mask_in_length
        query_padding_mask = attention_mask_in_length
        unpadding_function = bert_padding.unpad_input_for_concatenated_sequences

    _, indices_q, cu_seqlens_q, max_seqlen_q = unpadding_function(
        torch.empty(bsz, S, 1, device=device),
        query_padding_mask,
    )
    _, indices_k, cu_seqlens_k, max_seqlen_k = unpadding_function(
        torch.empty(bsz, past_key_len + S, 1, device=device),
        key_padding_mask,
    )
    _, indices_v, _, _ = unpadding_function(
        torch.empty(bsz, past_key_len + S, 1, device=device),
        key_padding_mask,
    )

    flash_attn_padding_info['indices_q'] = indices_q
    flash_attn_padding_info['indices_k'] = indices_k
    flash_attn_padding_info['indices_v'] = indices_v
    flash_attn_padding_info['cu_seqlens_q'] = cu_seqlens_q
    flash_attn_padding_info['cu_seqlens_k'] = cu_seqlens_k
    flash_attn_padding_info['max_seqlen_q'] = max_seqlen_q
    flash_attn_padding_info['max_seqlen_k'] = max_seqlen_k
    return flash_attn_padding_info


def apply_sequence_id(
    attn_bias: torch.Tensor,
    sequence_id: torch.LongTensor,
    max_seq_len: int,
) -> torch.Tensor:
    seq_len = sequence_id.shape[-1]
    if seq_len > max_seq_len:
        raise ValueError(
            f'sequence_id sequence length cannot exceed max_seq_len={max_seq_len}',
        )

    # select seq_len subset of attn mask
    attn_bias = attn_bias[..., :seq_len, :seq_len]

    # Restrict attention to tokens that share the same value
    # in sequence_id
    cannot_attend = torch.logical_not(
        torch.eq(
            sequence_id.view(-1, seq_len, 1),
            sequence_id.view(-1, 1, seq_len),
        ),
    ).unsqueeze(1)
    min_val = torch.finfo(attn_bias.dtype).min
    attn_bias = attn_bias.masked_fill(cannot_attend, min_val)

    return attn_bias


class MPTPreTrainedModel(PreTrainedModel):
    config_class = MPTConfig
    base_model_prefix = 'model'
    _no_split_modules = ['MPTBlock']


def _fsdp_wrap_fn(
    self: Union[MPTModel, MPTForCausalLM],
    module: nn.Module,
) -> bool:
    # FSDP Wrap function for MPT Models
    if hasattr(module, '_fsdp_kwargs_dict'):
        return module._fsdp_kwargs_dict
    return isinstance(module, MPTBlock)


class MPTModel(MPTPreTrainedModel):

    def __init__(self, config: MPTConfig):
        config._validate_config()
        super().__init__(config)

        self.attn_impl = config.attn_config['attn_impl']
        self.attn_uses_sequence_id = config.attn_config['attn_uses_sequence_id']
        self.alibi = config.attn_config['alibi']
        self.alibi_bias_max = config.attn_config['alibi_bias_max']

        self.learned_pos_emb = config.learned_pos_emb

        if config.init_device == 'mixed':
            if dist.get_local_rank() == 0:
                config.init_device = 'cpu'
            else:
                config.init_device = 'meta'

        if config.norm_type.lower() not in norms.get_all():
            norm_options = ' | '.join(norms.get_all())
            raise NotImplementedError(
                f'Requested norm type ({config.norm_type}) is not implemented within this repo (Options: {norm_options}).',
            )

        # CogView (https://arxiv.org/abs/2105.13290) and GLM-130B (https://arxiv.org/abs/2210.02414)
        # both report this helping with stabilizing training
        self.embedding_fraction = config.embedding_fraction

        self.wte = SharedEmbedding(
            config.vocab_size,
            config.d_model,
            device=config.init_device,
        )
        if self.learned_pos_emb:
            self.wpe = torch.nn.Embedding(
                config.max_seq_len,
                config.d_model,
                device=config.init_device,
            )
        self.emb_drop = nn.Dropout(config.emb_pdrop)
        self.mb_args = None
        self.shift_labels = True

        self.blocks = self.construct_blocks(config=config,)

        # Tag all modules in the transformer blocks with the corresponding block_idx and max_block_idx
        for i, block in enumerate(self.blocks):
            block.block_idx = i
            block.max_block_idx = config.n_layers - 1
            pass_on_block_idx(block)

        self.norm_f = build_norm(
            name=config.norm_type.lower(),
            normalized_shape=config.d_model,
            device=config.init_device,
        )

        self.rope = config.attn_config['rope']
        self.rope_impl = None
        if self.rope:
            self.rope_impl = config.attn_config['rope_impl']
            self.rotary_embedding = gen_rotary_embedding(
                rope_head_dim=config.d_model // config.n_heads,
                rope_impl=self.rope_impl,
                rope_theta=config.attn_config['rope_theta'],
                rope_dail_config=config.attn_config['rope_dail_config'],
                rope_hf_config=config.attn_config['rope_hf_config'],
                max_seq_len=self.config.max_seq_len,
            )

        if config.init_device != 'meta':
            log.info(
                f'We recommend using config.init_device="meta" with Composer + FSDP for faster initialization.',
            )
            self.apply(self.param_init_fn)

        self.is_causal = True

        # define attn mask
        self._attn_bias_initialized = False
        self.attn_bias = None
        self.attn_bias_shape = attn_bias_shape(
            self.attn_impl,
            config.n_heads,
            config.max_seq_len,
            self.alibi,
            causal=self.is_causal,
            use_sequence_id=self.attn_uses_sequence_id,
        )

        if config.no_bias:
            for module in self.modules():
                if hasattr(module,
                           'bias') and isinstance(module.bias, nn.Parameter):
                    log.debug(f'Removing bias from {module=}.')
                    module.register_parameter('bias', None)

                # For transformer engine
                if hasattr(module, 'use_bias') and module.use_bias is True:
                    log.debug(f'Setting use_bias=False for {module=}.')
                    module.use_bias = False

        log.debug(self)
        log.debug(f'Using {self.config.init_config["name"]} initialization.')

    @property
    def block_class(self) -> Type[MPTBlock]:
        return MPTBlock

    def construct_blocks(self, config: MPTConfig) -> nn.ModuleList:
        """Construct the nn.ModuleList with the Transformer blocks.

        Args:
            config (MPTConfig): The configuration object.

        Returns:
            nn.ModuleList: The list of Transformer blocks.
        """
        block_args = self.extract_block_args(config.to_dict())
        self.kv_cache_layers = None

        if config.block_overrides is not None:
            return self._construct_blocks_with_overrides(config, block_args)

        return nn.ModuleList([
            self.block_class(
                device=config.init_device,
                **block_args,
            ) for _ in range(config.n_layers)
        ])

    def _construct_blocks_with_overrides(
        self,
        config: MPTConfig,
        block_args: Dict[str, Any],
    ) -> nn.ModuleList:
        if config.block_overrides is None:
            raise ValueError(
                'config.block_overrides should not be None when calling _construct_blocks_with_overrides.',
            )
        modules_order_expanded = {}
        for type in ['start', 'repeating_pattern', 'end']:
            modules_order_expanded[type] = []
            if type in config.block_overrides:
                for block in config.block_overrides[type]:
                    if not isinstance(block['repeat'],
                                      int) or block['repeat'] < 0:
                        raise ValueError(
                            'repeat should be a non-negative integer.',
                        )
                    modules_order_expanded[type].extend([block['name']] *
                                                        block['repeat'])

        start_len = len(modules_order_expanded['start'])
        repeating_pattern_len = len(modules_order_expanded['repeating_pattern'])
        end_len = len(modules_order_expanded['end'])

        if repeating_pattern_len > 0:
            if (
                config.n_layers - (start_len + end_len)
            ) % repeating_pattern_len != 0:
                raise ValueError(
                    'Number of layers should be divisible by the specified custom modules order.',
                )
            num_repetitions = (
                config.n_layers - (start_len + end_len)
            ) // repeating_pattern_len
            modules_order_expanded[
                'repeating_pattern'
            ] = modules_order_expanded['repeating_pattern'] * num_repetitions

        model_modules_order_expanded = modules_order_expanded[
            'start'] + modules_order_expanded['repeating_pattern'
                                             ] + modules_order_expanded['end']

        module_list = []
        layer_description_list = []
        if len(model_modules_order_expanded) != config.n_layers:
            raise ValueError(
                f'The specified block overrides do not match the number of layers: {len(model_modules_order_expanded)} vs {config.n_layers}.',
            )

        for i in range(config.n_layers):
            module_name = model_modules_order_expanded[i]
            override_config = {}
            if module_name != 'default':
                override_config = copy.deepcopy(
                    config.block_overrides['overrides'][module_name],
                )
                override_attn_config = override_config.get('attn_config', None)
                if override_attn_config is not None and 'reuse_kv_layer_idx' in override_attn_config:
                    if override_attn_config['reuse_kv_layer_idx'] >= 0:
                        raise ValueError(
                            f'The relative index of kv layer to reuse, {override_attn_config["reuse_kv_layer_idx"]=}, should be negative.',
                        )
                    reuse_kv_layer_idx = i + override_attn_config[
                        'reuse_kv_layer_idx']
                    if reuse_kv_layer_idx < 0:
                        raise ValueError(
                            f'The absolute index of kv layer to reuse, {reuse_kv_layer_idx} should be non-negative.',
                        )
                    override_attn_config['reuse_kv_layer_idx'
                                        ] = reuse_kv_layer_idx
                    if self.kv_cache_layers is None:
                        self.kv_cache_layers = set()
                    self.kv_cache_layers.add(reuse_kv_layer_idx)
            layer_description_list.append([
                i,
                module_name,
                self._get_overrides_for_logging(override_config),
            ],)
            new_block_args = self._override_block_args(
                block_args,
                override_config,
                config.allowed_block_overrides,
            )
            module_list.append(
                MPTBlock(
                    device=config.init_device,
                    **new_block_args,
                ),
            )
        log.info(
            'The following is a summary of overrides per layer.\n' + tabulate(
                layer_description_list,
                headers=['idx', 'name', 'overrides'],
            ),
        )
        return nn.ModuleList(module_list)

    def _get_overrides_for_logging(
        self,
        override_config: Dict[str, Any],
    ) -> List[dict[str, str]]:
        overrides_list = []
        for k, v in override_config.items():
            if isinstance(v, dict):
                overrides_list.extend(self._get_overrides_for_logging(v))
            else:
                overrides_list.append({k: v})
        return overrides_list

    def _override_block_args(
        self,
        block_args: Dict[str, Any],
        override_config: Dict[str, Any],
        allowed_block_overrides: set,
    ) -> Dict[str, Any]:
        new_block_args = override_config | block_args
        common_keys = override_config.keys() & block_args.keys()
        for k in common_keys:
            if type(override_config[k]) != type(block_args[k]):
                raise ValueError(
                    f'Override config should have same value types as the original config. Found override_config[{k}]={override_config[k]} vs block_args[{k}]={block_args[k]}.',
                )
            if isinstance(override_config[k], dict):
                new_block_args[k] = self._override_block_args(
                    block_args[k],
                    override_config[k],
                    allowed_block_overrides,
                )
            else:
                if k not in allowed_block_overrides:
                    raise KeyError(f'Overriding {k} is not supported.')
                new_block_args[k] = override_config[k]
        return new_block_args

    def extract_block_args(self, block_args: Dict[str, Any]) -> Dict[str, Any]:
        """Sets the block args."""
        if block_args['ffn_config']['ffn_type'] in ffns_with_megablocks:
            block_args['ffn_config'] = config_moe_args(
                block_args['ffn_config'],
                block_args['d_model'],
                block_args['expansion_ratio'],
                block_args['n_layers'],
            )
            self.mb_args = block_args['ffn_config'].get('args')
        return block_args

    def get_input_embeddings(self) -> Union[SharedEmbedding, nn.Embedding]:
        return self.wte

    def set_input_embeddings(
        self,
        value: Union[SharedEmbedding, nn.Embedding],
    ) -> None:
        self.wte = value

    @torch.no_grad()
    def _attn_bias(
        self,
        device: torch.device,
        dtype: torch.dtype,
        attention_mask: Optional[torch.ByteTensor] = None,
        sequence_id: Optional[torch.LongTensor] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.ByteTensor]]:
        if not self._attn_bias_initialized:
            if self.attn_bias_shape:
                self.attn_bias = torch.zeros(
                    self.attn_bias_shape,
                    device=device,
                    dtype=dtype,
                )
                self.attn_bias = build_attn_bias(
                    self.attn_impl,
                    self.attn_bias,
                    self.config.n_heads,
                    self.config.max_seq_len,
                    causal=self.is_causal,
                    alibi=self.alibi,
                    alibi_bias_max=self.alibi_bias_max,
                )
            self._attn_bias_initialized = True

        # flash will incorporate any attention_mask inside the attention module
        if self.attn_impl == 'flash':
            return self.attn_bias, attention_mask

        if self.attn_bias is not None:
            # .to(*args, **kwargs) is a no-op if tensor is already on
            # specified device or of specified dtype
            self.attn_bias = self.attn_bias.to(dtype=dtype, device=device)

        attn_bias = self.attn_bias

        # If using torch, we incorporate sequence_id (if appropriate)
        if self.attn_uses_sequence_id and sequence_id is not None:
            assert isinstance(attn_bias, torch.Tensor)  # pyright
            attn_bias = apply_sequence_id(
                attn_bias,
                sequence_id,
                self.config.max_seq_len,
            )

        # If using torch, we incorporate attention_mask. This will output
        # None in place of attention_mask since it will not be further needed in the
        # attention modules.
        if attention_mask is not None:
            s_k = attention_mask.shape[-1]
            if attn_bias is None:
                attn_bias = torch.zeros((1, 1, 1, s_k),
                                        device=device,
                                        dtype=dtype)
            else:
                # clamp to 0 necessary for torch 2.0 compile()
                _s_k = max(0, attn_bias.size(-1) - s_k)
                attn_bias = attn_bias[:, :, :, _s_k:]
            min_val = torch.finfo(attn_bias.dtype).min
            attn_bias = attn_bias.masked_fill(
                ~attention_mask.view(-1, 1, 1, s_k),
                min_val,
            )

        return attn_bias, attention_mask

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.FloatTensor]]] = None,
        attention_mask: Optional[torch.ByteTensor] = None,
        sequence_id: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        use_cache: Optional[bool] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> BaseModelOutputWithPast:
        return_dict = (
            return_dict if return_dict is not None else self.config.return_dict
        )
        use_cache = (
            use_cache if use_cache is not None else self.config.use_cache
        )

        if attention_mask is not None:
            attention_mask = attention_mask.bool()  # type: ignore

        # These args are passed in by keyword in huggingface's generate function
        # https://github.com/huggingface/transformers/blob/68287689f2f0d8b7063c400230b3766987abf18d/src/transformers/generation/utils.py#L2201-L2206
        # but have not yet been fully implemented in MPTModel
        if not return_dict:
            raise NotImplementedError(
                'return_dict False is not implemented yet for MPT',
            )
        if output_attentions:
            if self.attn_impl != 'torch':
                raise NotImplementedError(
                    'output_attentions is not implemented for MPT when using attn_impl `flash`.',
                )

        if (
            self.training and attention_mask is not None and
            attention_mask[:, 0].sum() != attention_mask.shape[0]
        ):
            raise NotImplementedError(
                'MPT does not support training with left padding.',
            )

        if self.training:
            if self.attn_uses_sequence_id and sequence_id is None:
                raise ValueError(
                    'sequence_id is a required argument when MPT is configured with attn_uses_sequence_id=True '
                    + 'and the model is in train mode.',
                )
            elif (
                self.attn_uses_sequence_id is False and sequence_id is not None
            ):
                warnings.warn(
                    'MPT received non-None input for `sequence_id` but is configured with attn_uses_sequence_id=False. '
                    +
                    'This input will be ignored. If you want the model to use `sequence_id`, set attn_uses_sequence_id to True.',
                )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                'You cannot specify both input_ids and inputs_embeds.',
            )
        elif input_ids is not None:
            bsz = input_ids.size(0)
            x = self.wte(input_ids)
            input_device = input_ids.device
        elif inputs_embeds is not None:
            bsz = inputs_embeds.size(0)
            x = inputs_embeds
            input_device = inputs_embeds.device
        else:
            raise ValueError('You must specify input_ids or inputs_embeds')

        S = self.get_sequence_length(x)

        assert (
            S <= self.config.max_seq_len
        ), f'Cannot forward input with seq_len={S}, this model only supports seq_len<={self.config.max_seq_len}'

        rotary_emb_w_meta_info = None

        past_position = 0
        if past_key_values is not None:
            if len(past_key_values) != self.config.n_layers:
                raise ValueError(
                    f'past_key_values must provide a past_key_value for each attention '
                    +
                    f'layer in the network ({len(past_key_values)=}; {self.config.n_layers=}).',
                )
            # For attn_impl: flash, the past key tensor spec is (batch, seq, dim).
            # For attn_impl: torch, the past key tensor spec is (batch, heads, head_dim, seq).
            # Here we shift position embedding using the `seq` dim of the past key
            past_position = past_key_values[0][0].size(1)
            if self.attn_impl == 'torch':
                past_position = past_key_values[0][0].size(3)

        if self.learned_pos_emb or self.rope:
            if self.learned_pos_emb and (
                S + past_position > self.config.max_seq_len
            ):
                raise ValueError(
                    f'Cannot forward input with past sequence length {past_position} and current sequence length '
                    +
                    f'{S + 1}, this model only supports total sequence length <= {self.config.max_seq_len}.',
                )

            if self.learned_pos_emb or (self.rope and self.rope_impl == 'hf'):
                pos = torch.arange(
                    past_position,
                    S + past_position,
                    dtype=torch.long,
                    device=input_device,
                ).unsqueeze(0)
                if attention_mask is not None:
                    # adjust the position indices to account for padding tokens
                    pos = torch.clamp(
                        pos - torch.cumsum((~attention_mask).to(torch.int32),
                                           dim=1)[:, past_position:],
                        min=0,
                    )
                if self.learned_pos_emb:
                    x = x + self.wpe(pos)
                elif self.rope and self.rope_impl == 'hf':
                    rotary_emb_w_meta_info = {
                        'impl': self.rope_impl,
                        'rotary_emb': self.rotary_embedding,
                        'offset_info': pos,
                        'seq_len': S + past_position,
                    }
            elif self.rope and self.rope_impl == 'dail':
                rotary_emb_w_meta_info = {
                    'impl': self.rope_impl,
                    'rotary_emb': self.rotary_embedding,
                    'offset_info': past_position,
                    'seq_len': S + past_position,
                }

        if self.embedding_fraction == 1:
            x = self.emb_drop(x)
        else:
            # this implementation is proposed on page 7 of the GLM-130B paper https://arxiv.org/abs/2210.02414
            x_shrunk = (x * self.embedding_fraction
                       ) + (x.detach() * (1 - self.embedding_fraction))
            assert isinstance(self.emb_drop, nn.Module)  # pyright
            x = self.emb_drop(x_shrunk)

        attn_bias, attention_mask = self._attn_bias(
            device=x.device,
            dtype=torch.float32,
            attention_mask=attention_mask,
            sequence_id=sequence_id,
        )
        attention_mask_in_length = gen_attention_mask_in_length(
            sequence_id=sequence_id,
            S=S,
            attn_uses_sequence_id=self.attn_uses_sequence_id,
            attn_impl=self.attn_impl,
            attention_mask=attention_mask,
        )

        alibi_slopes = None  # alibi_slopes will only be used by flash attention for ALiBi
        if self.alibi and self.attn_impl == 'flash':
            alibi_slopes = gen_slopes(
                n_heads=self.config.n_heads,
                alibi_bias_max=self.alibi_bias_max,
                device=x.device,
                return_1d=True,
            )

        # initialize the past key values cache if it should be used
        presents = () if use_cache else None
        if (
            use_cache or self.kv_cache_layers is not None
        ) and past_key_values is None:
            past_key_values = [() for _ in range(self.config.n_layers)
                              ]  # type: ignore

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        flash_attn_padding_info = {}
        if self.attn_impl == 'flash':
            flash_attn_padding_info = gen_flash_attn_padding_info(
                bsz,
                S,
                past_position,
                x.device,
                attention_mask_in_length,
                attention_mask,
            )

        layer_kv_cache_dict = {}
        for b_idx, block in enumerate(self.blocks):
            attn_block = block.attn if hasattr(
                block,
                'attn',
            ) else block.norm_attn_norm.attn
            if attn_block.reuse_kv_layer_idx is not None:
                if attn_block.reuse_kv_layer_idx not in layer_kv_cache_dict:
                    raise KeyError(
                        f'kv cache for layer {block.reuse_kv_layer_idx} not found in {layer_kv_cache_dict=}.',
                    )
                prev_layer_key_value = layer_kv_cache_dict[
                    attn_block.reuse_kv_layer_idx]
            else:
                prev_layer_key_value = None
            if output_hidden_states:
                assert all_hidden_states is not None  # pyright
                all_hidden_states = all_hidden_states + (x,)
            past_key_value = (
                past_key_values[b_idx] if past_key_values is not None else None
            )
            x, attn_weights, present = block(
                x,
                past_key_value=past_key_value,
                attn_bias=attn_bias,
                rotary_emb_w_meta_info=rotary_emb_w_meta_info,
                attention_mask=attention_mask,
                is_causal=self.is_causal,
                output_attentions=bool(output_attentions),
                alibi_slopes=alibi_slopes,
                flash_attn_padding_info=flash_attn_padding_info,
                prev_layer_key_value=prev_layer_key_value,
            )
            if presents is not None:
                presents += (present,)
            if self.kv_cache_layers is not None and b_idx in self.kv_cache_layers:
                layer_kv_cache_dict[b_idx] = [
                    present[0][:, past_position:],
                    present[1][:, past_position:],
                ]

            if output_attentions:
                assert all_self_attns is not None  # pyright
                all_self_attns = all_self_attns + (attn_weights,)

        x = self.norm_f(x)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            assert all_hidden_states is not None  # pyright
            all_hidden_states = all_hidden_states + (x,)

        return BaseModelOutputWithPast(
            last_hidden_state=x,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def get_sequence_length(self, x: torch.Tensor) -> int:
        """Returns the sequence length.

        Args:
            x (torch.Tensor): The input Tensor.

        Returns:
            S (int): The sequence length.
        """
        return x.size(1)

    # Param Initialization, needed for device='meta' fast initialization
    def param_init_fn(self, module: nn.Module) -> None:
        init_fn_name = self.config.init_config['name']
        param_init_fns.get(init_fn_name)(
            module=module,
            n_layers=self.config.n_layers,
            d_model=self.config.d_model,
            **self.config.init_config,
        )

    # FSDP Wrap function
    def fsdp_wrap_fn(self, module: nn.Module) -> bool:
        return _fsdp_wrap_fn(self, module)

    # Activation Checkpointing
    def activation_checkpointing_fn(self, module: nn.Module) -> bool:
        return isinstance(module, MPTBlock)


class MPTForCausalLM(MPTPreTrainedModel):

    def __init__(self, config: MPTConfig):
        super().__init__(config)
        log.info(f'Instantiating an MPTForCausalLM model from {__file__}')

        self.transformer: MPTModel = self.backbone_model_class(config)

        self.lm_head = None
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(
                config.d_model,
                config.vocab_size,
                bias=False,
                device=config.init_device,
            )
            self.lm_head._fsdp_wrap = True

        for child in self.transformer.children():
            if isinstance(child, torch.nn.ModuleList):
                continue
            if isinstance(child, torch.nn.Module):
                child._fsdp_wrap = True

        # enables scaling output logits; similar to a softmax "temperature"
        # PaLM paper uses scale 1/sqrt(config.d_model)
        self.logit_scale = None
        if config.logit_scale is not None:
            logit_scale = config.logit_scale
            if isinstance(logit_scale, str):
                if logit_scale == 'inv_sqrt_d_model':
                    logit_scale = 1 / math.sqrt(config.d_model)
                else:
                    raise ValueError(
                        f"{logit_scale=} is not recognized as an option; use numeric value or 'inv_sqrt_d_model'.",
                    )
            self.logit_scale = logit_scale

    @property
    def backbone_model_class(self) -> Type[MPTModel]:
        return MPTModel

    def get_input_embeddings(self) -> Union[SharedEmbedding, nn.Embedding]:
        return self.transformer.get_input_embeddings()

    def set_input_embeddings(
        self,
        value: Union[SharedEmbedding, nn.Embedding],
    ) -> None:
        self.transformer.set_input_embeddings(value)

    def get_output_embeddings(
        self,
    ) -> Union[SharedEmbedding, nn.Embedding, nn.Linear]:
        if self.lm_head is not None:
            return self.lm_head
        return self.transformer.get_input_embeddings()

    def set_output_embeddings(
        self,
        new_embeddings: Union[SharedEmbedding, nn.Embedding, nn.Linear],
    ) -> None:
        if self.lm_head is not None:
            self.lm_head = new_embeddings
        else:
            if not isinstance(new_embeddings, (SharedEmbedding, nn.Embedding)):
                raise ValueError(
                    'new_embeddings must be an instance of SharedEmbedding ' +
                    f'or nn.Embedding, but got {type(new_embeddings)}.',
                )
            warnings.warn(
                'Using `set_output_embeddings` to set the embedding layer of ' +
                'MPTForCausalLM with tied weights. Given weights are tied, ' +
                'using `set_input_embeddings` is recommended over using ' +
                '`set_output_embeddings`.',
            )
            self.transformer.set_input_embeddings(new_embeddings)

    def tie_weights(self) -> None:
        if getattr(self.config, 'tie_word_embeddings', True):
            self.lm_head = None

    def set_decoder(self, decoder: MPTModel) -> None:
        self.transformer = decoder

    def get_decoder(self) -> MPTModel:
        return self.transformer

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.FloatTensor]]] = None,
        attention_mask: Optional[torch.ByteTensor] = None,
        sequence_id: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        use_cache: Optional[bool] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> CausalLMOutputWithPast:
        return_dict = (
            return_dict if return_dict is not None else self.config.return_dict
        )
        use_cache = (
            use_cache if use_cache is not None else self.config.use_cache
        )

        outputs = self.transformer(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            sequence_id=sequence_id,
            return_dict=return_dict,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            inputs_embeds=inputs_embeds,
        )

        if self.lm_head is not None:
            logits = self.lm_head(outputs.last_hidden_state)
        else:
            # move outputs to same device as weights for token embedding
            # needed to support HF `device_map`
            out = outputs.last_hidden_state
            out = out.to(self.transformer.wte.weight.device)
            logits = self.transformer.wte(out, True)

        if self.logit_scale is not None:
            if self.logit_scale == 0:
                warnings.warn(
                    f'Multiplying logits by {self.logit_scale=}. This will produce uniform (uninformative) outputs.',
                )
            logits *= self.logit_scale

        loss = None
        if labels is not None:
            _labels = torch.roll(labels, shifts=-1)
            _labels[:, -1] = -100
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                _labels.to(logits.device).view(-1),
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    # Param Initialization, needed for device='meta' fast initialization
    def param_init_fn(self, module: nn.Module) -> None:
        init_fn_name = self.config.init_config['name']
        param_init_fns.get(init_fn_name)(
            module=module,
            n_layers=self.config.n_layers,
            d_model=self.config.d_model,
            **self.config.init_config,
        )

    # FSDP Wrap function
    def fsdp_wrap_fn(self, module: nn.Module) -> bool:
        return _fsdp_wrap_fn(self, module)

    # Activation Checkpointing
    def activation_checkpointing_fn(self, module: nn.Module) -> bool:
        """The MPT activation checkpointing (act ckpt) function.

        When `activation_checkpointing` in fsdp_config is set to true, this function will be called on all the modules in the FSDP wrapped model and determine whether a given module should be activation checkpointed. It checks the checkpointing target (`activation_checkpointing_target` in `model`) which can be specified as below:
            1. null (or no such field): The whole MPTBlock will be activation checkpointed on all layers
            2. a list of modules to act ckpt on all layers, e.g.,
                activation_checkpointing_target:
                    - grouped_query_attention
                    - mptmlp
            3. a dictionary of module name with target_blocks, e.g.,
                activation_checkpointing_target:
                    {
                            "mptblock": target_blocks_1,
                            "grouped_query_attention": target_blocks_2
                    }
                target_blocks (target_blocks_1, target_blocks_2 above) can be:
                - a single integer n: the first n transformer block will be activation checkpointed
                - a string of first-n, middle-m, last-k, range-i-j: the first n, the middle m,  the last k, or the range [i, j) layers will be activation checkpointed. E.g, 'first-2, last-2' means the first 2 and last 2 transformer blocks will be activation checkpointed
                    middle-m is range [start, end) where ``start = max(max_block_idx // 2 - m // 2, 0), end = min(start + m, max_block_idx + 1)``
                - a list of integers corresponds to the list of transformer block ids, e.g., [2] means the second transformer block will be activation checkpointed. [2, 3] means the second and third transformer blocks will be activation checkpointed
                - a list of mixed integers and strings of first-n, middle-m, last-k, range-i-j

            An example in yaml config file:
                fsdp_config:
                    activation_checkpointing: true
                model:
                    activation_checkpointing_target:
                        {
                            "mptblock": 'first-5',
                            "grouped_query_attention": 'last-35'
                        }
        """
        if not hasattr(module, 'block_idx'):
            log.debug(
                f'{module.__class__.__name__} cannot be activation checkpointed. Only transformer block or its submodules are eligible for activation checkpointing.',
            )
            return False

        act_ckpt_target = getattr(
            self.config,
            'activation_checkpointing_target',
            None,
        )
        act_ckpt_mod_to_blocks = build_act_ckpt_mod_to_blocks(
            act_ckpt_target,
            MPTBlock,
            module.max_block_idx,
        )

        check_mapping_blocks_overlap(
            act_ckpt_mod_to_blocks,
            module.max_block_idx,
        )

        for k in act_ckpt_mod_to_blocks.keys():
            if isinstance(module, k):
                blocks = act_ckpt_mod_to_blocks[k]
                return True if blocks == -1 else module.block_idx in blocks

        return False

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[List[Tuple[torch.Tensor,
                                             torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        attention_mask = kwargs['attention_mask'].bool()
        if attention_mask[:, -1].sum() != attention_mask.shape[0]:
            raise NotImplementedError(
                'MPT does not support generation with right padding.',
            )

        if self.transformer.attn_uses_sequence_id and self.training:
            sequence_id = torch.zeros_like(input_ids[:1])
        else:
            sequence_id = None

        # only last token for inputs_ids if past is defined in kwargs
        if past_key_values is not None:
            input_ids = input_ids[:, -1].unsqueeze(-1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {'inputs_embeds': inputs_embeds}
        else:
            model_inputs = {'input_ids': input_ids}

        model_inputs.update({
            'attention_mask': attention_mask,
            'sequence_id': sequence_id,
            'past_key_values': past_key_values,
            'use_cache': kwargs.get('use_cache', True),
        })
        return model_inputs

    @staticmethod
    def _reorder_cache(
        past_key_values: List[Tuple[torch.Tensor, torch.Tensor]],
        beam_idx: torch.LongTensor,
    ) -> List[Tuple[torch.Tensor, ...]]:
        """Used by HuggingFace generate when using beam search with kv-caching.

        See https://github.com/huggingface/transformers/blob/3ec7a47664ebe40c40f4b722f6bb1cd30c3821ec/src/transformers/models/gpt2/modeling_gpt2.py#L1122-L1133
        for an example in transformers.
        """
        reordered_past = []
        for layer_past in past_key_values:
            reordered_past += [
                tuple(
                    past_state.index_select(0, beam_idx)
                    for past_state in layer_past
                ),
            ]
        return reordered_past


class ComposerMPTCausalLM(HuggingFaceModel):

    def __init__(
        self,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        use_train_metrics: Optional[bool] = True,
        additional_train_metrics: Optional[List] = None,
        loss_fn: Optional[Union[str, Dict]] = 'fused_crossentropy',
        **kwargs: Dict[str, Any],
    ):
        from llmfoundry.metrics import (
            DEFAULT_CAUSAL_LM_EVAL_METRICS,
            DEFAULT_CAUSAL_LM_TRAIN_METRICS,
        )
        from llmfoundry.utils.builders import build_metric

        additional_train_metrics = additional_train_metrics or []

        model = self.model_class(self.config_class(**kwargs))

        use_train_metrics = use_train_metrics
        train_metric_names = DEFAULT_CAUSAL_LM_TRAIN_METRICS + additional_train_metrics
        train_metrics = [
            build_metric(metric, {}) for metric in train_metric_names
        ] if use_train_metrics else []
        eval_metric_names = DEFAULT_CAUSAL_LM_EVAL_METRICS + additional_train_metrics
        eval_metrics = [
            build_metric(metric, {}) for metric in eval_metric_names
        ]

        super().__init__(
            model=model,
            tokenizer=tokenizer,
            use_logits=True,
            metrics=train_metrics,
            eval_metrics=eval_metrics,
            shift_labels=model.transformer.shift_labels,
            allow_embedding_resizing=True,
        )

        loss_fn_config = loss_fn
        if loss_fn_config == 'fused_crossentropy':
            try:
                from flash_attn.losses.cross_entropy import \
                    CrossEntropyLoss as FusedCrossEntropyLoss

                self.loss_fn = FusedCrossEntropyLoss(
                    ignore_index=-100,
                    reduction='none',
                )
            except:
                raise ValueError(
                    'Fused Cross Entropy is not installed. Either (1) have a CUDA-compatible GPU '
                    +
                    'and `pip install .[gpu]` if installing from source or `pip install xentropy-cuda-lib@git+https://github.com/HazyResearch/flash-attention.git@v1.0.3#subdirectory=csrc/xentropy` '
                    +
                    'if installing from pypi, or (2) set your config model.loss_fn=torch_crossentropy.',
                )
        elif loss_fn_config == 'torch_crossentropy':
            self.loss_fn = nn.CrossEntropyLoss(
                ignore_index=-100,
                reduction='none',
            )
        else:
            raise ValueError(
                f'Specified loss_fn={self.loss_fn} not recognized. `loss_fn` must be one of [`fused_crossentropy`, `torch_crossentropy`].',
            )

    @property
    def model_class(self) -> Type[MPTForCausalLM]:
        return MPTForCausalLM

    @property
    def config_class(self) -> Type[MPTConfig]:
        return MPTConfig

    def get_targets(self, batch: Mapping) -> torch.Tensor:
        targets = torch.roll(batch['labels'], shifts=-1)
        targets[:, -1] = -100
        return targets

    def forward(self, batch: MutableMapping) -> CausalLMOutputWithPast:
        if self.config.ffn_config['ffn_type'] in ffns_with_megablocks:
            # Clear MegaBlocks MoE load balancing loss cache
            try:  # Add try/catch to avoid transformers complaining and raising errors
                from megablocks.layers.moe import clear_load_balancing_loss
            except:
                raise RuntimeError(
                    'Requirements for MegaBlocks not installed; see install instructions in `README.md`.',
                )
            clear_load_balancing_loss()
        return self.model(
            input_ids=batch.get('input_ids', None),
            attention_mask=batch.get('attention_mask', None),
            sequence_id=batch.get('sequence_id', None),
            inputs_embeds=batch.get('inputs_embeds', None),
        )

    def loss(self, outputs: CausalLMOutputWithPast,
             batch: Mapping) -> Union[dict, torch.Tensor]:
        if self.shift_labels:
            targets = self.get_targets(batch)
        else:
            targets = batch['labels']

        losses = self.loss_fn(
            outputs.logits.view(-1, outputs.logits.size(-1)),
            targets.view(-1),
        )

        if torch.all(targets == self.loss_fn.ignore_index):
            loss = losses.sum()
        else:
            loss = losses.sum() / (targets != self.loss_fn.ignore_index).sum()
            if 'sample_weighing_factor' in batch:
                if batch['sample_weighing_factor'].shape[0] > 1:
                    raise ValueError(
                        'Sample weighing factor is not supported when batch["sample_weighing_factor"].shape[0] > 1.',
                    )
                loss = loss * batch['sample_weighing_factor'][0].item()

        if self.config.ffn_config['ffn_type'] in ffns_with_megablocks:
            # MegaBlocks MoE load balancing loss
            try:  # Add try/catch to avoid transformers complaining and raising errors
                from megablocks.layers.moe import batched_load_balancing_loss
            except:
                raise RuntimeError(
                    'Requirements for MegaBlocks not installed; see install instructions in `README.md`.',
                )
            lbl = batched_load_balancing_loss(self.model.transformer.mb_args)
            return {
                'total': loss + lbl,
                'loss': loss,
                'lbl': lbl,
            }

        return loss

    @cached_property
    def n_total_params(self):
        """Gets the total number of parameters in the model."""
        return mpt_get_total_params(self)

    @cached_property
    def n_active_params(self):
        """Gets the total number of active parameters in the model."""
        return mpt_get_active_params(self)

    def flops_per_batch(self, batch: Mapping):
        # Note: this computation does not take into account padding, and assumes
        # that the dataset has been constructed without padding. Additionally, we
        # assume the backward pass is approximately 2x the forward pass

        if self.model.config.block_overrides is not None:
            warnings.warn(
                'Warning, flop computation is not supported when using block overrides. Returning 0 flops per batch.',
            )
            return 0

        bs, msl = batch['input_ids'].shape[0:2]
        params = self.n_active_params
        params_flops_per_token = 2 * params
        params_flops_per_seq = params_flops_per_token * msl
        attn_flops_per_seq = self.get_attention_flops(msl)
        return (params_flops_per_seq + attn_flops_per_seq) * 3 * bs

    def get_attention_flops(self, msl: int) -> int:
        """Computes the attention flops for the batch.

        Args:
            msl (int): The batch sequence length.

        Returns:
            attn_flops (int): The attention flops.
        """
        return (
            self.model.config.n_layers * 2 * 2 *
            (self.model.config.d_model * (msl**2))
        )
