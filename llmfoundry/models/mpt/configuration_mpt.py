# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""A HuggingFace-style model configuration."""

import copy
import warnings
from typing import Any, Dict, Optional, Union

from transformers import PretrainedConfig

from llmfoundry.layers_registry import ffns_with_megablocks
from llmfoundry.models.layers.attention import (
    check_alibi_support,
    is_flash_v2_installed,
)
from llmfoundry.models.utils.config_defaults import (
    attn_config_defaults,
    fc_type_defaults,
    ffn_config_defaults,
    init_config_defaults,
)
from llmfoundry.utils.warnings import ExperimentalWarning


class MPTConfig(PretrainedConfig):
    model_type = 'mpt'

    def __init__(
        self,
        d_model: int = 2048,
        n_heads: int = 16,
        n_layers: int = 24,
        expansion_ratio: Union[int, float] = 4,
        max_seq_len: int = 2048,
        vocab_size: int = 50368,
        resid_pdrop: float = 0.0,
        emb_pdrop: float = 0.0,
        learned_pos_emb: bool = True,
        attn_config: Optional[Dict] = None,
        ffn_config: Optional[Dict] = None,
        init_device: str = 'cpu',
        logit_scale: Optional[Union[float, str]] = None,
        no_bias: bool = False,
        embedding_fraction: float = 1.0,
        norm_type: str = 'low_precision_layernorm',
        use_cache: bool = False,
        init_config: Optional[Dict] = None,
        fc_type: Union[str, Dict] = 'torch',
        tie_word_embeddings: bool = True,
        use_pad_tok_in_ffn: bool = True,
        block_overrides: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        """The MPT configuration class.

        Args:
            d_model (int): The size of the embedding dimension of the model.
            n_heads (int): The number of attention heads.
            n_layers (int): The number of layers in the model.
            expansion_ratio (Union[int, float]): The ratio of the up/down scale in the ffn.
            max_seq_len (int): The maximum sequence length of the model.
            vocab_size (int): The size of the vocabulary.
            resid_pdrop (float): The dropout probability applied to the attention output before combining with residual.
            emb_pdrop (float): The dropout probability for the embedding layer.
            learned_pos_emb (bool): Whether to use learned positional embeddings
            attn_config (Dict): A dictionary used to configure the model's attention module:
                attn_type (str): type of attention to use. Options: multihead_attention, multiquery_attention, grouped_query_attention
                attn_pdrop (float): The dropout probability for the attention layers.
                attn_impl (str): The attention implementation to use. One of 'torch' or 'flash'.
                qk_ln (bool): Whether to apply layer normalization to the queries and keys in the attention layer.
                qk_gn (bool): Whether to apply group normalization to the queries and keys in the attention layer.
                clip_qkv (Optional[float]): If not None, clip the queries, keys, and values in the attention layer to
                    this value.
                softmax_scale (Optional[float]): If not None, scale the softmax in the attention layer by this value. If None,
                    use the default scale of ``1/sqrt(d_keys)``.
                attn_uses_sequence_id (Optional[bool]): Whether to restrict attention to tokens that have the same sequence_id.
                    When the model is in `train` mode, this requires passing an extra `sequence_id` argument which indicates
                    which sub-sequence each token belongs to.
                    Defaults to ``False`` meaning any provided `sequence_id` will be ignored.
                sliding_window_size (int): Window size for sliding window local attention. Defaults to -1, which means no sliding window. Query at position i will only attend to keys between [i + seqlen_k - seqlen_q - window_size, i + seqlen_k - seqlen_q + window_size] inclusive. Only works for flash attention v2.3.0 or higher.
                alibi (bool): Whether to use the alibi bias instead of position embeddings.
                alibi_bias_max (int): The maximum value of the alibi bias.
                rope (bool): Whether to use rotary positional embeddings.
                rope_theta (int): The base frequency for rope.
                rope_impl (str): The implementation of rope to use. One of 'hf' (to use the implementation from https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) or 'dail' (to use the implementation from https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/layers/rotary.py).
                rope_dail_config (Dict): The configuration for the dail implementation of rope.
                    type (str): The type of rotary position embedding to use. Options: 'original' (for https://arxiv.org/pdf/2104.09864.pdf), 'xpos' (for https://arxiv.org/pdf/2212.10554.pdf).
                    pos_idx_in_fp32 (bool): If True, the position indices [0, ..., seqlen - 1] are in fp32, otherwise they might be in lower precision. A consequence could be, for example, that bf16 rounds position 1995 to 2000, which leads to them having the same positional embedding.
                    xpos_scale_base (float): The scale base for XPos (if using XPos).
                rope_hf_config (Dict): A dictionary used to configure rope's scaling behavior (when scaling beyond the training length).
                    type (str): Can be one of 'no_scaling', 'linear', or 'dynamic'. 'no_scaling' uses the default implementation for rotary embeddings, 'linear' uses linear scaling as proposed by the Reddit user /u/kaiokendev, and 'dynamic' uses Dynamic NTK scaling as proposed by the Reddit users /u/bloc97 and /u/emozilla.
                    factor (float): Scaling factor to use if using 'linear' or 'dynamic' as rope_scaling.type.
                kv_n_heads (Optional[int]): For grouped_query_attention only, allow user to specify number of kv heads.
            ffn_config (Dict): A dictionary used to configure the model's ffn module:
                ffn_type (str): type of ffn to use. Options: mptmlp, mptglu, te_ln_mlp
            init_device (str): The device to use for parameter initialization.
            logit_scale (Optional[Union[float, str]]): If not None, scale the logits by this value.
            no_bias (bool): Whether to use bias in all layers.
            embedding_fraction (float): The fraction to scale the gradients of the embedding layer by.
            norm_type (str): choose type of norm to use
            use_cache (bool): Whether or not the model should return the last key/values attentions
            init_config (Dict): A dictionary used to configure the model initialization:
                init_config.name: The parameter initialization scheme to use. Options: 'default_', 'baseline_',
                    'kaiming_uniform_', 'kaiming_normal_', 'neox_init_', 'small_init_', 'xavier_uniform_', or
                    'xavier_normal_'. These mimic the parameter initialization methods in PyTorch.
                init_div_is_residual (Union[int, float, str, bool]): Value to divide initial weights by if ``module._is_residual`` is True.
                emb_init_std (Optional[float]): The standard deviation of the normal distribution used to initialize the embedding layer.
                emb_init_uniform_lim (Optional[Union[Tuple[float, float], float]]): The lower and upper limits of the uniform distribution
                    used to initialize the embedding layer. Mutually exclusive with ``emb_init_std``.
                init_std (float): The standard deviation of the normal distribution used to initialize the model,
                    if using the baseline_ parameter initialization scheme.
                init_gain (float): The gain to use for parameter initialization with kaiming or xavier initialization schemes.
                fan_mode (str): The fan mode to use for parameter initialization with kaiming initialization schemes.
                init_nonlinearity (str): The nonlinearity to use for parameter initialization with kaiming initialization schemes.
                ---
                See llmfoundry.models.utils.param_init_fns.py for info on other param init config options
            fc_type (str | Dict): Choose fc layer implementation. Options: torch and te. te layers support fp8 when using H100 GPUs. Can
                also be a dictionary that specifies the fc layer name and any kwargs for the fc layer.
            tie_word_embeddings (bool): Whether to tie the input embedding and output layers.
            use_pad_tok_in_ffn (bool): Whether to forward the pad token in the feedforward networks.
            block_overrides: This allows for overriding default block configs for certain layers. This must contain `overrides` and `order`. `order` is a nested list which describes the order of the layers. For each kind of layer, specify the `overrides` in the overrides config (default refers to a layer that does not apply any overrides).
                To specify this model (https://research.character.ai/optimizing-inference/) , the following config will be needed:
                    block_overrides:
                        order:
                        - name: default
                        - repeat: 2
                          order:
                          - name: sliding_window_layer
                          - name: sliding_window_layer_reuse
                          - name: sliding_window_layer
                          - repeat: 2
                            name: sliding_window_layer_reuse
                          - name: reuse_kv_layer
                        overrides:
                            sliding_window_layer:
                                attn_config:
                                    sliding_window_size: 1024
                            sliding_window_layer_reuse:
                                attn_config:
                                    sliding_window_size: 1024
                                    reuse_kv_layer_idx: -1
                            reuse_kv_layer:
                                attn_config:
                                    reuse_kv_layer_idx: -6
        """
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.expansion_ratio = expansion_ratio
        if max_seq_len != int(max_seq_len):
            raise ValueError('max_seq_len must be an integer')
        self.max_seq_len = int(max_seq_len)
        self.vocab_size = vocab_size
        self.resid_pdrop = resid_pdrop
        self.emb_pdrop = emb_pdrop
        self.learned_pos_emb = learned_pos_emb
        self.attn_config = attn_config if attn_config is not None else copy.deepcopy(
            attn_config_defaults,
        )
        self.ffn_config = ffn_config if ffn_config is not None else copy.deepcopy(
            ffn_config_defaults,
        )
        self.init_device = init_device
        self.logit_scale = logit_scale
        self.no_bias = no_bias
        self.embedding_fraction = embedding_fraction
        self.norm_type = norm_type
        self.use_cache = use_cache
        self.init_config = init_config if init_config is not None else copy.deepcopy(
            init_config_defaults,
        )

        if 'reuse_kv_layer_idx' in self.attn_config and self.attn_config[
            'attn_impl'] == 'torch':
            raise NotImplementedError(
                'reusing kv cache from a previous layer is not implemented for torch attention.',
            )
        if block_overrides is not None:
            self._validate_block_overrides(block_overrides)
        self.block_overrides = block_overrides

        if isinstance(fc_type, str):
            fc_type = {'name': fc_type}
        self.fc_type = fc_type

        self.use_pad_tok_in_ffn = use_pad_tok_in_ffn

        if 'name' in kwargs:
            del kwargs['name']
        if 'loss_fn' in kwargs:
            del kwargs['loss_fn']
        if self.attn_config.get('alibi',
                                False) or self.attn_config.get('rope', False):
            self.learned_pos_emb = False
            warnings.warn(
                f'alibi or rope is turned on, setting `learned_pos_emb` to `False.`',
            )
        # tie_word_embeddings is set in Huggingface's PretrainedConfig __init__
        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

        self._validate_config()

    def _validate_block_overrides(self, block_overrides: Dict[str, Any]):
        warnings.warn(ExperimentalWarning('block_overrides'))
        if 'start' not in block_overrides and 'repeating_pattern' not in block_overrides and 'end' not in block_overrides:
            raise ValueError(
                'either start, repeating_pattern, or end should be defined in block_overrides',
            )
        if 'overrides' not in block_overrides:
            raise ValueError('overrides should be defined in block_overrides',)
        for name, override in block_overrides['overrides'].items():
            if name == 'default':
                raise ValueError('block overrides cannot be named "default".',)
            if 'attn_config' in override and 'reuse_kv_layer_idx' in override[
                'attn_config'] and self.attn_config['attn_impl'] == 'torch':
                raise NotImplementedError(
                    'reusing kv cache from a previous layer is not implemented for torch attention.',
                )

    def _set_config_defaults(
        self,
        config: Dict[str, Any],
        config_defaults: Dict[str, Any],
    ) -> Dict[str, Any]:
        # set config defaults
        for k, v in config_defaults.items():
            if k not in config:
                config[k] = v
            elif isinstance(v, dict):
                # recursively set default values for any sub-dicts
                config[k] = self._set_config_defaults(
                    config[k] if (config[k] is not None) else {},
                    v,
                )
        return config

    def validate_attention_config(self) -> None:
        if 'seq_parallel_world_size' in self.attn_config and self.attn_config[
            'seq_parallel_world_size'] is None:
            del self.attn_config['seq_parallel_world_size']
        if self.attn_config.get('seq_parallel_world_size', 1) > 1:
            raise NotImplementedError('Sequence Parallelism is not supported.')

    def _validate_config(self) -> None:
        # set config defaults
        self.attn_config = self._set_config_defaults(
            self.attn_config,
            attn_config_defaults,
        )
        self.ffn_config = self._set_config_defaults(
            self.ffn_config,
            ffn_config_defaults,
        )
        self.init_config = self._set_config_defaults(
            self.init_config,
            init_config_defaults,
        )
        self.fc_type = self._set_config_defaults(
            self.fc_type,
            fc_type_defaults,
        )

        if self.d_model % self.n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads')
        if any(
            prob < 0 or prob > 1 for prob in
            [self.attn_config['attn_pdrop'], self.resid_pdrop, self.emb_pdrop]
        ):
            raise ValueError(
                "self.attn_config['attn_pdrop'], resid_pdrop, emb_pdrop are probabilities and must be between 0 and 1",
            )
        if self.attn_config['attn_impl'] not in ['torch', 'flash']:
            raise ValueError(
                f"Unknown attn_impl={self.attn_config['attn_impl']}",
            )
        if self.attn_config['alibi'] and not check_alibi_support(
            self.attn_config['attn_impl'],
        ):
            raise NotImplementedError(
                'alibi only implemented with torch and flash (v2.4.2 or higher) attention.',
            )
        if self.attn_config['attn_uses_sequence_id'] and not (
            self.attn_config['attn_impl'] == 'torch' or (
                self.attn_config['attn_impl'] == 'flash' and
                is_flash_v2_installed(v2_version='v2.1.2')
            )
        ):
            raise NotImplementedError(
                'attn_uses_sequence_id only implemented with torch and flash (v2.1.2 or higher) attention.',
            )
        if self.attn_config['rope'] and (
            self.attn_config['rope_impl'] not in ['dail', 'hf']
        ):
            raise ValueError(
                'If rope is being used then rope_impl should be either "dail", or "hf".',
            )
        if self.attn_config['rope'] and (
            self.attn_config['rope_impl'] == 'hf'
        ) and self.attn_config['rope_hf_config']['type'] not in [
            'no_scaling',
            'linear',
            'dynamic',
        ]:
            raise ValueError(
                'If using hf implementation of rope, the type should be one of "no_scaling", "linear" or "dynamic".',
            )
        if self.attn_config['rope'] and (
            self.attn_config['rope_impl'] == 'dail'
        ):
            if self.attn_config['rope_dail_config']['type'] not in [
                'original',
                'xpos',
            ]:
                raise ValueError(
                    'If using the dail implementation of rope, the type should be one of "original" or "xpos".',
                )
            if not is_flash_v2_installed(v2_version='2.0.1'):
                raise ImportError(
                    'If using the dail implementation of rope, the flash_attn library v2.0.1 or higher must be installed. Please check the instructions at https://github.com/mosaicml/llm-foundry/blob/main/TUTORIAL.md#what-kinds-of-positional-embeddings-does-llm-foundry-support',
                )
        if self.attn_config['sliding_window_size'] != -1 and not (
            self.attn_config['attn_impl'] == 'flash' and
            is_flash_v2_installed(v2_version='v2.3.0')
        ):
            raise NotImplementedError(
                'sliding window only implemented with flash attention v2.3.0 or higher.',
            )
        if self.embedding_fraction > 1 or self.embedding_fraction <= 0:
            raise ValueError(
                'model.embedding_fraction must be between 0 (exclusive) and 1 (inclusive)!',
            )
        if isinstance(
            self.logit_scale,
            str,
        ) and self.logit_scale != 'inv_sqrt_d_model':
            raise ValueError(
                f"{self.logit_scale=} is not recognized as an option; use numeric value or 'inv_sqrt_d_model'.",
            )
        if self.init_config.get('name', None) is None:
            raise ValueError(f"{self.init_config=} 'name' needs to be set.")
        if not (
            self.learned_pos_emb or self.attn_config['alibi'] or
            self.attn_config['rope']
        ):
            warnings.warn(
                f'Positional information not being provided to the model using either learned_pos_emb or alibi or rope.',
            )
        if self.fc_type['name'] == 'te' or self.ffn_config['ffn_type'
                                                          ] == 'te_ln_mlp':
            try:
                import transformer_engine.pytorch as te
                del te  # unused
            except:
                raise ImportError(
                    'TransformerEngine import fail. `fc_type: te` requires TransformerEngine be installed. '
                    +
                    'The required version of transformer_engine also requires FlashAttention v1.0.6 is installed:\n'
                    + 'pip install flash-attn==1.0.6 --no-build-isolation \n' +
                    'pip install git+https://github.com/NVIDIA/TransformerEngine.git@144e4888b2cdd60bd52e706d5b7a79cb9c1a7156',
                )

        self.ffn_config['fc_type'] = self.fc_type
        if self.ffn_config['ffn_type'] == 'mptgeglu':
            raise ValueError(
                'API CHANGE: `ffn_type=="mptgeglu"` changed to `ffn_type=="mptglu"`. '
                +
                'See [#829](https://github.com/mosaicml/llm-foundry/pull/829) for details.',
            )
        elif self.ffn_config['ffn_type'] in ffns_with_megablocks:
            self.ffn_config['return_bias'] = False
        elif self.ffn_config['ffn_type'] == 'te_ln_mlp':
            self.ffn_config['bias'] = not self.no_bias
            if 'ffn_act_fn' in self.ffn_config.keys():
                raise ValueError(
                    f'Transformer Engine block does not support custom activation functions.',
                )
        if not self.use_pad_tok_in_ffn:
            try:
                from flash_attn.bert_padding import unpad_input, pad_input  # type: ignore # yapf: disable # isort: skip
            except:
                raise ImportError(
                    'In order to set `use_pad_tok_in_ffn=False`, please install flash-attn==1.0.9 or flash-attn==2.3.6',
                )

        self.validate_attention_config()

    @property
    def allowed_block_overrides(self):
        return {
            'sliding_window_size',
            'reuse_kv_layer_idx',
        }
