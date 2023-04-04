# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

"""A HuggingFace-style model configuration."""

from typing import Optional, Tuple, Union

from transformers import PretrainedConfig


class MosaicGPTConfig(PretrainedConfig):
    model_type = 'mosaic_gpt'

    def __init__(
        self,
        d_model: int = 2048,
        n_heads: int = 16,
        n_layers: int = 24,
        mlp_ratio: int = 4,
        max_seq_len: int = 2048,
        vocab_size: int = 50368,
        attn_pdrop: float = 0.0,
        resid_pdrop: float = 0.0,
        emb_pdrop: float = 0.0,
        attn_impl: str = 'triton',
        attn_qk_ln: bool = False,
        attn_clip_qkv: Optional[float] = None,
        softmax_scale: Optional[float] = None,
        prefix_lm: Optional[bool] = False,
        attn_uses_sequence_id: Optional[bool] = False,
        alibi: bool = False,
        alibi_bias_max: int = 8,
        init_device: str = 'cpu',
        logit_scale: Optional[Union[float, str]] = None,
        no_bias: bool = False,
        verbose: int = 0,
        param_init_fn: str = 'kaiming_normal_',
        init_div_is_residual: Union[int, float, str, bool] = True,
        init_std: float = 0.02,
        emb_init_std: Optional[float] = None,
        emb_init_uniform_lim: Optional[Union[Tuple[float, float],
                                             float]] = None,
        init_gain: float = 0,
        fan_mode: str = 'fan_in',
        init_nonlinearity: str = 'relu',
        embedding_fraction: float = 1.0,
        low_precision_layernorm: bool = True,
        use_cache: bool = False,
        **kwargs,
    ):
        """The MosaicGPT configuration class.

        Args:
            d_model (int): The size of the embedding dimension of the model.
            n_heads (int): The number of attention heads.
            n_layers (int): The number of layers in the model.
            mlp_ratio (int): The ratio of the up/down scale in the MLP.
            max_seq_len (int): The maximum sequence length of the model.
            vocab_size (int): The size of the vocabulary.
            attn_pdrop (float): The dropout probability for the attention layers.
            resid_pdrop (float): The dropout probability applied to the attention output before combining with residual.
            emb_pdrop (float): The dropout probability for the embedding layer.
            attn_impl (str): The attention implementation to use. One of 'torch', 'flash', or 'triton'.
            attn_qk_ln (bool): Whether to apply layer normalization to the queries and keys in the attention layer.
            attn_clip_qkv (Optional[float]): If not None, clip the queries, keys, and values in the attention layer to
                this value.
            softmax_scale (Optional[float]): If not None, scale the softmax in the attention layer by this value. If None,
                use the default scale of ``1/sqrt(d_keys)``.
            prefix_lm (Optional[bool]): Whether the model should operate as a Prefix LM. This requires passing an
                extra `prefix_mask` argument which indicates which tokens belong to the prefix. Tokens in the prefix
                can attend to one another bi-directionally. Tokens outside the prefix use causal attention.
            attn_uses_sequence_id (Optional[bool]): Whether to restrict attention to tokens that have the same sequence_id.
                When the model is in `train` mode, this requires passing an extra `sequence_id` argument which indicates
                which sub-sequence each token belongs to.
                Defaults to ``False`` meaning any provided `sequence_id` will be ignored.
            alibi (bool): Whether to use the alibi bias instead of position embeddings.
            alibi_bias_max (int): The maximum value of the alibi bias.
            init_device (str): The device to use for parameter initialization.
            logit_scale (Optional[Union[float, str]]): If not None, scale the logits by this value.
            no_bias (bool): Whether to use bias in all layers.
            verbose (int): The verbosity level. 0 is silent.
            param_init_fn (str): The parameter initialization scheme to use. One of 'default_', 'baseline_', 'kaiming_uniform_',
                'kaiming_normal_', 'neox_init_', 'small_init_', 'xavier_uniform_', or 'xavier_normal_'.
            init_div_is_residual (Union[int, float, str, bool]): Value to divide initial weights by if ``module._is_residual`` is True.
            init_std (float): The standard deviation of the normal distribution used to initialize the model,
                if using the baseline_ parameter initialization scheme.
            emb_init_std (Optional[float]): The standard deviation of the normal distribution used to initialize the embedding layer.
            emb_init_uniform_lim (Optional[Union[Tuple[float, float], float]]): The lower and upper limits of the uniform distribution
                used to initialize the embedding layer. Mutually exclusive with ``emb_init_std``.
            init_gain (float): The gain to use for parameter initialization with kaiming or xavier initialization schemes.
            fan_mode (str): The fan mode to use for parameter initialization with kaiming initialization schemes.
            init_nonlinearity (str): The nonlinearity to use for parameter initialization with kaiming initialization schemes.
            embedding_fraction (float): The fraction to scale the gradients of the embedding layer by.
            low_precision_layernorm (bool): Whether to use low precision layer normalization.
            use_cache (bool): Whether or not the model should return the last key/values attentions
        """
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.mlp_ratio = mlp_ratio
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.attn_pdrop = attn_pdrop
        self.resid_pdrop = resid_pdrop
        self.emb_pdrop = emb_pdrop
        self.attn_impl = attn_impl
        self.attn_qk_ln = attn_qk_ln
        self.attn_clip_qkv = attn_clip_qkv
        self.softmax_scale = softmax_scale
        self.prefix_lm = prefix_lm
        self.attn_uses_sequence_id = attn_uses_sequence_id
        self.alibi = alibi
        self.alibi_bias_max = alibi_bias_max
        self.init_device = init_device
        self.logit_scale = logit_scale
        self.no_bias = no_bias
        self.verbose = verbose
        self.param_init_fn = param_init_fn
        self.init_div_is_residual = init_div_is_residual
        self.init_std = init_std
        self.emb_init_std = emb_init_std
        self.emb_init_uniform_lim = emb_init_uniform_lim
        self.init_std = init_std
        self.init_gain = init_gain
        self.fan_mode = fan_mode
        self.init_nonlinearity = init_nonlinearity
        self.embedding_fraction = embedding_fraction
        self.low_precision_layernorm = low_precision_layernorm
        self.use_cache = use_cache
        if 'name' in kwargs:
            del kwargs['name']
        if 'loss_fn' in kwargs:
            del kwargs['loss_fn']
        super().__init__(**kwargs)

        self._validate_config()

    def _validate_config(self):
        if self.d_model % self.n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads')
        if any(prob < 0 or prob > 1
               for prob in [self.attn_pdrop, self.resid_pdrop, self.emb_pdrop]):
            raise ValueError(
                'attn_pdrop, resid_pdrop, emb_pdrop are probabilities and must be between 0 and 1'
            )
        if self.attn_impl not in ['torch', 'flash', 'triton']:
            raise ValueError(f'Unknown attn_impl={self.attn_impl}')
        if self.prefix_lm and self.attn_impl not in ['torch', 'triton']:
            raise NotImplementedError(
                'prefix_lm only implemented with torch and triton attention.')
        if self.alibi and self.attn_impl not in ['torch', 'triton']:
            raise NotImplementedError(
                'alibi only implemented with torch and triton attention.')
        if self.attn_uses_sequence_id and self.attn_impl not in [
                'torch', 'triton'
        ]:
            raise NotImplementedError(
                'attn_uses_sequence_id only implemented with torch and triton attention.'
            )
        if self.embedding_fraction > 1 or self.embedding_fraction <= 0:
            raise ValueError(
                'model.embedding_fraction must be between 0 (exclusive) and 1 (inclusive)!'
            )
        if isinstance(self.logit_scale,
                      str) and self.logit_scale != 'inv_sqrt_d_model':
            raise ValueError(
                f"{self.logit_scale=} is not recognized as an option; use numeric value or 'inv_sqrt_d_model'."
            )
