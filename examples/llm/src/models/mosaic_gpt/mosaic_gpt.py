# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

"""A simple, flexible implementation of a GPT model.

Inspired by https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
"""

import math
import warnings
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from composer.algorithms.low_precision_layernorm.low_precision_layernorm import \
    LPLayerNorm
from composer.metrics import (InContextLearningLMAccuracy,
                              InContextLearningMultipleChoiceAccuracy)
from composer.metrics.nlp import LanguageCrossEntropy, LanguagePerplexity
from composer.models import HuggingFaceModel
from omegaconf import DictConfig
from omegaconf import OmegaConf as om
from transformers import AutoTokenizer, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

import examples.llm.src.models.layers.attention as attention
import examples.llm.src.models.layers.gpt_blocks as gpt_blocks
from examples.llm.src.models.mosaic_gpt.configuration_mosaic_gpt import \
    MosaicGPTConfig
from examples.llm.src.models.utils import (MODEL_INIT_REGISTRY,
                                           add_bidirectional_mask_if_missing)


class MosaicGPT(PreTrainedModel):
    config_class = MosaicGPTConfig
    base_model_prefix = 'mosaic_gpt'

    def __init__(self, config: MosaicGPTConfig):
        super().__init__(config)

        self.attn_impl = config.attn_impl
        self.prefix_lm = config.prefix_lm
        self.attn_uses_sequence_id = config.attn_uses_sequence_id
        self.alibi = config.alibi
        self.alibi_bias_max = config.alibi_bias_max

        layernorm_class = LPLayerNorm if config.low_precision_layernorm else nn.LayerNorm

        # CogView (https://arxiv.org/abs/2105.13290) and GLM-130B (https://arxiv.org/abs/2210.02414)
        # both report this helping with stabilizing training
        self.embedding_fraction = config.embedding_fraction

        self.transformer = nn.ModuleDict({
            'wte':
                nn.Embedding(config.vocab_size,
                             config.d_model,
                             device=config.init_device)
        })
        if not self.alibi:
            self.transformer.update({
                'wpe':
                    nn.Embedding(config.max_seq_len,
                                 config.d_model,
                                 device=config.init_device)
            })
        self.transformer.update({'emb_drop': nn.Dropout(config.emb_pdrop)})
        self.transformer.update({
            'blocks':
                nn.ModuleList([
                    gpt_blocks.GPTBlock(device=config.init_device,
                                        **config.to_dict())
                    for _ in range(config.n_layers)
                ])
        })
        self.transformer.update({
            'ln_f': layernorm_class(config.d_model, device=config.init_device)
        })

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
                        f"{logit_scale=} is not recognized as an option; use numeric value or 'inv_sqrt_d_model'."
                    )
            self.logit_scale = logit_scale

        if config.init_device != 'meta':
            print(
                f'You are using {config.init_device=}, but you can also use config.init_device="meta" with Composer + FSDP for fast initialization.'
            )
            self.apply(self.param_init_fn)

        self.is_causal = not self.prefix_lm

        # define attn mask
        self._attn_bias_initialized = False
        self.attn_bias = None
        self.attn_bias_shape = attention.attn_bias_shape(
            self.attn_impl,
            config.n_heads,
            config.max_seq_len,
            self.alibi,
            prefix_lm=self.prefix_lm,
            causal=self.is_causal,
            use_sequence_id=self.attn_uses_sequence_id)

        if config.no_bias:
            for module in self.modules():
                if hasattr(module, 'bias') and isinstance(
                        module.bias, nn.Parameter):
                    if config.verbose:
                        print(f'Removing bias ({module.bias}) from {module}.')
                    module.register_parameter('bias', None)

        if config.verbose and config.verbose > 2:
            print(self)

    @torch.no_grad()
    def _attn_bias(self,
                   device,
                   dtype,
                   attention_mask: Optional[torch.ByteTensor] = None,
                   prefix_mask: Optional[torch.ByteTensor] = None,
                   sequence_id: Optional[torch.LongTensor] = None):
        if not self._attn_bias_initialized:
            if self.attn_bias_shape:
                self.attn_bias = torch.zeros(self.attn_bias_shape,
                                             device=device,
                                             dtype=dtype)
                self.attn_bias = attention.attn_bias(
                    self.attn_impl,
                    self.attn_bias,
                    self.config.n_heads,
                    self.config.max_seq_len,
                    causal=self.is_causal,
                    alibi=self.alibi,
                    alibi_bias_max=self.alibi_bias_max)
            self._attn_bias_initialized = True

        # flash does not support prefix_lm and will incorporate any
        # attention_mask inside the attention module
        if self.attn_impl == 'flash':
            return self.attn_bias, attention_mask

        attn_bias = self.attn_bias

        # If using torch or triton, we incorporate the prefix_mask (if appropriate)
        if self.prefix_lm:
            assert isinstance(attn_bias, torch.Tensor)  # pyright
            assert isinstance(prefix_mask, torch.Tensor)  # pyright
            attn_bias = self._apply_prefix_mask(attn_bias, prefix_mask)

        # If using torch or triton, we incorporate sequence_id (if appropriate)
        if self.attn_uses_sequence_id and sequence_id is not None:
            assert isinstance(attn_bias, torch.Tensor)  # pyright
            attn_bias = self._apply_sequence_id(attn_bias, sequence_id)

        # If using torch or triton, we incorporate attention_mask. This will output
        # None in place of attention_mask since it will not be further needed in the
        # attention modules.
        if attention_mask is not None:
            s_k = attention_mask.shape[-1]
            if attn_bias is None:
                attn_bias = torch.zeros((1, 1, 1, s_k),
                                        device=device,
                                        dtype=dtype)
            else:
                attn_bias = attn_bias[:, :, :, -s_k:]
            if prefix_mask is not None and (attention_mask.shape !=
                                            prefix_mask.shape):
                raise ValueError(
                    f'attention_mask shape={attention_mask.shape} ' +\
                    f'and prefix_mask shape={prefix_mask.shape} are not equal.'
                )
            min_val = torch.finfo(attn_bias.dtype).min
            attn_bias = attn_bias.masked_fill(
                ~attention_mask.view(-1, 1, 1, s_k), min_val)

        return attn_bias, None

    def _apply_prefix_mask(self, attn_bias: torch.Tensor,
                           prefix_mask: torch.Tensor):
        s_k, s_q = attn_bias.shape[-2:]
        if (s_k != self.config.max_seq_len) or (s_q != self.config.max_seq_len):
            raise ValueError(
                'attn_bias does not match the expected shape. ' +\
                f'The last two dimensions should both be {self.config.max_length} ' +\
                f'but are {s_k} and {s_q}.'
            )
        seq_len = prefix_mask.shape[-1]
        if seq_len > self.config.max_seq_len:
            raise ValueError(
                f'prefix_mask sequence length cannot exceed max_seq_len={self.config.max_seq_len}'
            )

        # select seq_len subset of attn mask
        attn_bias = attn_bias[..., :seq_len, :seq_len]

        # Mix the causal max and the bidirectional mask to get the full
        # allowable attention (i.e. full = not accounting for padding yet)
        causal = torch.tril(
            torch.ones((seq_len, seq_len),
                       dtype=torch.bool,
                       device=prefix_mask.device)).view(1, 1, seq_len, seq_len)
        prefix = prefix_mask.view(-1, 1, 1, seq_len)
        cannot_attend = ~torch.logical_or(causal, prefix.bool())

        min_val = torch.finfo(attn_bias.dtype).min
        attn_bias = attn_bias.masked_fill(cannot_attend, min_val)

        return attn_bias

    def _apply_sequence_id(self, attn_bias: torch.Tensor,
                           sequence_id: torch.LongTensor):
        seq_len = sequence_id.shape[-1]
        if seq_len > self.config.max_seq_len:
            raise ValueError(
                f'sequence_id sequence length cannot exceed max_seq_len={self.config.max_seq_len}'
            )

        # select seq_len subset of attn mask
        attn_bias = attn_bias[..., :seq_len, :seq_len]

        # Restrict attention to tokens that share the same value
        # in sequence_id
        cannot_attend = torch.logical_not(
            torch.eq(sequence_id.view(-1, seq_len, 1),
                     sequence_id.view(-1, 1, seq_len))).unsqueeze(1)
        min_val = torch.finfo(attn_bias.dtype).min
        attn_bias = attn_bias.masked_fill(cannot_attend, min_val)

        return attn_bias

    def forward(
            self,
            input_ids: torch.LongTensor,
            past_key_values: Optional[List[Tuple[torch.FloatTensor]]] = None,
            attention_mask: Optional[torch.ByteTensor] = None,
            prefix_mask: Optional[torch.ByteTensor] = None,
            sequence_id: Optional[torch.LongTensor] = None,
            return_dict: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            use_cache: Optional[bool] = None):
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        # These args are passed in by keyword in huggingface's generate function
        # https://github.com/huggingface/transformers/blob/68287689f2f0d8b7063c400230b3766987abf18d/src/transformers/generation/utils.py#L2201-L2206
        # but have not yet been fully implemented in MosaicGPT
        if not return_dict:
            raise NotImplementedError(
                'return_dict False is not implemented yet for MosaicGPT')
        if output_attentions:
            raise NotImplementedError(
                'output_attentions is not implemented yet for MosaicGPT')

        if attention_mask is not None and attention_mask[:, 0].sum(
        ) != attention_mask.shape[0] and self.training:
            raise NotImplementedError(
                'MosaicGPT does not support training with left padding.')

        if self.prefix_lm and prefix_mask is None:
            raise ValueError(
                'prefix_mask is a required argument when MosaicGPT is configured with prefix_lm=True.'
            )

        if self.training:
            if self.attn_uses_sequence_id and sequence_id is None:
                raise ValueError(
                    'sequence_id is a required argument when MosaicGPT is configured with attn_uses_sequence_id=True ' +\
                    'and the model is in train mode.'
                )
            elif (self.attn_uses_sequence_id is False) and (sequence_id
                                                            is not None):
                warnings.warn(
                    'MosaicGPT received non-None input for `sequence_id` but is configured with attn_uses_sequence_id=False. ' +\
                    'This input will be ignored. If you want the model to use `sequence_id`, set attn_uses_sequence_id to True.'
                )

        S = input_ids.size(1)

        assert (
            S <= self.config.max_seq_len
        ), f'Cannot forward input with seq_len={S}, this model only supports seq_len<={self.config.max_seq_len}'

        tok_emb = self.transformer.wte(input_ids)  # type: ignore
        if self.alibi:
            x = tok_emb
        else:
            past_position = 0
            if past_key_values is not None:
                if len(past_key_values) != self.config.n_layers:
                    raise ValueError(
                        f'past_key_values must provide a past_key_value for each attention ' +\
                        f'layer in the network ({len(past_key_values)=}; {self.config.n_layers=}).'
                    )
                # get the key tensor whose spec should be (batch, seq, dim), and
                # collect the `seq`, so that the position embedding is shifted
                past_position = past_key_values[0][0].size(1)

            if S + past_position > self.config.max_seq_len:
                raise ValueError(
                    f'Cannot forward input with past sequence length {past_position} and current sequence length '
                    f'{S + 1}, this model only supports total sequence length <= {self.config.max_seq_len}.'
                )
            pos = torch.arange(past_position,
                               S + past_position,
                               dtype=torch.long,
                               device=input_ids.device).unsqueeze(0)
            if attention_mask is not None:
                # adjust the position indices to account for padding tokens
                pos = torch.clamp(pos - torch.cumsum(
                    (~attention_mask).to(torch.int32), dim=1)[:,
                                                              past_position:],
                                  min=0)

            pos_emb = self.transformer.wpe(pos)  # type: ignore
            x = tok_emb + pos_emb

        if self.embedding_fraction == 1:
            x = self.transformer.emb_drop(x)  # type: ignore
        else:
            # this implementation is proposed on page 7 of the GLM-130B paper https://arxiv.org/abs/2210.02414
            x_shrunk = (x * self.embedding_fraction) + (
                x.detach() * (1 - self.embedding_fraction))
            assert isinstance(self.transformer.emb_drop, nn.Module)  # pyright
            x = self.transformer.emb_drop(x_shrunk)

        attn_bias, attention_mask = self._attn_bias(
            device=x.device,
            dtype=x.dtype,
            attention_mask=attention_mask,
            prefix_mask=prefix_mask,
            sequence_id=sequence_id)

        # initialize the past key values cache if it should be used
        if use_cache and past_key_values is None:
            past_key_values = [() for _ in range(self.config.n_layers)
                              ]  # type: ignore

        all_hidden_states = () if output_hidden_states else None
        for b_idx, block in enumerate(self.transformer.blocks):  # type: ignore
            if output_hidden_states:
                assert all_hidden_states is not None  # pyright
                all_hidden_states = all_hidden_states + (x,)
            past_key_value = past_key_values[
                b_idx] if past_key_values is not None else None
            x, past_key_value = block(x,
                                      past_key_value=past_key_value,
                                      attn_bias=attn_bias,
                                      attention_mask=attention_mask,
                                      is_causal=self.is_causal)
            if past_key_values is not None:
                past_key_values[b_idx] = past_key_value

        x = self.transformer.ln_f(x)  # type: ignore

        # output embedding weight tied to input embedding
        assert isinstance(self.transformer.wte, nn.Module)  # pyright
        assert isinstance(self.transformer.wte.weight, torch.Tensor)  # pyright
        logits = F.linear(x, self.transformer.wte.weight, None)

        if self.logit_scale is not None:
            if self.logit_scale == 0:
                warnings.warn(
                    f'Multiplying logits by {self.logit_scale=}. This will produce uniform (uninformative) outputs.'
                )
            logits *= self.logit_scale

        return CausalLMOutputWithPast(logits=logits,
                                      past_key_values=past_key_values,
                                      hidden_states=all_hidden_states)

    # Param Initialization, needed for device='meta' fast initialization
    def param_init_fn(self, module):
        init_fn_name = self.config.param_init_fn
        if self.config.verbose > 1:
            warnings.warn(f'Using {init_fn_name} initialization.')
        MODEL_INIT_REGISTRY[init_fn_name](module=module,
                                          **self.config.to_dict())

    # FSDP Wrap function
    def fsdp_wrap_fn(self, module):
        return isinstance(module, gpt_blocks.GPTBlock)

    # Activation Checkpointing
    def activation_checkpointing_fn(self, module):
        return isinstance(module, gpt_blocks.GPTBlock)

    def prepare_inputs_for_generation(self,
                                      input_ids,
                                      past_key_values=None,
                                      inputs_embeds=None,
                                      **kwargs):
        if inputs_embeds is not None:
            raise NotImplementedError(
                'inputs_embeds is not implemented for MosaicGPT yet')

        attention_mask = kwargs['attention_mask'].bool()
        if attention_mask[:, -1].sum() != attention_mask.shape[0]:
            raise NotImplementedError(
                'MosaicGPT does not support generation with right padding.')

        if self.attn_uses_sequence_id and self.training:
            sequence_id = torch.zeros_like(input_ids[:1])
        else:
            sequence_id = None

        if past_key_values is not None:
            input_ids = input_ids[:, -1].unsqueeze(-1)

        if self.prefix_lm:
            # Leverage a convenience of sequential generation!
            prefix_mask = torch.ones_like(attention_mask)
            # This requires that we're using the cache
            if kwargs.get('use_cache') == False:
                raise NotImplementedError(
                    'MosaicGPT with prefix_lm=True does not support use_cache=False.'
                )
        else:
            prefix_mask = None

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'prefix_mask': prefix_mask,
            'sequence_id': sequence_id,
            'past_key_values': past_key_values,
            'use_cache': kwargs.get('use_cache', True),
        }

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        """Used by HuggingFace generate when using beam search with kv-caching.

        See https://github.com/huggingface/transformers/blob/3ec7a47664ebe40c40f4b722f6bb1cd30c3821ec/src/transformers/models/gpt2/modeling_gpt2.py#L1122-L1133
        for an example in transformers.
        """
        reordered_past = []
        for layer_past in past_key_values:
            reordered_past += [
                tuple(
                    past_state.index_select(0, beam_idx)
                    for past_state in layer_past)
            ]
        return reordered_past


class ComposerMosaicGPT(HuggingFaceModel):

    def __init__(self, om_model_config: DictConfig,
                 om_tokenizer_config: DictConfig):
        resolved_om_model_config = om.to_container(om_model_config,
                                                   resolve=True)
        hf_config = MosaicGPTConfig.from_dict(resolved_om_model_config)
        model = MosaicGPT(hf_config)

        resolved_om_tokenizer_config = om.to_container(om_tokenizer_config,
                                                       resolve=True)
        tokenizer_kwargs = resolved_om_tokenizer_config.get(  # type: ignore
            'kwargs', {})
        tokenizer_name = resolved_om_tokenizer_config['name']  # type: ignore
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name,
                                                  **tokenizer_kwargs)

        # HuggingFace does not respect the model_max_length kwarg, and overrides it with
        # min(kwargs['model_max_length'], original_config['model_max_length']), so we explicitly
        # set it here
        if 'model_max_length' in tokenizer_kwargs:
            tokenizer.model_max_length = tokenizer_kwargs['model_max_length']

        train_metrics = [
            LanguageCrossEntropy(hf_config.vocab_size),
            LanguagePerplexity(hf_config.vocab_size)
        ]
        eval_metrics = [
            LanguageCrossEntropy(hf_config.vocab_size),
            LanguagePerplexity(hf_config.vocab_size),
            InContextLearningLMAccuracy(),
            InContextLearningMultipleChoiceAccuracy(),
        ]

        super().__init__(
            model=model,
            tokenizer=tokenizer,
            use_logits=True,
            metrics=train_metrics,
            eval_metrics=eval_metrics,
            shift_labels=True,
            allow_embedding_resizing=True,
        )

        self.n_active_params = sum(p.numel() for p in self.parameters())

        loss_fn_config = om_model_config.get('loss_fn', 'fused_crossentropy')
        if loss_fn_config == 'fused_crossentropy':
            try:
                from flash_attn.losses.cross_entropy import CrossEntropyLoss as FusedCrossEntropyLoss  # type: ignore # isort: skip
                warnings.warn('Using Fused Cross Entropy Loss.')
                self.loss_fn = FusedCrossEntropyLoss(ignore_index=-100)
            except:
                raise ValueError(
                    'Fused Cross Entropy is not installed. Either (1) have a CUDA-compatible GPU and `pip install .[llm]`, or (2) set your config model.loss_fn=torch_crossentropy.'
                )
        elif loss_fn_config == 'torch_crossentropy':
            self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        else:
            raise ValueError(
                f'Specified loss_fn={self.loss_fn} not recognized. `loss_fn` must be one of [`fused_crossentropy`, `torch_crossentropy`].'
            )

    def get_targets(self, batch):
        targets = torch.roll(batch['labels'], shifts=-1)
        targets[:, -1] = -100
        return targets

    def forward(self, batch):
        if self.model.prefix_lm:
            add_bidirectional_mask_if_missing(batch)
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask'].bool(
        ) if 'attention_mask' in batch else None
        sequence_id = batch.get('sequence_id', None)
        prefix_mask = batch['bidirectional_mask'].bool(
        ) if 'bidirectional_mask' in batch else None
        # Note: prefix_mask is only used if model.prefix_lm is True
        return self.model(input_ids=input_ids,
                          attention_mask=attention_mask,
                          prefix_mask=prefix_mask,
                          sequence_id=sequence_id)

    def loss(self, outputs, batch):
        targets = self.get_targets(batch)
        return self.loss_fn(outputs.logits.view(-1, outputs.logits.size(-1)),
                            targets.view(-1))

    def flops_per_batch(self, batch):
        # Note: this computation does not take into account padding, and assumes
        # that the dataset has been constructed without padding. Additionally, we
        # assume the backward pass is approximately 2x the forward pass

        bs, msl = batch['input_ids'].shape[0:2]
        params_flops_per_token = 2 * self.n_active_params
        params_flops_per_seq = params_flops_per_token * msl
        attn_flops_per_seq = self.model.config.n_layers * 2 * 2 * (
            self.model.config.d_model * (msl**2))

        return (params_flops_per_seq + attn_flops_per_seq) * 3 * bs
