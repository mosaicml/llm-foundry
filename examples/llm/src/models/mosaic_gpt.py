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
from composer.metrics import METRIC_DEFAULT_CTORS, InContextLearningMetric
from composer.metrics.nlp import LanguageCrossEntropy, Perplexity
from composer.models.base import ComposerModel
from omegaconf import DictConfig
from omegaconf import OmegaConf as om
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

import examples.llm.src.models.layers.attention as attention
import examples.llm.src.models.layers.gpt_blocks as gpt_blocks
from examples.llm.src.models.configuration_mosaic_gpt import MosaicGPTConfig
from examples.llm.src.models.param_init_fns import MODEL_INIT_REGISTRY


class MosaicGPT(PreTrainedModel):
    config_class = MosaicGPTConfig
    base_model_prefix = 'mosaic_gpt'

    def __init__(self, config: MosaicGPTConfig):
        super().__init__(config)

        self.attn_impl = config.attn_impl
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

        self.is_causal = True

        # define attn mask
        self._attn_bias_initialized = False
        self.attn_bias = None
        self.attn_bias_shape = attention.attn_bias_shape(self.attn_impl,
                                                         config.n_heads,
                                                         config.max_seq_len,
                                                         self.alibi,
                                                         causal=self.is_causal)

        if config.no_bias:
            for module in self.modules():
                if hasattr(module, 'bias') and isinstance(
                        module.bias, nn.Parameter):
                    if config.verbose:
                        print(f'Removing bias ({module.bias}) from {module}.')
                    module.register_parameter('bias', None)

        if config.verbose and config.verbose > 2:
            print(self)

    def _attn_bias(self, device, dtype):
        if not self._attn_bias_initialized:
            if self.attn_bias_shape:
                self.attn_bias = torch.zeros(self.attn_bias_shape,
                                             device=device,
                                             dtype=dtype)
                attention.attn_bias(self.attn_impl,
                                    self.attn_bias,
                                    self.config.n_heads,
                                    self.config.max_seq_len,
                                    causal=self.is_causal,
                                    alibi=self.alibi,
                                    alibi_bias_max=self.alibi_bias_max)
            self._attn_bias_initialized = True

        return self.attn_bias

    def forward(
            self,
            input_ids: torch.LongTensor,
            past_key_values: Optional[List[Tuple[torch.FloatTensor]]] = None,
            attention_mask: Optional[torch.ByteTensor] = None,
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
        if output_hidden_states:
            raise NotImplementedError(
                'output_hidden_states is not implemented yet for MosaicGPT')

        if attention_mask is not None and attention_mask[:, 0].sum(
        ) != attention_mask.shape[0] and self.training:
            raise NotImplementedError(
                'MosaicGPT does not support training with left padding.')

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
                    (attention_mask == 0)[:, past_position:], dim=1),
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

        attn_bias = self._attn_bias(device=x.device, dtype=x.dtype)

        # initialize the past key values cache if it should be used
        if use_cache and past_key_values is None:
            past_key_values = [() for _ in range(self.config.n_layers)
                              ]  # type: ignore

        for b_idx, block in enumerate(self.transformer.blocks):  # type: ignore
            past_key_value = past_key_values[
                b_idx] if past_key_values is not None else None
            x, past_key_value = block(x,
                                      past_key_value=past_key_value,
                                      attn_bias=attn_bias,
                                      key_padding_mask=attention_mask,
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
                                      past_key_values=past_key_values)

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

        if past_key_values is not None:
            input_ids = input_ids[:, -1].unsqueeze(-1)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'past_key_values': past_key_values,
            'use_cache': kwargs.get('use_cache'),
        }


class ComposerMosaicGPT(ComposerModel):

    def __init__(self, om_model_config: DictConfig):
        super().__init__()

        resolved_om_config = om.to_container(om_model_config, resolve=True)
        self.hf_config = MosaicGPTConfig.from_dict(resolved_om_config)
        self.model = MosaicGPT(self.hf_config)
        self.num_fwd_flops = self._compute_num_fwd_flops()
        self.train_metrics = {
            'LanguageCrossEntropy':
                LanguageCrossEntropy(self.hf_config.vocab_size),
            'Perplexity':
                Perplexity(),
        }
        self.eval_metrics = {
            'LanguageCrossEntropy':
                LanguageCrossEntropy(self.hf_config.vocab_size),
            'Perplexity':
                Perplexity(),
        }

    def get_targets(self, batch):
        targets = torch.roll(batch['labels'], shifts=-1)
        targets[:, -1] = -100
        return targets

    def forward(self, batch):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask'].bool(
        ) if 'attention_mask' in batch else None
        return self.model(input_ids=input_ids,
                          attention_mask=attention_mask).logits

    def eval_forward(self, batch, outputs=None):
        return outputs if outputs is not None else self.forward(batch)

    def loss(self, outputs, batch):
        targets = self.get_targets(batch)
        return F.cross_entropy(outputs.view(-1, outputs.size(-1)),
                               targets.view(-1),
                               ignore_index=-100)

    def get_metrics(self, is_train=False):
        return self.train_metrics if is_train else self.eval_metrics

    def update_metric(self, batch, outputs, metric) -> None:
        if isinstance(metric, InContextLearningMetric):
            if batch.get('mode', None) == 'icl_task':
                # only apply ICL metrics to specially constructed
                # icl_task batches
                targets = self.get_targets(batch)
                metric.update(batch, outputs, targets)
        else:
            outputs = outputs.view(-1, outputs.size(-1))
            targets = self.get_targets(batch).view(-1)
            metric.update(outputs, targets)

    def add_eval_metrics(self, evaluator):
        evaluator_metrics = {
            m: METRIC_DEFAULT_CTORS[m]() for m in evaluator.metric_names
        }
        if self.eval_metrics is not None:
            self.eval_metrics.update(evaluator_metrics)
        else:
            self.eval_metrics = evaluator_metrics

    def _compute_num_fwd_flops(self):
        n_params = sum(p.numel() for p in self.parameters())
        # the number of paramters is approximately the number of multiply-accumulates (MAC) in the network
        # each MAC has 2 FLOPs - we multiply by 2 ie 2 * n_param
        # this gets us FLOPs / token
        params_flops_per_token = 2 * n_params
        params_flops_per_seq = params_flops_per_token * self.model.config.max_seq_len
        # there are 2 FLOPS per mac; there is A=Q*K^T and out=A*V ops (ie mult by 2)
        attn_flops_per_seq = self.model.config.n_layers * 2 * 2 * (
            self.model.config.d_model * (self.model.config.max_seq_len**2))
        return params_flops_per_seq + attn_flops_per_seq

    def flops_per_batch(self, batch):
        # Note: this computation does not take into account padding, and assumes
        # that the dataset has been constructed without padding. Additionally, we
        # assume the backward pass is approximately 2x the forward pass
        return self.num_fwd_flops * 3 * batch['input_ids'].shape[0]
