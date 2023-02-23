# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

"""A simple, flexible implementation of a GPT model.

Inspired by https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
"""

import math
from functools import partial
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from composer.metrics import METRIC_DEFAULT_CTORS, InContextLearningMetric
from composer.metrics.nlp import LanguageCrossEntropy, Perplexity
from composer.models.base import ComposerModel
from omegaconf import DictConfig

import examples.llm.src.models.layers.attention as attention
import examples.llm.src.models.layers.gpt_blocks as gpt_blocks


class MosaicGPT(nn.Module):

    def __init__(self, cfg: DictConfig):
        super().__init__()
        assert cfg.name == 'mosaic_gpt', f'Tried to build MosaicGPT model with cfg.name={cfg.name}'
        self.cfg = cfg
        if cfg.attn_impl == 'torch':
            self.causal_attn_cls = attention.TorchCausalAttention
        elif cfg.attn_impl == 'flash':
            self.causal_attn_cls = attention.FlashCausalAttention
        elif cfg.attn_impl == 'triton':
            self.causal_attn_cls = attention.TritonFlashCausalAttention
        else:
            raise ValueError(f'Unknown attn_impl={cfg.attn_impl}')

        if cfg.get('attn_qk_ln') and cfg.attn_impl != 'flash':
            raise NotImplementedError(
                'LayerNorm over queries and keys in attention is only implemented with flash attention.'
            )

        self.alibi = cfg.get('alibi', False)
        self.alibi_bias_max = cfg.get('alibi_bias_max',
                                      8 if self.alibi else None)
        # CogView (https://arxiv.org/abs/2105.13290) and GLM-130B (https://arxiv.org/abs/2210.02414)
        # both report this helping with stabilizing training
        self.embedding_fraction = cfg.get('embedding_fraction', 1)
        assert 0 < self.embedding_fraction <= 1, 'model.embedding_fraction must be between 0 (exclusive) and 1 (inclusive)!'

        self.transformer = nn.ModuleDict({
            'wte':
                nn.Embedding(cfg.vocab_size,
                             cfg.d_model,
                             device=cfg.init_device)
        })
        if not self.alibi:
            self.transformer.update({
                'wpe':
                    nn.Embedding(cfg.max_seq_len,
                                 cfg.d_model,
                                 device=cfg.init_device)
            })
        self.transformer.update({'emb_drop': nn.Dropout(cfg.emb_pdrop)})
        self.transformer.update({
            'blocks':
                nn.ModuleList([
                    gpt_blocks.GPTBlock(cfg,
                                        causal_attn_cls=self.causal_attn_cls,
                                        device=cfg.init_device)
                    for _ in range(cfg.n_layers)
                ])
        })
        self.transformer.update(
            {'ln_f': nn.LayerNorm(cfg.d_model, device=cfg.init_device)})

        if cfg.init_device != 'meta':
            print(
                f'You are using {cfg.init_device=}, but you can also use cfg.init_device="meta" with Composer + FSDP for fast initialization.'
            )
            self.apply(self.param_init_fn)

        # define attn mask
        self._attn_mask_initialized = False
        mask_shape = self.causal_attn_cls.mask_shape(cfg.n_heads,
                                                     cfg.max_seq_len,
                                                     self.alibi)
        if mask_shape is not None:
            self.register_buffer(
                'attn_mask', torch.empty(mask_shape, device=cfg.init_device))
        else:
            self.attn_mask = None

    def _attn_mask(self,
                   batch_size=None,
                   seq_len=None,
                   key_padding_mask=None,
                   dtype=None):
        if not self._attn_mask_initialized:
            self.causal_attn_cls.attn_mask_(self.attn_mask,
                                            self.cfg.n_heads,
                                            self.cfg.max_seq_len,
                                            alibi=self.alibi,
                                            alibi_bias_max=self.alibi_bias_max)
            self._attn_mask_initialized = True

        return self.causal_attn_cls.generate_attn_mask(
            self.attn_mask,
            batch_size,
            self.cfg.n_heads,
            seq_len,
            key_padding_mask=key_padding_mask,
            alibi=self.alibi,
            dtype=dtype)

    def forward(self,
                input_ids: torch.LongTensor,
                key_padding_mask: Optional[torch.ByteTensor] = None):
        B, S = input_ids.size()
        assert (
            S <= self.cfg.max_seq_len
        ), f'Cannot forward input with seq_len={S}, this model only supports seq_len<={self.cfg.max_seq_len}'

        tok_emb = self.transformer.wte(input_ids)  # type: ignore
        if self.alibi:
            x = tok_emb
        else:
            pos = torch.arange(0, S, dtype=torch.long,
                               device=input_ids.device).unsqueeze(0)
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

        attn_mask = self._attn_mask(batch_size=B,
                                    seq_len=S,
                                    key_padding_mask=key_padding_mask,
                                    dtype=x.dtype)
        if self.cfg.attn_impl == 'flash' and key_padding_mask is None:
            # HazyResearch FlashMHA appears to use more memory when `key_padding_mask=None`
            # in certain settings like MosaicGPT-7B. So we always provide a tensor.
            # See https://github.com/mosaicml/examples/pull/163 for more details.
            mod_key_padding_mask = torch.ones_like(input_ids, dtype=torch.bool)
        elif self.cfg.attn_impl == 'triton':
            mod_key_padding_mask = None
        else:
            mod_key_padding_mask = key_padding_mask
        for block in self.transformer.blocks:  # type: ignore
            x = block(x, mod_key_padding_mask, attn_mask)
        x = self.transformer.ln_f(x)  # type: ignore
        # output embedding weight tied to input embedding
        assert isinstance(self.transformer.wte, nn.Module)  # pyright
        assert isinstance(self.transformer.wte.weight, torch.Tensor)  # pyright
        logits = F.linear(x, self.transformer.wte.weight, None)
        return logits

    # Param Initialization, needed for device='meta' fast initialization
    def param_init_fn(self, module):
        init_fn = partial(torch.nn.init.normal_,
                          mean=0.0,
                          std=self.cfg.init_std)
        # Linear
        if isinstance(module, nn.Linear):
            init_fn(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

            if getattr(module, '_is_residual', False):
                module.weight.data.normal_(
                    mean=0.0,
                    std=(self.cfg.init_std / math.sqrt(2 * self.cfg.n_layers)))

        # Embedding
        if isinstance(module, nn.Embedding):
            init_fn(module.weight)

        # LayerNorm
        if isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

        # torch's MultiheadAttention
        if isinstance(module, nn.MultiheadAttention):
            if module._qkv_same_embed_dim:
                assert module.in_proj_weight is not None
                assert module.q_proj_weight is None and module.k_proj_weight is None and module.v_proj_weight is None
                init_fn(module.in_proj_weight)
            else:
                assert module.q_proj_weight is not None and module.k_proj_weight is not None and module.v_proj_weight is not None
                assert module.in_proj_weight is None
                init_fn(module.q_proj_weight)
                init_fn(module.k_proj_weight)
                init_fn(module.v_proj_weight)

            # bias
            if module.in_proj_bias is not None:
                torch.nn.init.zeros_(module.in_proj_bias)
            if module.bias_k is not None:
                torch.nn.init.zeros_(module.bias_k)
            if module.bias_v is not None:
                torch.nn.init.zeros_(module.bias_v)

            # out proj
            if module.out_proj._is_residual:
                module.out_proj.weight.data.normal_(
                    mean=0.0,
                    std=(self.cfg.init_std / math.sqrt(2 * self.cfg.n_layers)))
            else:
                init_fn(module.out_proj.weight)
            if module.out_proj.bias is not None:
                torch.nn.init.zeros_(module.out_proj.bias)

    # FSDP Wrap function
    def fsdp_wrap_fn(self, module):
        return isinstance(module, gpt_blocks.GPTBlock)

    # Activation Checkpointing
    def activation_checkpointing_fn(self, module):
        return isinstance(module, gpt_blocks.GPTBlock)


class ComposerMosaicGPT(ComposerModel):

    def __init__(self, cfg):
        super().__init__()
        self.model = MosaicGPT(cfg)
        self.__num_fwd_flops = None
        self.train_metrics = {
            'LanguageCrossEntropy': LanguageCrossEntropy(cfg.vocab_size),
            'Perplexity': Perplexity(),
        }
        self.eval_metrics = {
            'LanguageCrossEntropy': LanguageCrossEntropy(cfg.vocab_size),
            'Perplexity': Perplexity(),
        }

    def get_targets(self, batch):
        targets = torch.roll(batch['labels'], shifts=-1)
        targets[:, -1] = -100
        return targets

    def forward(self, batch):
        input_ids = batch['input_ids']
        key_padding_mask = batch['attention_mask'].bool(
        ) if 'attention_mask' in batch else None
        return self.model(input_ids=input_ids,
                          key_padding_mask=key_padding_mask)

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

    @property
    def num_fwd_flops(self):
        if self.__num_fwd_flops:
            return self.__num_fwd_flops
        n_params = sum(p.numel() for p in self.parameters())
        # the number of paramters is approximately the number of multiply-accumulates (MAC) in the network
        # each MAC has 2 FLOPs - we multiply by 2 ie 2 * n_param
        # this gets us FLOPs / token
        params_flops_per_token = 2 * n_params
        params_flops_per_seq = params_flops_per_token * self.model.cfg.max_seq_len
        # there are 2 FLOPS per mac; there is A=Q*K^T and out=A*V ops (ie mult by 2)
        attn_flops_per_seq = self.model.cfg.n_layers * 2 * 2 * (
            self.model.cfg.d_model * (self.model.cfg.max_seq_len**2))
        self.__num_fwd_flops = params_flops_per_seq + attn_flops_per_seq
        return self.__num_fwd_flops
