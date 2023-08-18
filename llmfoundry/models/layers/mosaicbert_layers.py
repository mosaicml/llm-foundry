# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018-2021, NVIDIA CORPORATION.  All rights reserved.
# Copyright (c) 2022, Tri Dao.

"""Implements MosaicBERT, with an eye towards the Hugging Face API.

MosaicBERT improves performance over Hugging Face BERT through the following:

1. ALiBi. This architectural change removes positional embeddings and instead encodes positional
information through attention biases based on query-key position distance. It improves the effectiveness
of training with shorter sequence lengths by enabling extrapolation to longer sequences.

2. Gated Linear Units (GLU). This architectural change replaces the FFN component of the BERT layer
to improve overall expressiveness, providing better convergence properties.

3. Flash Attention. The MosaicBERT's self-attention layer makes use of Flash Attention, which dramatically
improves the speed of self-attention. Our implementation utilizes a bleeding edge implementation that
supports attention biases, which allows us to use Flash Attention with ALiBi.

4. Unpadding. Padding is often used to simplify batching across sequences of different lengths. Standard BERT
implementations waste computation on padded tokens. MosaicBERT internally unpads to reduce unnecessary computation
and improve speed. It does this without changing how the user interfaces with the model, thereby
preserving the simple API of standard implementations.


Currently, MosaicBERT is available for masked language modeling :class:`BertForMaskedLM` and sequence
classification :class:`BertForSequenceClassification`. We aim to expand this catalogue in future releases.

See :file:`./mosaic_bert.py` for utilities to simplify working with MosaicBERT in Composer, and for example usage
of the core MosaicBERT classes.
"""

import copy
import logging
import math
import warnings
from typing import List, Optional, Union

import torch
import torch.nn as nn
from einops import rearrange
from transformers.activations import ACT2FN

from llmfoundry.models.utils.bert_padding import (index_first_axis, pad_input,
                                                  unpad_input, unpad_input_only)

try:
    from llmfoundry.models.layers.flash_attn_triton import \
        flash_attn_qkvpacked_func
except ImportError as e:
    flash_attn_qkvpacked_func = None

logger = logging.getLogger(__name__)


class BertEmbeddings(nn.Module):
    """Construct the embeddings for words, ignoring position.

    There are no positional embeddings since we use ALiBi and token_type
    embeddings.

    This module is modeled after the Hugging Face BERT's
    :class:`~transformers.model.bert.modeling_bert.BertEmbeddings`, but is
    modified as part of MosaicBERT's ALiBi implementation. The key change is
    that position embeddings are removed. Position information instead comes
    from attention biases that scale linearly with the position distance
    between query and key tokens.

    This module ignores the `position_ids` input to the `forward` method.
    """

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size,
                                            config.hidden_size,
                                            padding_idx=config.pad_token_id)
        # ALiBi doesn't use position embeddings
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size,
                                                  config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model
        # variable name and be able to load any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size,
                                      eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.register_buffer('token_type_ids',
                             torch.zeros(config.max_position_embeddings,
                                         dtype=torch.long),
                             persistent=False)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values_length: int = 0,
    ) -> torch.Tensor:
        if (input_ids is not None) == (inputs_embeds is not None):
            raise ValueError('Must specify either input_ids or input_embeds!')
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            assert inputs_embeds is not None  # just for type checking
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            # great! ALiBi
            pass

        # Setting the token_type_ids to the registered buffer in constructor
        # where it is all zeros, which usually occurs when it's auto-generated;
        # registered buffer helps users when tracing the model without passing
        # token_type_ids, solves issue #5664
        if token_type_ids is None:
            if hasattr(self, 'token_type_ids'):
                assert isinstance(self.token_type_ids, torch.LongTensor)
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(
                    input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded  # type: ignore
            else:
                token_type_ids = torch.zeros(input_shape,  # type: ignore
                                             dtype=torch.long,
                                             device=self.word_embeddings.device) # type: ignore  # yapf: disable

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        # no position embeddings! ALiBi
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertUnpadSelfAttention(nn.Module):
    """Performs multi-headed self attention on a batch of unpadded sequences.

    If Triton is installed, this module uses Flash Attention to greatly improve throughput.
    The Flash Attention implementation used in MosaicBERT supports arbitrary attention biases (which
    we use to implement ALiBi), but does not support attention dropout. If either Triton is not installed
    or `config.attention_probs_dropout_prob > 0`, the implementation will default to a
    math-equivalent pytorch version, which is much slower.

    See `forward` method for additional detail.
    """

    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(
                config, 'embedding_size'):
            raise ValueError(
                f'The hidden size ({config.hidden_size}) is not a multiple of the number of attention '
                + f'heads ({config.num_attention_heads})')

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size /
                                       config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.p_dropout = config.attention_probs_dropout_prob
        self.Wqkv = nn.Linear(self.all_head_size, 3 * config.hidden_size)

        # Warn if defaulting to pytorch because of import issues
        if flash_attn_qkvpacked_func is None:
            warnings.warn(
                'Unable to import Triton; defaulting MosaicBERT attention implementation to pytorch (this will reduce throughput when using this model).'
            )

    def forward(self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor,
                max_seqlen_in_batch: int, indices: torch.Tensor,
                attn_mask: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        """Perform self-attention.

        If dropout is zero, then we can use the Triton kernel, so we do that. However, if not, we send through a standard PyTorch
        implementation of self-attention.

        The arguments are unpadded, and our implementations of attention require padded arguments,
        so we first call `pad_input`. Once we compute attention, we re-unpad our outputs for the other layers.
        The pad/unpad operations add overhead, but not sending pad tokens through ffs saves compute.
        It is possible to write an unpadded implementation of attention (in Triton and PyTorch), which we will eventually do.

        Args:
            hidden_states: (total_nnz, dim)
            cu_seqlens: (batch + 1,)
            max_seqlen_in_batch: int
            indices: (total_nnz,)
            attn_mask: (batch, max_seqlen_in_batch)
            bias: (batch, heads, max_seqlen_in_batch, max_seqlen_in_batch)

        Returns:
            attention: (total_nnz, dim)
        """
        qkv = self.Wqkv(hidden_states)
        qkv = pad_input(qkv, indices, cu_seqlens.shape[0] - 1,
                        max_seqlen_in_batch)  # batch, max_seqlen_in_batch, thd
        qkv = rearrange(qkv,
                        'b s (t h d) -> b s t h d',
                        t=3,
                        h=self.num_attention_heads)
        if self.p_dropout or flash_attn_qkvpacked_func is None:
            # if we have nonzero attention dropout (e.g. during fine-tuning) or no Triton, compute attention in PyTorch
            q = qkv[:, :, 0, :, :].permute(0, 2, 1, 3)  # b h s d
            k = qkv[:, :, 1, :, :].permute(0, 2, 3, 1)  # b h d s
            v = qkv[:, :, 2, :, :].permute(0, 2, 1, 3)  # b h s d
            attention_scores = torch.matmul(q, k) / math.sqrt(
                self.attention_head_size)
            attention_scores = attention_scores + bias
            attention_probs = nn.functional.softmax(attention_scores, dim=-1)
            attention_probs = self.dropout(attention_probs)
            attention = torch.matmul(attention_probs, v).permute(0, 2, 1,
                                                                 3)  # b s h d
        else:
            # Triton implementation only supports 0 attention dropout
            convert_dtype = qkv.dtype not in [torch.float16, torch.bfloat16]
            if convert_dtype:
                # Triton implementation only supports fp16 and bf16
                orig_dtype = qkv.dtype
                qkv = qkv.to(torch.float16)
                bias_dtype = bias.dtype
                bias = bias.to(torch.float16)
                attention = flash_attn_qkvpacked_func(qkv, bias)
                attention = attention.to(orig_dtype)
                bias = bias.to(bias_dtype)
            else:
                attention = flash_attn_qkvpacked_func(qkv, bias)

        # attn_mask is 1 for attend and 0 for don't
        attention = unpad_input_only(attention, torch.squeeze(attn_mask) == 1)
        return rearrange(attention, 'nnz h d -> nnz (h d)')


# Copy of transformer's library BertSelfOutput that will not be caught by surgery methods looking for HF BERT modules.
class BertSelfOutput(nn.Module):
    """Computes the output of the attention layer.

    This module is modeled after the Hugging Face BERT's
    :class:`~transformers.model.bert.modeling_bert.BertSelfOutput`.
    The implementation is identical. Rather than use the original module
    directly, we re-implement it here so that MosaicBERT's modules will not
    be affected by any Composer surgery algorithm that modifies Hugging Face
    BERT modules.
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size,
                                      eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor,
                input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertUnpadAttention(nn.Module):
    """Chains attention, Dropout, and LayerNorm for MosaicBERT."""

    def __init__(self, config):
        super().__init__()
        self.self = BertUnpadSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(
        self,
        input_tensor: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_s: int,
        subset_idx: Optional[torch.Tensor] = None,
        indices: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for scaled self-attention without padding.

        Arguments:
            input_tensor: (total_nnz, dim)
            cu_seqlens: (batch + 1,)
            max_s: int
            subset_idx: () set of indices whose values we care about at the end of the layer
                        (e.g., the masked tokens, if this is the final layer).
            indices: None or (total_nnz,)
            attn_mask: None or (batch, max_seqlen_in_batch)
            bias: None or (batch, heads, max_seqlen_in_batch, max_seqlen_in_batch)
        """
        self_output = self.self(input_tensor, cu_seqlens, max_s, indices,
                                attn_mask, bias)
        if subset_idx is not None:
            return self.output(index_first_axis(self_output, subset_idx),
                               index_first_axis(input_tensor, subset_idx))
        else:
            return self.output(self_output, input_tensor)


class BertGatedLinearUnitMLP(nn.Module):
    """Applies the FFN at the end of each MosaicBERT layer.

    Compared to the default BERT architecture, this block replaces :class:`~transformers.model.bert.modeling_bert.BertIntermediate`
    and :class:`~transformers.model.bert.modeling_bert.SelfOutput` with a single module that has similar functionality, but
    introduces Gated Linear Units.

    Note: MosaicBERT adds parameters in order to implement Gated Linear Units. To keep parameter count consistent with that of a
    standard Hugging Face BERT, scale down `config.intermediate_size` by 2/3. For example, a MosaicBERT constructed with
    `config.intermediate_size=2048` will have the same parameter footprint as its Hugging Face BERT counterpart constructed
    with the `config.intermediate_size=3072`.
    However, in most cases it will not be necessary to adjust `config.intermediate_size` since, despite the increased
    parameter size, MosaicBERT typically offers a net higher throughput than a Hugging Face BERT built from the same `config`.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gated_layers = nn.Linear(config.hidden_size,
                                      config.intermediate_size * 2,
                                      bias=False)
        self.act = nn.GELU(approximate='none')
        self.wo = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layernorm = nn.LayerNorm(config.hidden_size,
                                      eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute new hidden states from current hidden states.

        Args:
            hidden_states (torch.Tensor): The (unpadded) hidden states from
                the attention layer [nnz, dim].
        """
        residual_connection = hidden_states
        # compute the activation
        hidden_states = self.gated_layers(hidden_states)
        gated = hidden_states[:, :self.config.intermediate_size]
        non_gated = hidden_states[:, self.config.intermediate_size:]
        hidden_states = self.act(gated) * non_gated
        hidden_states = self.dropout(hidden_states)
        # multiply by the second matrix
        hidden_states = self.wo(hidden_states)
        # add the residual connection and post-LN
        hidden_states = self.layernorm(hidden_states + residual_connection)
        return hidden_states


class BertLayer(nn.Module):
    """Composes the MosaicBERT attention and FFN blocks into a single layer."""

    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertUnpadAttention(config)
        self.mlp = BertGatedLinearUnitMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        seqlen: int,
        subset_idx: Optional[torch.Tensor] = None,
        indices: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for a BERT layer, including both attention and MLP.

        Args:
            hidden_states: (total_nnz, dim)
            cu_seqlens: (batch + 1,)
            seqlen: int
            subset_idx: () set of indices whose values we care about at the end of the layer
                        (e.g., the masked tokens, if this is the final layer).
            indices: None or (total_nnz,)
            attn_mask: None or (batch, max_seqlen_in_batch)
            bias: None or (batch, heads, max_seqlen_in_batch, max_seqlen_in_batch)
        """
        attention_output = self.attention(hidden_states, cu_seqlens, seqlen,
                                          subset_idx, indices, attn_mask, bias)
        layer_output = self.mlp(attention_output)
        return layer_output


class BertEncoder(nn.Module):
    """A stack of BERT layers providing the backbone of MosaicBERT.

    This module is modeled after the Hugging Face BERT's :class:`~transformers.model.bert.modeling_bert.BertEncoder`,
    but with substantial modifications to implement unpadding and ALiBi.

    Compared to the analogous Hugging Face BERT module, this module handles unpadding to reduce unnecessary computation
    at padded tokens, and pre-computes attention biases to implement ALiBi.
    """

    def __init__(self, config):
        super().__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList(
            [copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])

        self.num_attention_heads = config.num_attention_heads

        # The alibi mask will be dynamically expanded if it is too small for
        # the input the model receives. But it generally helps to initialize it
        # to a reasonably large size to help pre-allocate CUDA memory.
        # The default `alibi_starting_size` is 512.
        self._current_alibi_size = int(config.alibi_starting_size)
        self.alibi = torch.zeros(
            (1, self.num_attention_heads, self._current_alibi_size,
             self._current_alibi_size))
        self.rebuild_alibi_tensor(size=config.alibi_starting_size)

    def rebuild_alibi_tensor(self,
                             size: int,
                             device: Optional[Union[torch.device, str]] = None):
        # Alibi
        # Following https://github.com/ofirpress/attention_with_linear_biases/issues/5 (Implementation 1)
        # In the causal case, you can exploit the fact that softmax is invariant to a uniform translation
        # of the logits, which makes the math work out *after* applying causal masking. If no causal masking
        # will be applied, it is necessary to construct the diagonal mask.
        n_heads = self.num_attention_heads

        def _get_alibi_head_slopes(n_heads: int) -> List[float]:

            def get_slopes_power_of_2(n_heads: int) -> List[float]:
                start = (2**(-2**-(math.log2(n_heads) - 3)))
                ratio = start
                return [start * ratio**i for i in range(n_heads)]

            # In the paper, they only train models that have 2^a heads for some a. This function
            # has some good properties that only occur when the input is a power of 2. To
            # maintain that even when the number of heads is not a power of 2, we use a
            # workaround.
            if math.log2(n_heads).is_integer():
                return get_slopes_power_of_2(n_heads)

            closest_power_of_2 = 2**math.floor(math.log2(n_heads))
            slopes_a = get_slopes_power_of_2(closest_power_of_2)
            slopes_b = _get_alibi_head_slopes(2 * closest_power_of_2)
            slopes_b = slopes_b[0::2][:n_heads - closest_power_of_2]
            return slopes_a + slopes_b

        context_position = torch.arange(size, device=device)[:, None]
        memory_position = torch.arange(size, device=device)[None, :]
        relative_position = torch.abs(memory_position - context_position)
        # [n_heads, max_token_length, max_token_length]
        relative_position = relative_position.unsqueeze(0).expand(
            n_heads, -1, -1)
        slopes = torch.Tensor(_get_alibi_head_slopes(n_heads)).to(device)
        alibi = slopes.unsqueeze(1).unsqueeze(1) * -relative_position
        # [1, n_heads, max_token_length, max_token_length]
        alibi = alibi.unsqueeze(0)
        assert alibi.shape == torch.Size([1, n_heads, size, size])

        self._current_alibi_size = size
        self.alibi = alibi

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        output_all_encoded_layers: Optional[bool] = True,
        subset_mask: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        attention_mask_bool = attention_mask.bool()
        batch, seqlen = hidden_states.shape[:2]
        # Unpad inputs and mask. It will remove tokens that are padded.
        # Assume ntokens is total number of tokens (padded and non-padded)
        # and ntokens_unpad is total number of non-padded tokens.
        # Then unpadding performs the following compression of the inputs:
        # hidden_states[ntokens,hidden] -> hidden_states[ntokens_unpad,hidden]
        hidden_states, indices, cu_seqlens, _ = unpad_input(
            hidden_states, attention_mask_bool)

        # Add alibi matrix to extended_attention_mask
        if self._current_alibi_size < seqlen:
            # Rebuild the alibi tensor when needed
            warnings.warn(
                f'Increasing alibi size from {self._current_alibi_size} to {seqlen}'
            )
            self.rebuild_alibi_tensor(size=seqlen, device=hidden_states.device)
        elif self.alibi.device != hidden_states.device:
            # Device catch-up
            self.alibi = self.alibi.to(hidden_states.device)
        alibi_bias = self.alibi[:, :, :seqlen, :seqlen]
        attn_bias = extended_attention_mask[:, :, :seqlen, :seqlen]
        alibi_attn_mask = attn_bias + alibi_bias

        all_encoder_layers = []
        if subset_mask is None:
            for layer_module in self.layer:
                hidden_states = layer_module(hidden_states,
                                             cu_seqlens,
                                             seqlen,
                                             None,
                                             indices,
                                             attn_mask=attention_mask,
                                             bias=alibi_attn_mask)
                if output_all_encoded_layers:
                    all_encoder_layers.append(hidden_states)
            # Pad inputs and mask. It will insert back zero-padded tokens.
            # Assume ntokens is total number of tokens (padded and non-padded)
            # and ntokens_unpad is total number of non-padded tokens.
            # Then padding performs the following de-compression:
            #     hidden_states[ntokens_unpad,hidden] -> hidden_states[ntokens,hidden]
            hidden_states = pad_input(hidden_states, indices, batch, seqlen)
        else:
            for i in range(len(self.layer) - 1):
                layer_module = self.layer[i]
                hidden_states = layer_module(hidden_states,
                                             cu_seqlens,
                                             seqlen,
                                             None,
                                             indices,
                                             attn_mask=attention_mask,
                                             bias=alibi_attn_mask)
                if output_all_encoded_layers:
                    all_encoder_layers.append(hidden_states)
            subset_idx = torch.nonzero(subset_mask[attention_mask_bool],
                                       as_tuple=False).flatten()
            hidden_states = self.layer[-1](hidden_states,
                                           cu_seqlens,
                                           seqlen,
                                           subset_idx=subset_idx,
                                           indices=indices,
                                           attn_mask=attention_mask,
                                           bias=alibi_attn_mask)

        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class BertPooler(nn.Module):

    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self,
                hidden_states: torch.Tensor,
                pool: Optional[bool] = True) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0] if pool else hidden_states
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPredictionHeadTransform(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = torch.nn.LayerNorm(config.hidden_size, eps=1e-12)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


###################
# Bert Heads
###################
class BertLMPredictionHead(nn.Module):

    def __init__(self, config, bert_model_embedding_weights):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)
        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(bert_model_embedding_weights.size(1),
                                 bert_model_embedding_weights.size(0))
        self.decoder.weight = bert_model_embedding_weights

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class BertOnlyMLMHead(nn.Module):

    def __init__(self, config, bert_model_embedding_weights):
        super().__init__()
        self.predictions = BertLMPredictionHead(config,
                                                bert_model_embedding_weights)

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class BertOnlyNSPHead(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output: torch.Tensor) -> torch.Tensor:
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score
