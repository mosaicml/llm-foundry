# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

"""Converts Huggingface Causal LM to Prefix LM.

Conversion does lightweight surgery on a HuggingFace
Causal LM to convert it to a Prefix LM.

Prefix LMs accepts a `bidirectional_mask` input in `forward`
and treat the input prompt as the prefix in `generate`.
"""

import math
import warnings
from types import MethodType
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from transformers.models.bloom.modeling_bloom import (
    BaseModelOutputWithPastAndCrossAttentions, BloomForCausalLM, BloomModel,
    CausalLMOutputWithCrossAttentions, CrossEntropyLoss)
from transformers.models.bloom.modeling_bloom import \
    _expand_mask as _expand_mask_bloom
from transformers.models.bloom.modeling_bloom import \
    _make_causal_mask as _make_causal_mask_bloom
from transformers.models.bloom.modeling_bloom import logging
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
from transformers.models.gpt_neo.modeling_gpt_neo import GPTNeoForCausalLM
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXForCausalLM
from transformers.models.gptj.modeling_gptj import GPTJForCausalLM
from transformers.models.opt.modeling_opt import OPTForCausalLM
from transformers.models.opt.modeling_opt import \
    _expand_mask as _expand_mask_opt
from transformers.models.opt.modeling_opt import \
    _make_causal_mask as _make_causal_mask_opt

logger = logging.get_logger(__name__)

_SUPPORTED_GPT_MODELS = (
    GPT2LMHeadModel,
    GPTJForCausalLM,
    GPTNeoForCausalLM,
    GPTNeoXForCausalLM,
)

CAUSAL_GPT_TYPES = Union[GPT2LMHeadModel, GPTJForCausalLM, GPTNeoForCausalLM,
                         GPTNeoXForCausalLM,]


def _convert_gpt_causal_lm_to_prefix_lm(
        model: CAUSAL_GPT_TYPES) -> CAUSAL_GPT_TYPES:
    """Converts a GPT-style Causal LM to a Prefix LM.

    Supported HuggingFace model classes:
        - `GPT2LMHeadModel`
        - `GPTNeoForCausalLM`
        - `GPTNeoXForCausalLM`
        - `GPTJForCausalLM`

    See `convert_hf_causal_lm_to_prefix_lm` for more details.
    """
    if hasattr(model, '_prefix_lm_converted'):
        return model

    assert isinstance(model, _SUPPORTED_GPT_MODELS)
    assert model.config.add_cross_attention == False, 'Only supports GPT-style decoder-only models'

    def _get_attn_modules(model: CAUSAL_GPT_TYPES) -> List[torch.nn.Module]:
        """Helper that gets a list of the model's attention modules.

        Each module has a `bias` buffer used for causal masking. The Prefix LM
        conversion adds logic to dynamically manipulate these biases to support
        Prefix LM attention masking.
        """
        attn_modules = []

        if isinstance(model, GPTNeoXForCausalLM):
            blocks = model.gpt_neox.layers
        else:
            blocks = model.transformer.h

        for block in blocks:  # type: ignore
            if isinstance(model, GPTNeoForCausalLM):
                # Ignore "local" layers in this model type
                if block.attn.attention_type != 'global':
                    continue
                attn_module = block.attn.attention
            elif isinstance(model, GPTNeoXForCausalLM):
                attn_module = block.attention
            else:
                attn_module = block.attn

            attn_modules.append(attn_module)

        return attn_modules

    # Rename methods to allow:
    #  - new `forward` to wrap original `forward`
    #  - new `generate` to wrap original `generate`
    setattr(model, '_original_forward', getattr(model, 'forward'))
    setattr(model, '_original_generate', getattr(model, 'generate'))

    def forward(
        self: CAUSAL_GPT_TYPES,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        bidirectional_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """Wraps original forward to enable PrefixLM attention."""

        def call_og_forward():
            if isinstance(self, GPTNeoXForCausalLM):
                return self._original_forward(
                    input_ids=input_ids,
                    past_key_values=past_key_values,
                    attention_mask=attention_mask,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds,
                    labels=labels,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
            else:
                return self._original_forward(
                    input_ids=input_ids,
                    past_key_values=past_key_values,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds,
                    labels=labels,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )

        if bidirectional_mask is None:
            # This wrapper is a no-op if bidirectional masks are not supplied
            return call_og_forward()  # type: ignore
        assert isinstance(bidirectional_mask, torch.Tensor)

        attn_modules = _get_attn_modules(model)

        # Handle bidirectional_mask sizing
        # Note: all attn_modules.bias have the same size
        b, s = bidirectional_mask.shape
        max_length = attn_modules[0].bias.shape[-1]  # type: ignore
        if s > max_length:
            raise ValueError(
                f'bidirectional_mask sequence length (={s}) exceeds the ' +\
                f'max length allowed by the model ({max_length}).'
            )
        assert s <= max_length
        if s < max_length:
            pad = torch.zeros((int(b), int(max_length - s)),
                              dtype=bidirectional_mask.dtype,
                              device=bidirectional_mask.device)
            bidirectional_mask = torch.cat([bidirectional_mask, pad], dim=1)
        bidirectional = bidirectional_mask.unsqueeze(1).unsqueeze(1)

        # Incorporate the bidirectional mask into the original causal mask
        for attn_module in attn_modules:
            attn_module.bias.data = torch.logical_or(
                attn_module.bias.data, bidirectional)  # type: ignore

        # Collect outputs using the model's original forward method
        output = call_og_forward()

        # Reset the masks
        for attn_module in attn_modules:
            attn_module.bias.data = torch.tril(
                attn_module.bias.data[0, 0])[None, None]  # type: ignore

        # Return the outputs
        return output

    def generate(self: CAUSAL_GPT_TYPES, *args: tuple, **kwargs: Dict[str,
                                                                      Any]):
        """Wraps original generate to enable PrefixLM attention."""
        attn_modules = _get_attn_modules(model)

        # A convenient answer to PrefixLM generation is to set the causal mask
        # to be bidirectional. All the tokens in the input prompt can attend to
        # one another and, since tokens are generated one-by-one, each new
        # token gets to see everything behind it. This depends on activations
        # being cached and not updated, which is how the HF implementation works.
        for attn_module in attn_modules:
            attn_module.bias.data[:] = 1  # type: ignore

        # Collect outputs using the model's original forward method
        output = self._original_generate(*args, **kwargs)  # type: ignore

        # Reset the masks
        for attn_module in attn_modules:
            attn_module.bias.data = torch.tril(
                attn_module.bias.data[0, 0])[None, None]  # type: ignore

        # Return the outputs
        return output

    # Replace `forward` and `generate` with the new wrappers
    setattr(model, 'forward', MethodType(forward, model))
    setattr(model, 'generate', MethodType(generate, model))

    # Finally, tag the model so that this conversion cannot happen again.
    setattr(model, '_prefix_lm_converted', True)
    return model


def _convert_bloom_causal_lm_to_prefix_lm(
        model: BloomForCausalLM) -> BloomForCausalLM:
    """Converts a BLOOM Causal LM to a Prefix LM.

    Supported HuggingFace model classes:
        - `BloomForCausalLM`

    See `convert_hf_causal_lm_to_prefix_lm` for more details.
    """
    if hasattr(model, '_prefix_lm_converted'):
        return model

    assert isinstance(model, BloomForCausalLM)
    assert model.config.add_cross_attention == False, 'Only supports BLOOM decoder-only models'

    # Modified from transformers.models.bloom.modeling_bloom.BloomModel._prepare_attn_mask
    # https://github.com/huggingface/transformers/blob/v4.25.1/src/transformers/models/bloom/modeling_bloom.py#L648
    def _prepare_attn_mask(
        self: BloomModel,
        attention_mask: torch.Tensor,
        bidirectional_mask: Optional[torch.Tensor],
        input_shape: Tuple[int, int],
        past_key_values_length: int,
    ) -> torch.BoolTensor:
        # create causal mask
        # [batch_size, seq_length] -> [batch_size, 1, tgt_length, src_length]
        combined_attention_mask = None
        device = attention_mask.device
        _, src_length = input_shape

        if src_length > 1:
            combined_attention_mask = _make_causal_mask_bloom(
                input_shape,
                device=device,
                past_key_values_length=past_key_values_length)
            # Make use of the batch-specific `bidirectional_mask` attribute set
            # by the parent module in its (new) `forward` method wrapper
            if bidirectional_mask is not None:
                # The two masks should have the same size
                assert attention_mask.shape == bidirectional_mask.shape

                # [batch_size, seq_length] -> [batch_size, 1, tgt_length, src_length]
                expanded_bidirectional_mask = _expand_mask_bloom(
                    bidirectional_mask, tgt_length=src_length)
                combined_attention_mask = torch.logical_and(
                    combined_attention_mask, expanded_bidirectional_mask)

        # [batch_size, seq_length] -> [batch_size, 1, tgt_length, src_length]
        expanded_attn_mask = _expand_mask_bloom(attention_mask,
                                                tgt_length=src_length)
        combined_attention_mask = (expanded_attn_mask
                                   if combined_attention_mask is None else
                                   expanded_attn_mask | combined_attention_mask)

        return combined_attention_mask

    # Modified from transformers.models.bloom.modeling_bloom._prepare_alibi_transformer
    # https://github.com/huggingface/transformers/blob/v4.25.1/src/transformers/models/bloom/modeling_bloom.py#L87
    def _build_alibi_tensor(
        self: BloomModel,
        batch_size: int,
        query_length: int,
        key_length: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        num_heads = self.config.n_head

        closest_power_of_2 = 2**math.floor(math.log2(num_heads))
        base = torch.tensor(2**(-(2**-(math.log2(closest_power_of_2) - 3))),
                            device=device,
                            dtype=torch.float32)
        powers = torch.arange(1,
                              1 + closest_power_of_2,
                              device=device,
                              dtype=torch.int32)
        slopes = torch.pow(base, powers)

        if closest_power_of_2 != num_heads:
            extra_base = torch.tensor(
                2**(-(2**-(math.log2(2 * closest_power_of_2) - 3))),
                device=device,
                dtype=torch.float32)
            num_remaining_heads = min(closest_power_of_2,
                                      num_heads - closest_power_of_2)
            extra_powers = torch.arange(1,
                                        1 + 2 * num_remaining_heads,
                                        2,
                                        device=device,
                                        dtype=torch.int32)
            slopes = torch.cat(
                [slopes, torch.pow(extra_base, extra_powers)], dim=0)

        qa = torch.arange(query_length, device=device,
                          dtype=torch.int32).view(-1, 1)
        ka = torch.arange(key_length, device=device,
                          dtype=torch.int32).view(1, -1)
        diffs = qa - ka + key_length - query_length
        diffs = -diffs.abs()
        alibi = slopes.view(1, num_heads, 1, 1) * diffs.view(
            1, 1, query_length, key_length)
        alibi = alibi.expand(batch_size, -1, -1,
                             -1).reshape(-1, query_length, key_length)
        return alibi.to(dtype)

    # Modified from transformers.models.bloom.modeling_bloom.BloomModel.forward
    # Note: The modified code is surrounded with #### START/END #### comments
    # and one new argument (`bidirectional_mask`) is added to the signature.
    KeyValueT = Tuple[torch.Tensor, torch.Tensor]

    def forward(  # type: ignore
        self: BloomModel,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[KeyValueT, ...]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        bidirectional_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **deprecated_arguments) -> Union[Tuple[
            torch.Tensor, ...], BaseModelOutputWithPastAndCrossAttentions]:
        if deprecated_arguments.pop('position_ids', False) is not False:
            # `position_ids` could have been `torch.Tensor` or `None` so
            # defaulting pop to `False` allows to detect if users were
            # passing explicitly `None`
            warnings.warn(
                '`position_ids` have no functionality in BLOOM and will be removed in v5.0.0. ' +\
                'You can safely ignore passing `position_ids`.',
                FutureWarning,
            )
        if len(deprecated_arguments) > 0:
            raise ValueError(
                f'Got unexpected arguments: {deprecated_arguments}')

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (output_hidden_states
                                if output_hidden_states is not None else
                                self.config.output_hidden_states)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                'You cannot specify both input_ids and inputs_embeds at the same time'
            )
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError(
                'You have to specify either input_ids or inputs_embeds')

        if past_key_values is None:
            past_key_values = tuple([None] * len(self.h))  # type: ignore

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape batch_size x num_heads x N x N
        # head_mask has shape n_layer x batch x num_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        hidden_states = self.word_embeddings_layernorm(inputs_embeds)

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        # Compute alibi tensor: check build_alibi_tensor documentation
        seq_length_with_past = seq_length
        past_key_values_length = 0
        if past_key_values[0] is not None:  # type: ignore
            tmp = past_key_values[0][0]  # type: ignore
            past_key_values_length = tmp.shape[2]  # type: ignore
            seq_length_with_past = seq_length_with_past + past_key_values_length
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length_with_past),
                                        device=hidden_states.device)
        else:
            attention_mask = attention_mask.to(hidden_states.device)

        ##### ALL NON-SIGNATURE MODIFICATIONS ARE CONTAINED TO THIS BLOCK [STARTS HERE] #####
        alibi = self._build_alibi_tensor(
            batch_size=batch_size,
            query_length=seq_length,
            key_length=seq_length_with_past,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        causal_mask = self._prepare_attn_mask(
            attention_mask,
            bidirectional_mask,
            input_shape=(batch_size, seq_length),
            past_key_values_length=past_key_values_length,
        )
        ##### ALL NON-SIGNATURE MODIFICATIONS ARE CONTAINED TO THIS BLOCK [ENDS HERE] #####

        for i, (block,
                layer_past) in enumerate(zip(self.h,
                                             past_key_values)):  # type: ignore

            if output_hidden_states:
                hst = (hidden_states,)
                all_hidden_states = all_hidden_states + hst  # type: ignore

            if self.gradient_checkpointing and self.training:

                if use_cache:
                    logger.warning(
                        '`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...'
                    )
                    use_cache = False

                def create_custom_forward(module):

                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs,
                                      use_cache=use_cache,
                                      output_attentions=output_attentions)

                    return custom_forward

                outputs = torch.utils.checkpoint.checkpoint(  # type: ignore
                    create_custom_forward(block),
                    hidden_states,
                    alibi,
                    causal_mask,
                    head_mask[i],  # type: ignore
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=causal_mask,
                    head_mask=head_mask[i],  # type: ignore
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    alibi=alibi,
                )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)  # type: ignore

            if output_attentions:
                oa = (outputs[2 if use_cache else 1],)  # type: ignore
                all_self_attentions = all_self_attentions + oa  # type: ignore

        # Add last hidden state
        hidden_states = self.ln_f(hidden_states)

        if output_hidden_states:
            hst = (hidden_states,)
            all_hidden_states = all_hidden_states + hst  # type: ignore

        if not return_dict:
            return tuple(v for v in [
                hidden_states, presents, all_hidden_states, all_self_attentions
            ] if v is not None)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

    # Make it so model.transformer has the new helper methods and new
    # `forward` method
    setattr(model.transformer, '_prepare_attn_mask',
            MethodType(_prepare_attn_mask, model.transformer))
    setattr(model.transformer, '_build_alibi_tensor',
            MethodType(_build_alibi_tensor, model.transformer))
    setattr(model.transformer, 'forward', MethodType(forward,
                                                     model.transformer))

    # In order to actually use the new argument we've added to
    # model.transformer, we need to update the parent module's `forward` to
    # accept/pass the same new argument.
    # We add 2 lines to handle that change.
    # Both lines are tagged with "# WE'RE ADDING A NEW ARGUMENT!"
    KeyValueT = Tuple[torch.Tensor, torch.Tensor]

    def forward(
        self: BloomForCausalLM,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[KeyValueT, ...]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        # WE'RE ADDING A NEW ARGUMENT! (Change 1/2)
        bidirectional_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **deprecated_arguments
    ) -> Union[Tuple[torch.Tensor], CausalLMOutputWithCrossAttentions]:
        """Replacement forward method for BloomCausalLM."""
        if deprecated_arguments.pop('position_ids', False) is not False:
            # `position_ids` could have been `torch.Tensor` or `None` so
            # defaulting pop to `False` allows to detect if users were passing
            # explicitly `None`
            warnings.warn(
                '`position_ids` have no functionality in BLOOM and will be removed ' +\
                'in v5.0.0. You can safely ignore passing `position_ids`.',
                FutureWarning,
            )
        if len(deprecated_arguments) > 0:
            raise ValueError(
                f'Got unexpected arguments: {deprecated_arguments}')

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            # WE'RE ADDING A NEW ARGUMENT! (Change 2/2)
            bidirectional_mask=bidirectional_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            batch_size, seq_length, vocab_size = shift_logits.shape
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(batch_size * seq_length, vocab_size),
                shift_labels.view(batch_size * seq_length))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    # To handle generation, re-write `prepare_inputs_for_generation` to
    # implement the bidirectional logic.
    def prepare_inputs_for_generation(self: BloomForCausalLM,
                                      input_ids: torch.LongTensor,
                                      past: Optional[torch.Tensor] = None,
                                      attention_mask: Optional[
                                          torch.Tensor] = None,
                                      **kwargs) -> dict:
        # only last token for input_ids if past is not None
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)  # type: ignore
            # We can turn off bidirectional masking after the prefix
            # has been encoded into `past`
            bidirectional_mask = None

            # the cache may be in the stardard format (e.g. in contrastive
            # search), convert to bloom's format if needed
            if past[0][0].shape[0] == input_ids.shape[0]:
                past = self._convert_to_bloom_cache(past)

        else:
            # If we're here, `input_ids` contains the prefix. Encode it with
            # bidirectional attention.
            bidirectional_mask = torch.ones_like(input_ids)

        return {
            'input_ids': input_ids,
            'past_key_values': past,
            # "use_cache": kwargs.get("use_cache"),
            # Requires this. TODO(Alex): Confirm this supports other decoding strategies.
            'use_cache': True,
            'attention_mask': attention_mask,
            'bidirectional_mask': bidirectional_mask,
        }

    # Register the new `forward` and `prepare_inputs_for_generation` methods
    # with the model
    setattr(model, 'forward', MethodType(forward, model))
    setattr(model, 'prepare_inputs_for_generation',
            MethodType(prepare_inputs_for_generation, model))

    # Finally, tag the model so that this conversion cannot happen again.
    setattr(model, '_prefix_lm_converted', True)
    return model


def _convert_opt_causal_lm_to_prefix_lm(
        model: OPTForCausalLM) -> OPTForCausalLM:
    """Converts an OPT Causal LM to a Prefix LM.

    Supported HuggingFace model classes:
        - `OPTForCausalLM`

    See `convert_hf_causal_lm_to_prefix_lm` for more details.
    """
    if hasattr(model, '_prefix_lm_converted'):
        return model

    assert isinstance(model, OPTForCausalLM)
    assert model.config.add_cross_attention == False, 'Only supports OPT decoder-only models'

    # Rename methods to allow:
    #  - new `forward` to wrap original `forward`
    #  - new `generate` to wrap original `generate`
    setattr(model, '_original_forward', getattr(model, 'forward'))
    setattr(model, '_original_generate', getattr(model, 'generate'))

    model.model.decoder.bidirectional_mask = None

    # Modified from transformers.models.bloom.modeling_opt.OPTDecoder._prepare_decoder_attn_mask
    # https://github.com/huggingface/transformers/blob/v4.25.1/src/transformers/models/opt/modeling_opt.py#L532
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape,
                                        inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            # 'g' indicates generation mode. Causal mask replaced with 0.
            if self.bidirectional_mask == 'g':
                bsz, src_length = input_shape
                combined_attention_mask = torch.zeros(
                    (bsz, 1, src_length, src_length + past_key_values_length),
                    dtype=inputs_embeds.dtype,
                    device=inputs_embeds.device)
            else:
                combined_attention_mask = _make_causal_mask_opt(
                    input_shape,
                    inputs_embeds.dtype,
                    past_key_values_length=past_key_values_length).to(
                        inputs_embeds.device)

                # Make use of the batch-specific `bidirectional_mask` attribute
                # set by the parent module in its (new) `forward` method wrapper
                if self.bidirectional_mask is not None:
                    # The two masks should have the same size
                    assert attention_mask.shape == self.bidirectional_mask.shape

                    # [batch_size, seq_length] -> [batch_size, 1, tgt_length, src_length]
                    expanded_bidirectional_mask = _expand_mask_opt(
                        self.bidirectional_mask,
                        inputs_embeds.dtype,
                        tgt_len=input_shape[-1]).to(inputs_embeds.device)
                    combined_attention_mask = torch.maximum(
                        expanded_bidirectional_mask, combined_attention_mask)

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask_opt(attention_mask,
                                                  inputs_embeds.dtype,
                                                  tgt_len=input_shape[-1]).to(
                                                      inputs_embeds.device)
            combined_attention_mask = (expanded_attn_mask
                                       if combined_attention_mask is None else
                                       expanded_attn_mask +
                                       combined_attention_mask)

        return combined_attention_mask

    # Make it so model.model.decoder uses the above `_prepare_decoder_attn_mask`
    # in place of the original method
    setattr(model.model.decoder, '_prepare_decoder_attention_mask',
            MethodType(_prepare_decoder_attention_mask, model.model.decoder))

    def forward(
        self: OPTForCausalLM,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        bidirectional_mask: Optional[torch.ByteTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):

        def call_og_forward():
            return self._original_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        if bidirectional_mask is None:
            # This wrapper is a no-op if bidirectional masks are not supplied
            return call_og_forward()

        # Temporarily set `bidirectional_mask` in the child module
        self.model.decoder.bidirectional_mask = bidirectional_mask

        # Apply the original forward method (the model will use the mask that
        # was just set)
        try:
            outputs = call_og_forward()
        except:
            self.model.decoder.bidirectional_mask = None
            raise

        # Reset the `bidirectional_mask` attribute to None
        self.model.decoder.bidirectional_mask = None

        # Return the outputs
        return outputs

    def generate(self: OPTForCausalLM, *args: tuple, **kwargs: Dict[str, Any]):
        """Wraps original generate to enable PrefixLM-style attention."""
        # Flag the child module to use generation-style attention masking
        self.model.decoder.bidirectional_mask = 'g'

        # Collect outputs using the model's original forward method
        try:
            output = self._original_generate(*args, **kwargs)
        except:
            self.model.decoder.bidirectional_mask = None
            raise

        # Reset the `bidirectional_mask` attribute to None
        self.model.decoder.bidirectional_mask = None

        # Return the output
        return output

    # Replace `forward` and `generate` with the new wrappers
    setattr(model, 'forward', MethodType(forward, model))
    setattr(model, 'generate', MethodType(generate, model))

    # Finally, tag the model so that this conversion cannot happen again.
    setattr(model, '_prefix_lm_converted', True)
    return model


_SUPPORTED_HF_MODELS = _SUPPORTED_GPT_MODELS + (BloomForCausalLM,
                                                OPTForCausalLM)

CAUSAL_LM_TYPES = Union[GPT2LMHeadModel, GPTJForCausalLM, GPTNeoForCausalLM,
                        GPTNeoXForCausalLM, BloomForCausalLM, OPTForCausalLM]


def convert_hf_causal_lm_to_prefix_lm(
        model: CAUSAL_LM_TYPES) -> CAUSAL_LM_TYPES:
    """Converts a HuggingFace Causal LM to a Prefix LM.

    Supported HuggingFace model classes:
        - `GPT2LMHeadModel`
        - `GPTNeoForCausalLM`
        - `GPTNeoXForCausalLM`
        - `GPTJForCausalLM`
        - `BloomForCausalLM`
        - `OPTForCausalLM`

    Conversion to a Prefix LM is done by modifying the `forward` method, and possibly also the
    `generate` method and/or select underlying methods depending on the model class.

    These changes preserve the model API, but add a new input to `forward`: "bidirectional_mask".

    Notes on training:
        To actually train the converted model as a Prefix LM, training batches will need to indicate
        the prefix/target structure by including `bidirectional_mask` as part of the batch inputs.

        **This is not a standard input and requires custom layers either within or after your dataloader.**

        In addition to adding `bidirectional_mask` to the batch, this custom code should modify `labels`
        such that `batch['labels'][batch['bidirectional_mask'] == 1] == -100`.
        That is, the prefix portion of the sequence should not generate any loss. Loss should only be
        generated by the target portion of the sequence.

    Notes on `GPTNeoForCausalLM`:
        To simplify the implementation, "global" and "local" attention layers are handled differently.
        For "global" layers, we handle conversion as described above. For "local" layers, which use a
        causal attention mask within a restricted local window, we do not alter the masking.

    Notes on `forward` method conversion:
        After conversion, the `forward` method will handle a new input, `bidirectional_mask`,
        which should be a [batch_size, seq_length] byte tensor, where 1 indicates token positions
        belonging to the prefix (prefix tokens can attend to one another bidirectionally), and
        0 indicates token positions belonging to the target.

        The new `forward` method will incorporate `bidirectional_mask` (if supplied) into the existing
        causal mask, call the original `forward` method, and (if the causal mask is a buffer) reset
        the causal masks before returning the result.

    Notes on `generate` method conversion:
        After conversion, the `generate` method will have the same signature but will internally
        convert all causal masks to be purely bidirectional, call the original `generate` method, and
        (where appropriate) reset the causal masks before returning the result.

        This works thanks to the logic of the HuggingFace `generate` API, which first encodes the token
        "prompt" passed to `generate` (which is treated as the prefix) and then sequentially generates
        each new token. Encodings are cached as generation happens, so all prefix tokens can attend to one
        another (as expected in a Prefix LM) and generated tokens can only attend to prefix tokens and
        previously-generated tokens (also as expected in a Prefix LM).

    To preserve the API, the original methods are renamed to `_original_forward` and
    `_original_generate`, and replaced with new `forward` and `generate` methods that wrap
    them, respectively. Although implementation details vary by model class.
    """
    if isinstance(model, _SUPPORTED_GPT_MODELS):
        return _convert_gpt_causal_lm_to_prefix_lm(model)

    elif isinstance(model, BloomForCausalLM):
        return _convert_bloom_causal_lm_to_prefix_lm(model)

    elif isinstance(model, OPTForCausalLM):
        return _convert_opt_causal_lm_to_prefix_lm(model)

    else:
        raise TypeError(
            f'Cannot convert model to Prefix LM. ' +\
            f'Model does not belong to set of supported HF models:' +\
            f'\n{_SUPPORTED_HF_MODELS}'
        )
