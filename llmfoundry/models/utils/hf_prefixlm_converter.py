# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Converts Huggingface Causal LM to Prefix LM.

Conversion does lightweight surgery on a HuggingFace
Causal LM to convert it to a Prefix LM.

Prefix LMs accepts a `bidirectional_mask` input in `forward`
and treat the input prompt as the prefix in `generate`.
"""

from types import MethodType
from typing import Any, List, MutableMapping, Optional, Tuple, Union

import torch
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
from transformers.models.gpt_neo.modeling_gpt_neo import GPTNeoForCausalLM
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXForCausalLM
from transformers.models.gptj.modeling_gptj import GPTJForCausalLM

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

        for block in blocks:
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
            return call_og_forward()
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
            assert isinstance(attn_module.bias, torch.Tensor)
            attn_module.bias.data = torch.logical_or(attn_module.bias.data,
                                                     bidirectional)

        # Collect outputs using the model's original forward method
        output = call_og_forward()

        # Reset the masks
        for attn_module in attn_modules:
            attn_module.bias.data = torch.tril(
                attn_module.bias.data[0, 0])[None, None]  # type: ignore

        # Return the outputs
        return output

    def generate(self: CAUSAL_GPT_TYPES, *args: Any, **kwargs: Any):
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
        output = self._original_generate(*args, **kwargs)

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


_SUPPORTED_HF_MODELS = _SUPPORTED_GPT_MODELS

CAUSAL_LM_TYPES = Union[GPT2LMHeadModel, GPTJForCausalLM, GPTNeoForCausalLM,
                        GPTNeoXForCausalLM]


def convert_hf_causal_lm_to_prefix_lm(
        model: CAUSAL_LM_TYPES) -> CAUSAL_LM_TYPES:
    """Converts a HuggingFace Causal LM to a Prefix LM.

    Supported HuggingFace model classes:
        - `GPT2LMHeadModel`
        - `GPTNeoForCausalLM`
        - `GPTNeoXForCausalLM`
        - `GPTJForCausalLM`

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
    else:
        raise TypeError(
            f'Cannot convert model to Prefix LM. ' +\
            f'Model does not belong to set of supported HF models:' +\
            f'\n{_SUPPORTED_HF_MODELS}'
        )


def add_bidirectional_mask_if_missing(batch: MutableMapping):
    """Attempts to add bidirectional_mask to batch if missing.

    Raises:
        KeyError if bidirectional_mask is missing and can't be inferred
    """
    if 'bidirectional_mask' not in batch:
        if batch.get('mode', None) == 'icl_task':
            batch['bidirectional_mask'] = batch['attention_mask'].clone()
            for i, continuation_indices in enumerate(
                    batch['continuation_indices']):
                batch['bidirectional_mask'][i, continuation_indices] = 0
        elif ('labels' in batch) and ('attention_mask' in batch):
            batch['bidirectional_mask'] = torch.logical_and(
                torch.eq(batch['attention_mask'], 1),
                torch.eq(batch['labels'], -100),
            ).type_as(batch['attention_mask'])
        else:
            raise KeyError(
                'No bidirectional_mask in batch and not sure how to construct one.'
            )
