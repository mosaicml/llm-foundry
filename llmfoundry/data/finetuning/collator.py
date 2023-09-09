# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import logging
import warnings
from typing import Any, Dict, List, Optional, Union

import torch
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

log = logging.getLogger(__name__)

# HuggingFace hardcodes the ignore index to -100
_HF_IGNORE_INDEX = -100


class Seq2SeqFinetuningCollator:
    """A general-purpose collator for sequence-to-sequence training/evaluation.

    Args:
        tokenizer: A HuggingFace tokenizer. Must have a pad_token set.
        max_seq_len (int): The maximum sequence length of the combined
            context/target sequence (decoder-only format) or of each the
            context sequence and target sequence (encoder-decoder format).
        decoder_only_format (bool): Whether to format the batches for a
            decoder-only model (if True) or an encoder-decoder model (if False).
        allow_pad_trimming (bool, optional): Whether to allow the collator
            to trim padding, which may result in smaller but inconsistent batch
            sizes. Default: ``False`` ensures that all sequences are max_seq_len.
        separator_text (str | bool, optional): If a string is provided, it will
            be used to separate the context and target sequences (appended to end
            of context). If ``True``, will use the tokenizer's sep_token, which must
            be defined. Only applicable for decoder-only formatting.
        format_for_generation (bool, optional): Whether to format the batch such
            that context and target sequences remain separated, which is useful
            when using the context to generate text which should be compared to the
            target (e.g., during evaluation). Default: ``False``.
        batch_metadata (dict, optional): A dictionary of metadata which will be added
            to the batch.
    """

    def __init__(
        self,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        max_seq_len: int,
        decoder_only_format: bool,
        allow_pad_trimming: bool = False,
        separator_text: Optional[Union[str, bool]] = None,
        format_for_generation: bool = False,
        batch_metadata: Optional[Dict[str, Any]] = None,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.decoder_only_format = decoder_only_format
        self.format_for_generation = format_for_generation
        self.batch_metadata = batch_metadata or {}

        # Trimming will always be skipped on at least the first __call__
        self._allow_pad_trimming = allow_pad_trimming
        self._seen_first_batch = False

        illegal_keys = [
            'input_ids', 'labels', 'attention_mask', 'decoder_input_ids',
            'decoder_attention_mask', 'generate_output'
        ]
        found_keys = []
        for illegal_key in illegal_keys:
            if illegal_key in self.batch_metadata:
                found_keys.append(illegal_key)
        if found_keys:
            raise ValueError(
                f'The following keys are in batch_metadata but are not allowed: {", ".join(found_keys)}.\n' +\
                f'You cannot use keys that are used directly by the models. The prohibited keys are:\n' +\
                f'{", ".join(illegal_keys)}'
            )
        if self.format_for_generation:
            self.batch_metadata['generate_output'] = True

        if (max_seq_len % 8) != 0:
            log.warning(
                'For performance, a max_seq_len as a multiple of 8 is recommended.'
            )

        if self.tokenizer.pad_token_id is None:
            raise ValueError(
                f'{self.__class__.__name__} requires that the tokenizer has the pad token set, but it is None'
            )

        self.separator_tokens = []
        if separator_text and decoder_only_format:
            if separator_text == True:
                # Use the tokenizer's sep token or throw an error if undefined
                if self.tokenizer.sep_token_id is None:
                    raise ValueError(
                        'Setting separator_text=True requires that the tokenizer has sep_token_id but it has not been set. ' +\
                        'Please pass a string argument for separator_text or set sep_token_id in the tokenizer.'
                    )
                self.separator_tokens = [self.tokenizer.sep_token_id]
            else:
                # Convert the string separator_text into token(s)
                self.separator_tokens = tokenizer(
                    separator_text, add_special_tokens=False).input_ids

        self._warned_context = False
        self._warned_target = False

    def __call__(self, examples: List[Dict[str,
                                           Any]]) -> Dict[str, torch.Tensor]:
        for check_key in ['input_ids', 'labels', 'attention_mask']:
            if check_key not in examples[0]:
                raise KeyError(
                    f'Examples returned by dataset do not include required key: {check_key}'
                )

        if self.decoder_only_format:
            batch = self._process_and_batch_decoder_only(examples)
        else:
            batch = self._process_and_batch_encoder_decoder(examples)

        # Add any batch_metadata
        batch_size = batch['input_ids'].shape[0]
        batch.update({
            k: torch.tensor([v] * batch_size)
            for k, v in self.batch_metadata.items()
        })

        return batch

    def _process_and_batch_decoder_only(
            self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Steps explained in comments
        processed_examples = []
        for example in examples:
            context = ensure_list(example['input_ids'])
            target = ensure_list(example['labels'])
            # First, get rid of any padding tokens
            context = [t for t in context if t != self.tokenizer.pad_token_id]
            target = [t for t in target if t != self.tokenizer.pad_token_id]
            # Second, append any separator tokens to the context tokens
            if self.separator_tokens:
                context = context + self.separator_tokens
            # Third, ensure that the target text ends with an eos tag
            if target[-1] != self.tokenizer.eos_token_id:
                target = target + [self.tokenizer.eos_token_id]

            n_context = len(context)
            n_target = len(target)

            if n_context >= self.max_seq_len:
                if not self._warned_context:
                    warnings.warn(
                        f'Skipping example because CONTEXT length={n_context} leaves no room ' +\
                        f'for TARGET tokens because max_seq_len={self.max_seq_len}. ' +\
                        f'If this causes downstream issues because of inconsistent batch sizes, ' +\
                        f'consider increasing max_seq_len or using example packing.'
                    )
                    self._warned_context = True
                continue

            if self.format_for_generation:
                # When formatting for generation, we need to keep input_ids and
                # labels separate. The input_ids (context) will be fed into the
                # generator and the labels will be used by the eval metric.
                input_ids = context[-self.max_seq_len:]
                n_context = len(input_ids)
                attention_mask = [1] * n_context
                bidirectional_mask = [1] * n_context
                # Annoyingly, we need to pad the everything but input_ids
                # and attention_mask ourselves
                i_pad = [self.tokenizer.pad_token_id
                        ] * (self.max_seq_len - n_target)
                z_pad = [0] * (self.max_seq_len - n_context)
                if self.tokenizer.padding_side == 'left':
                    labels = i_pad + target
                    bidirectional_mask = z_pad + bidirectional_mask
                else:
                    labels = target + i_pad
                    bidirectional_mask = bidirectional_mask + z_pad

            else:
                # We need to concatenate the context and target to get the
                # full input sequence, cutting off any excess tokens from the
                # end of the target
                if n_context + n_target > self.max_seq_len:
                    old_n_target = int(n_target)
                    n_target = self.max_seq_len - n_context
                    if not self._warned_target:
                        warnings.warn(
                            f'Truncating TARGET sequence of length={old_n_target} to length={n_target}, ' +\
                            f'so context+target fit max_seq_len={self.max_seq_len}. If truncation is ' +\
                            f'a problem, consider increasing max_seq_len.')
                        self._warned_target = True
                    target = target[-n_target:]
                    target[-1] = self.tokenizer.eos_token_id
                n_total = n_context + n_target

                input_ids = context + target
                labels = ([_HF_IGNORE_INDEX] * n_context) + target
                attention_mask = [1] * n_total
                # bidirectional_mask is used by our prefix lm model variants
                bidirectional_mask = ([1] * n_context) + ([0] * n_target)

                # Annoyingly, we need to pad the everything but input_ids
                # and attention_mask ourselves
                i_pad = [_HF_IGNORE_INDEX] * (self.max_seq_len - n_total)
                z_pad = [0] * (self.max_seq_len - n_total)
                if self.tokenizer.padding_side == 'left':
                    labels = i_pad + labels
                    bidirectional_mask = z_pad + bidirectional_mask
                else:
                    labels = labels + i_pad
                    bidirectional_mask = bidirectional_mask + z_pad

            # Update the example
            example['input_ids'] = input_ids
            example['labels'] = labels
            example['attention_mask'] = attention_mask
            example['bidirectional_mask'] = bidirectional_mask

            processed_examples.append(example)

        batch = self.tokenizer.pad(
            processed_examples,
            padding='max_length',
            max_length=self.max_seq_len,
            return_tensors='pt',
        )

        # This logic prevents trimming on at least the first batch
        if not (self._allow_pad_trimming and self._seen_first_batch):
            self._seen_first_batch = True
            return batch
        self._seen_first_batch = True

        # The batch is ready, but we can trim padding for efficiency
        multiple_of = 8

        n_non_padding = batch['attention_mask'].sum(dim=1).max()
        keep_tokens = int(multiple_of * torch.ceil(n_non_padding / multiple_of))
        for k, v in batch.items():
            if len(v.shape) < 2:
                continue
            if k == 'labels' and self.format_for_generation:
                continue
            if self.tokenizer.padding_side == 'left':
                batch[k] = v[:, -keep_tokens:].contiguous()
            else:
                batch[k] = v[:, :keep_tokens].contiguous()

        return batch

    def _process_and_batch_encoder_decoder(
            self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # The encoder-decoder case is has some gotchas.
        # Steps are explained in comments.
        processed_examples = []
        for example in examples:
            context = ensure_list(example['input_ids'])
            target = ensure_list(example['labels'])
            # ... first, get rid of any padding that was already applied
            context = [t for t in context if t != self.tokenizer.pad_token_id]
            target = [t for t in target if t != self.tokenizer.pad_token_id]
            # ... second, ensure that the target text ends with an eos tag
            if target[-1] != self.tokenizer.eos_token_id:
                target = target + [self.tokenizer.eos_token_id]
            # ... third, we need to pad labels ourselves. Because HF.
            if len(target) < self.max_seq_len:
                i_pad = [_HF_IGNORE_INDEX] * (self.max_seq_len - len(target))
                target = target + i_pad
            else:
                if not self._warned_target:
                    warnings.warn(
                        f'Truncating TARGET sequence of length={len(target)} ' +\
                        f'to max_seq_len={self.max_seq_len}. If truncation is ' +\
                        f'a problem, consider increasing max_seq_len.')
                    self._warned_target = True
                target = target[:self.max_seq_len -
                                1] + [self.tokenizer.eos_token_id]

            # We might need to truncate the context. Preserve the beginning.
            if len(context) > self.max_seq_len:
                if not self._warned_context:
                    warnings.warn(
                        f'Truncating CONTEXT sequence of length={len(context)} ' +\
                        f'to max_seq_len={self.max_seq_len}. If truncation is ' +\
                        f'a problem, consider increasing max_seq_len.')
                    self._warned_context = True
                context = context[:self.max_seq_len -
                                  1] + [self.tokenizer.eos_token_id]

            # Back into the example
            example['input_ids'] = context
            example['attention_mask'] = [1] * len(context)
            example['labels'] = target

            processed_examples.append(example)

        # Batch examples into a single dict (this also pads)
        batch = self.tokenizer.pad(
            processed_examples,
            padding='max_length',
            max_length=self.max_seq_len,
            return_tensors='pt',
        )
        # We're still missing decoder_input_ids and decoder_attention_mask
        batch['decoder_input_ids'] = torch.cat([
            torch.full((len(processed_examples), 1),
                       self.tokenizer.pad_token_id), batch['labels'][:, :-1]
        ],
                                               dim=1)
        batch['decoder_input_ids'].masked_fill_(
            batch['decoder_input_ids'] == _HF_IGNORE_INDEX,
            self.tokenizer.pad_token_id)
        batch['decoder_attention_mask'] = torch.not_equal(
            batch['labels'], _HF_IGNORE_INDEX)

        # This logic prevents trimming on at least the first batch
        if not (self._allow_pad_trimming and self._seen_first_batch):
            self._seen_first_batch = True
            return batch
        self._seen_first_batch = True

        # The batch is now valid, but we can trim padding for efficiency
        multiple_of = 8
        # (first for the encoder)
        n_non_padding = batch['attention_mask'].sum(dim=1).max()
        keep_tokens = int(multiple_of * torch.ceil(n_non_padding / multiple_of))
        for k in ['input_ids', 'attention_mask']:
            batch[k] = batch[k][:, :keep_tokens].contiguous()
        # (then for the decoder)
        n_non_padding = batch['decoder_attention_mask'].sum(dim=1).max()
        keep_tokens = int(multiple_of * torch.ceil(n_non_padding / multiple_of))
        for k in ['decoder_input_ids', 'decoder_attention_mask', 'labels']:
            batch[k] = batch[k][:, :keep_tokens].contiguous()

        return batch


def ensure_list(x: Union[List, torch.Tensor]) -> List:
    if isinstance(x, torch.Tensor):
        x = list(x.flatten())
    assert isinstance(x, list)
    return x
