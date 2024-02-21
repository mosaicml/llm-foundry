# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import logging
import warnings
from typing import Any, Dict, List, Optional, Union

import torch
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from llmfoundry.data.finetuning.tasks import TokenizedExample

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
        target_responses (str): For multi-turn examples, this controls which
            responses are treated as training targets (i.e. generate loss).
            Options are:
                "last": (Default) Only the final response is used as the training
                    target; non-terminal responses are only part of the context.
                "all": All of the responses are used as training targets.
        target_prompts (str): This controls which prompts are treated as
            training targets (i.e. generate loss).
            Options are:
                "none": (Default) Prompts are never used as training targets.
                "all": Prompts are always used as training targets.
                "length>=XX": Prompt sequences are used as training targets when
                    they have length of at least XX tokens. For instance,
                    setting "length>=512" instructs the collator to use a prompt
                    sequence as a training target when it is at least 512 tokens long.
        allow_pad_trimming (bool, optional): Whether to allow the collator
            to trim padding, which may result in smaller but inconsistent batch
            sizes. Default: ``False`` ensures that all sequences are max_seq_len.
        separator_text (str | bool, optional): If a string is provided, it will
            be used to separate the context and target sequences (appended to end
            of context). If ``True``, will use the tokenizer's sep_token, which must
            be defined. Only applicable for decoder-only formatting.
        batch_metadata (dict, optional): A dictionary of metadata which will be added
            to the batch.
    """

    def __init__(
        self,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        max_seq_len: int,
        decoder_only_format: bool,
        target_responses: str = 'last',
        target_prompts: str = 'none',
        allow_pad_trimming: bool = False,
        separator_text: Optional[Union[str, bool]] = None,
        format_for_generation: bool = False,
        batch_metadata: Optional[Dict[str, Any]] = None,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.decoder_only_format = decoder_only_format
        self.target_responses = target_responses.lower()
        self.target_prompts = target_prompts.lower()
        self.batch_metadata = batch_metadata or {}

        if format_for_generation:
            raise ValueError(
                'Collator feature `format_for_generation` has been removed.')
        self.format_for_generation = False

        # Trimming will always be skipped on at least the first __call__
        self._allow_pad_trimming = allow_pad_trimming
        self._seen_first_batch = False

        illegal_keys = [
            'input_ids',
            'labels',
            'attention_mask',
            'decoder_input_ids',
            'decoder_attention_mask',
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

        if (max_seq_len % 8) != 0:
            log.warning(
                'For performance, a max_seq_len as a multiple of 8 is recommended.'
            )

        if self.tokenizer.pad_token_id is None:
            raise ValueError(
                f'{self.__class__.__name__} requires that the tokenizer has the pad token set, but it is None'
            )

        if self.target_responses not in {'all', 'last'}:
            raise ValueError(
                f'target_responses must be either "last" or "all" but {self.target_responses=}'
            )

        if self.target_prompts.startswith('length>='):
            thresh = self.target_prompts[8:]
            if not thresh.isdigit() or int(thresh) <= 0:
                raise ValueError(
                    f'target_prompts must either be "all", "none" or "length>=XX" where "XX" is a positive integer, but {self.target_prompts=}'
                )
        elif self.target_prompts not in {'all', 'none'}:
            raise ValueError(
                f'target_prompts must either be "all", "none" or "length>=XX" where "XX" is a positive integer, but {self.target_prompts=}'
            )

        if (not self.decoder_only_format) and (self.target_prompts != 'none' or
                                               self.target_responses != 'last'):
            raise ValueError(
                f'When using encoder_decoder format, you must use target_prompts="none" and target_responses="last".'
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

        self._warned_skipped = False
        self._warned_truncated = False
        self._warned_context = False
        self._warned_target = False

    def __call__(self,
                 examples: List[TokenizedExample]) -> Dict[str, torch.Tensor]:
        for check_key in ['input_ids', 'labels']:
            if check_key not in examples[0]['turns'][0]:
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
            self, examples: List[TokenizedExample]) -> Dict[str, torch.Tensor]:
        # Steps explained in comments
        processed_examples = []
        for example in examples:
            example = example['turns']
            input_ids = []
            labels = []
            for idx, turn in enumerate(example):
                is_last_turn = idx + 1 == len(example)
                # We assume that no padding has been applied. If there is a pad token,
                # it may be because the pad token is the same as another special token.
                context = ensure_list(turn['input_ids'])
                target = ensure_list(turn['labels'])
                # Append any separator tokens to the context tokens
                if self.separator_tokens:
                    context = context + self.separator_tokens
                # Ensure that the target text ends with an eos tag on the last turn
                if is_last_turn and target[-1] != self.tokenizer.eos_token_id:
                    target = target + [self.tokenizer.eos_token_id]
                # Extend the input_ids
                input_ids += context
                input_ids += target
                # Extend the labels, with values depending on the loss-generating policies
                labels += _context_to_labels(context, self.target_prompts)
                labels += _target_to_labels(target, is_last_turn,
                                            self.target_responses)

            if len(input_ids) != len(labels):
                raise ValueError(
                    f'input_ids and labels should be the same length, {len(input_ids)=}, {len(labels)=}'
                )

            orig_size = len(input_ids)
            # We may need to truncate the input_ids / labels in order to maintain max_seq_len
            if orig_size > self.max_seq_len:
                input_ids = input_ids[:self.max_seq_len]
                labels = labels[:self.max_seq_len]

                # Check to make sure there are still loss-generating tokens. Skip if not.
                if len([l for l in labels if l != _HF_IGNORE_INDEX]) == 0:
                    if not self._warned_skipped:
                        warnings.warn(
                            f'Skipping example because truncating to max_seq_len={self.max_seq_len} has ' +\
                            f'removed all loss-generating tokens. Pre-truncation sequence length was {orig_size}.' +\
                            f'If this causes downstream issues because of inconsistent batch sizes, ' +\
                            f'consider increasing max_seq_len or using example packing.'
                        )
                        self._warned_skipped = True
                    continue

                # Still issue a warning when truncating
                if not self._warned_truncated:
                    warnings.warn(
                        f'Truncating sequence of length={orig_size} to fit max_seq_len={self.max_seq_len}. ' +\
                        f'If truncation is a problem, consider increasing max_seq_len.'
                    )
                    self._warned_truncated = True

            attention_mask = [1] * len(input_ids)
            # bidirectional_mask is used by our prefix lm model variants
            # Note: this will be malformed if any loss-generating tokens are followed by non-loss-generating tokens
            # (such as in the case of multi-turn chat examples)
            bidirectional_mask = [
                1 if label == _HF_IGNORE_INDEX else 0 for label in labels
            ]

            # Annoyingly, we need to pad the everything but input_ids
            # and attention_mask ourselves
            n_total = len(input_ids)
            i_pad = [_HF_IGNORE_INDEX] * (self.max_seq_len - n_total)
            z_pad = [0] * (self.max_seq_len - n_total)
            if self.tokenizer.padding_side == 'left':
                labels = i_pad + labels
                bidirectional_mask = z_pad + bidirectional_mask
            else:
                labels = labels + i_pad
                bidirectional_mask = bidirectional_mask + z_pad

            # Update the example
            processed_example = {
                'input_ids': input_ids,
                'labels': labels,
                'attention_mask': attention_mask,
                'bidirectional_mask': bidirectional_mask,
            }

            processed_examples.append(processed_example)

        batch = self.tokenizer.pad(
            processed_examples,
            padding='max_length',
            max_length=self.max_seq_len,
            return_tensors='pt',
        )
        
        batch['sequence_id'] = batch['attention_mask'] - 1

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
            if self.tokenizer.padding_side == 'left':
                batch[k] = v[:, -keep_tokens:].contiguous()
            else:
                batch[k] = v[:, :keep_tokens].contiguous()

        return batch

    def _process_and_batch_encoder_decoder(
            self, examples: List[TokenizedExample]) -> Dict[str, torch.Tensor]:
        # The encoder-decoder case is has some gotchas.
        # Steps are explained in comments.
        processed_examples = []
        for example in examples:
            example = example['turns']
            context = []
            target = None
            for idx, turn in enumerate(example):
                is_last_turn = idx + 1 == len(example)
                # We assume that no padding has been applied. If there is a pad token,
                # it may be because the pad token is the same as another special token.
                turn_context = ensure_list(turn['input_ids'])
                turn_target = ensure_list(turn['labels'])
                # Context always goes into input_ids.
                context += turn_context
                # Non-terminal turns go into the input_ids. Terminal target is the target.
                if is_last_turn:
                    # Ensure that the target text ends with an eos tag on the last turn
                    if turn_target[-1] != self.tokenizer.eos_token_id:
                        turn_target = turn_target + [
                            self.tokenizer.eos_token_id
                        ]
                    target = turn_target
                else:
                    context += turn_target
            assert isinstance(target, list)

            # We need to pad labels ourselves. Because HF.
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
            processed_example = {
                'input_ids': context,
                'labels': target,
                'attention_mask': [1] * len(context),
            }

            processed_examples.append(processed_example)

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


def _context_to_labels(context: list[int], policy: str):
    policy = policy.lower()
    if policy == 'none':
        return [_HF_IGNORE_INDEX] * len(context)
    if policy.startswith('length>='):
        thresh = policy[8:]
        if not thresh.isdigit():
            raise ValueError(
                f'policy must either be "all", "none" or "length>=XX" where "XX" is a positive integer, but {policy=}'
            )
        thresh = int(thresh)
        if thresh <= 0:
            raise ValueError(
                f'policy must either be "all", "none" or "length>=XX" where "XX" is a positive integer, but {policy=}'
            )
    elif policy == 'all':
        thresh = 0
    else:
        raise ValueError(
            f'policy must either be "all", "none" or "length>=XX" where "XX" is a positive integer, but {policy=}'
        )
    if len(context) >= thresh:
        return context
    else:
        return [_HF_IGNORE_INDEX] * len(context)


def _target_to_labels(target: list[int], is_last_turn: bool, policy: str):
    policy = policy.lower()
    if policy == 'last':
        return target if is_last_turn else [_HF_IGNORE_INDEX] * len(target)
    elif policy == 'all':
        return target
    else:
        raise ValueError(f'policy must either be "all", "last", but {policy=}')
