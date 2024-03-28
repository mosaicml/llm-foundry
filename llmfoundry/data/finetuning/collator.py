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

TokenizedExample = Dict[str, List[Dict[str, List[int]]]]


def ensure_list(x: Union[List, torch.Tensor]) -> List:
    if isinstance(x, torch.Tensor):
        x = list(x.flatten())
    assert isinstance(x, list)
    return x


def validate_target_settings(target_prompts: str, target_responses: str,
                             decoder_only_format: bool):
    """Raises an error if target settings are invalid."""
    if (not decoder_only_format) and (target_prompts != 'none' or
                                      target_responses != 'last'):
        raise ValueError(
            f'When using encoder_decoder format, you must use target_prompts="none" and target_responses="last".'
        )

    if target_responses not in {'all', 'last'}:
        raise ValueError(
            f'target_responses must be either "last" or "all" but {target_responses=}'
        )

    if target_prompts.startswith('length>='):
        cutoff = target_prompts[8:]
        if not cutoff.isdigit():
            raise ValueError(
                f'target_prompts starts with "length>=" but the rest of the string is not digits ({target_prompts=}). ' +\
                'To use this configuration option, set target_prompts "length>=XX" where "XX" is a positive integer indicating ' +\
                'the length cutoff. Prompts of at least XX tokens in length will be treated as targets.'
            )
        cutoff = int(cutoff)
        if cutoff <= 0:
            raise ValueError(
                f'You are trying to set the target_prompts length cutoff to a negative number {cutoff=}. This is not allowed.'
            )
    elif target_prompts not in {'all', 'none'}:
        raise ValueError(
            f'target_prompts must either be "all", "none" or "length>=XX" where "XX" is a positive integer, but {target_prompts=}'
        )


###### Functions to implement target_prompts and target_responses choices #####
def _sequence_to_labels_all(sequence: list[int],
                            is_last_turn: bool,
                            cutoff: Optional[int] = None) -> list[int]:
    del is_last_turn, cutoff  # unused
    return sequence


def _sequence_to_labels_none(sequence: list[int],
                             is_last_turn: bool,
                             cutoff: Optional[int] = None) -> list[int]:
    del is_last_turn, cutoff  # unused
    return [_HF_IGNORE_INDEX] * len(sequence)


def _sequence_to_labels_last(sequence: list[int],
                             is_last_turn: bool,
                             cutoff: Optional[int] = None) -> list[int]:
    del cutoff  # unused
    if is_last_turn:
        return sequence
    else:
        return [_HF_IGNORE_INDEX] * len(sequence)


def _sequence_to_labels_cutoff(sequence: list[int],
                               is_last_turn: bool,
                               cutoff: Optional[int] = None) -> list[int]:
    del is_last_turn  # unused
    if cutoff is None:
        raise ValueError('input ``cutoff`` must be provided')
    if len(sequence) >= cutoff:
        return sequence
    else:
        return [_HF_IGNORE_INDEX] * len(sequence)


_TARGET_POLICY_LOOKUP = {
    'all': _sequence_to_labels_all,
    'none': _sequence_to_labels_none,
    'last': _sequence_to_labels_last,
    'length': _sequence_to_labels_cutoff,
}


def stitch_turns_decoder_only(
        example_turns: list[dict[str, list[int]]],
        target_prompts: str,
        target_responses: str,
        eos_token_id: Optional[int] = None,
        validate: bool = False) -> tuple[list[int], list[int]]:
    target_prompts = target_prompts.lower()
    target_responses = target_responses.lower()

    if validate:
        validate_target_settings(target_prompts,
                                 target_responses,
                                 decoder_only_format=True)

    if target_prompts.startswith('length'):
        prompt_cutoff = int(target_prompts.split('>=')[-1])
        prompt_to_target = _TARGET_POLICY_LOOKUP['length']
    else:
        prompt_cutoff = None
        prompt_to_target = _TARGET_POLICY_LOOKUP[target_prompts]
    response_to_target = _TARGET_POLICY_LOOKUP[target_responses]

    input_ids = []
    labels = []
    for idx, turn in enumerate(example_turns):
        is_last_turn = idx + 1 == len(example_turns)
        # We assume that no padding has been applied. If there is a pad token,
        # it may be because the pad token is the same as another special token.
        context = ensure_list(turn['input_ids'])
        target = ensure_list(turn['labels'])
        # If an EOS token id is given, ensure that the target sequence ends with it.
        if is_last_turn and eos_token_id is not None:
            if target[-1] != eos_token_id:
                target = target + [eos_token_id]
        # Extend the input_ids
        input_ids += context
        input_ids += target
        # Extend the labels, with values depending on the loss-generating policies
        labels += prompt_to_target(context, is_last_turn, prompt_cutoff)
        labels += response_to_target(target, is_last_turn)

    if len(input_ids) != len(labels):
        raise ValueError(
            f'input_ids and labels should be the same length, {len(input_ids)=}, {len(labels)=}'
        )
    return input_ids, labels


def stitch_turns_encoder_decoder(
    example_turns: list[dict[str, list[int]]],
    eos_token_id: Optional[int] = None,
) -> tuple[list[int], list[int]]:
    context = []
    target = None
    for idx, turn in enumerate(example_turns):
        is_last_turn = idx + 1 == len(example_turns)
        # We assume that no padding has been applied. If there is a pad token,
        # it may be because the pad token is the same as another special token.
        turn_context = ensure_list(turn['input_ids'])
        turn_target = ensure_list(turn['labels'])
        # Context always goes into input_ids.
        context += turn_context
        # Non-terminal turns go into the input_ids. Terminal target is the target.
        if is_last_turn:
            # If an EOS token id is given, ensure that the target sequence ends with it.
            if eos_token_id is not None and turn_target[-1] != eos_token_id:
                turn_target = turn_target + [eos_token_id]
            target = turn_target
        else:
            context += turn_target
    if target is None:
        raise ValueError('target is still None but should be list[int]')
    return context, target


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
        batch_metadata: Optional[Dict[str, Any]] = None,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.decoder_only_format = decoder_only_format
        self.target_responses = target_responses.lower()
        self.target_prompts = target_prompts.lower()
        self.batch_metadata = batch_metadata or {}

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

        validate_target_settings(self.target_prompts, self.target_responses,
                                 self.decoder_only_format)
        if self.target_prompts.startswith('length'):
            self.prompt_cutoff = int(self.target_prompts.split('>=')[-1])
            self.prompt_to_target = _TARGET_POLICY_LOOKUP['length']
        else:
            self.prompt_cutoff = None
            self.prompt_to_target = _TARGET_POLICY_LOOKUP[self.target_prompts]
        self.response_to_target = _TARGET_POLICY_LOOKUP[self.target_responses]

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
            input_ids, labels = stitch_turns_decoder_only(
                example_turns=example['turns'],
                target_prompts=self.target_prompts,
                target_responses=self.target_responses,
                eos_token_id=self.tokenizer.eos_token_id,
            )

            orig_size = len(input_ids)
            # We may need to truncate the input_ids / labels in order to maintain max_seq_len
            if orig_size > self.max_seq_len:
                input_ids = input_ids[:self.max_seq_len]
                labels = labels[:self.max_seq_len]

                # Check to make sure there are still loss-generating tokens. Error if not.
                if len([l for l in labels if l != _HF_IGNORE_INDEX]) == 0:
                    raise ValueError(
                        f'Truncating to max_seq_len={self.max_seq_len} has removed all loss-generating tokens. ' +\
                        f'Pre-truncation sequence length was {orig_size}. ' +\
                        'This sample should have been filtered out before reaching the collator. If using ' +\
                        'pre-tokenized streaming data, this may have resulted from using different ' +\
                        '``target_prompts``, ``target_responses``, or ``max_seq_len`` ' +\
                        'settings when preparing the streaming dataset than what are currently being used.'
                    )

                # Still issue a warning when truncating
                if not self._warned_truncated:
                    warnings.warn(
                        f'Truncating sequence of length={orig_size} to fit max_seq_len={self.max_seq_len}. ' +\
                        f'If truncation is a problem, consider increasing max_seq_len.'
                    )
                    self._warned_truncated = True

            attention_mask = [1] * len(input_ids)

            # Annoyingly, we need to pad everything but input_ids
            # and attention_mask ourselves
            n_total = len(input_ids)
            i_pad = [_HF_IGNORE_INDEX] * (self.max_seq_len - n_total)
            if self.tokenizer.padding_side == 'left':
                labels = i_pad + labels
            else:
                labels = labels + i_pad

            # Update the example
            processed_example = {
                'input_ids': input_ids,
                'labels': labels,
                'attention_mask': attention_mask,
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
            context, target = stitch_turns_encoder_decoder(
                example_turns=example['turns'],
                eos_token_id=self.tokenizer.eos_token_id,
            )

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
