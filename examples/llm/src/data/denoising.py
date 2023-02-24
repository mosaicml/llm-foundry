# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

"""Streaming dataloader for (mixture of) denoising task(s)."""

import logging
import random
import sys
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
import torch
from omegaconf import DictConfig
from omegaconf import OmegaConf as om
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from examples.common.text_data import StreamingTextDataset
from examples.llm.src.models import utils

__all__ = ['MixtureOfDenoisersCollator', 'build_text_denoising_dataloader']

log = logging.getLogger(__name__)

# HuggingFace hardcodes the ignore index to -100
_HF_IGNORE_INDEX = -100

Tokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]

# Required signature of any `prefix_function` (see below)
PREFIX_FUNCTION = Callable[[float, Optional[float], Tokenizer], Sequence[int]]


def ul2_prefix_function(
    mask_ratio: float,
    mean_length: Optional[float],
    tokenizer: Tokenizer,
) -> Sequence[int]:
    """Generates prefixes based on UL2 paper.

    See: http://arxiv.org/abs/2205.05131
    """
    if mean_length is None:
        # This is the case for "sequence to sequence"
        prefix = '[S2S]'
    elif mean_length >= 12 or mask_ratio >= 0.3:
        # UL2 tags this corruption rate "extreme"
        prefix = '[NLG]'
    else:
        # UL2 tags this corruption rate as "regular"
        prefix = '[NLU]'
    return tokenizer(prefix, add_special_tokens=False).input_ids


class MixtureOfDenoisersCollator:
    """Data collator for mixture of span-corruption denoisers, as in UL2.

    This collator supports a variety of tasks used to pre-train an
    encoder-decoder model or a (prefix LM) decoder-only model. This is meant
    to be used with a dataset that yields tokenized text sequences. It is not
    required that the token sequences are already padded or truncate, as this
    collator will internally truncate and pad as needed.

    For the denoising mixture recommended in the original UL2 paper,
    http://arxiv.org/abs/2205.05131, use:
    .. python:
        MixtureOfDenoisersCollator(
            ...,
            span_mean_lengths_and_ratios=[
                [3, .15],
                [8, .15],
                [3, .50],
                [8, .50],
                [64, .15],
                [64, .50],
            ],
            sequence_mask_ratios=0.25
        )

    Args:
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer used to
            prepare the data from raw text. Any missing sentinel tokens will
            be added by the collator.
        max_seq_length (int): The maximum length of sequences produced by this
            collator. Incoming sequences may be truncated to accommodate this
            limit.
            Note that when formatting for decoder-only models, the context
            tokens and target tokens are concatenated, and max_seq_length
            applies to their combined length. For encoder-decoder models, both
            the encoder and decoder will see up to max_seq_length tokens.
        decoder_only_format (bool, optional): Whether to format the batches
            for a decoder-only model (i.e. a prefix LM) or, if ``False``, an
            encoder-decoder model. Default: ``False``.
        span_mean_lengths_and_rations (optional): A length-2 list of a
            ``[mean_length, mask_ratio]`` pair, or a list of such pairs. Each
            pair adds a span corruption denoising task to the task mixture. For
            example, ``[3, 0.15]`` adds the original span corruption task used
            for pre-training a T5 model as in http://arxiv.org/abs/1910.10683,
            which trained with a single span corruption task that used a mean
            span length of 3 and a mask ratio of 15%.
            Default: ``None`` does not add any span corruption tasks.
        sequence_mask_ratios (optional): A float or list of floats, one for each
            sequence corruption denoising task to add to the task mixture. Each
            sequence mask ratio must be greater than 0.0 and less than 0.5.
            This type of task is a special instance of span corruption, with
            exactly one masked span take from the end of the sequence. The
            length of the span is sampled uniformly from
            [1, 2*mask_ratio*n_tokens], where n_tokens is the length of the
            unmasked token sequence.
            Default: ``None` does not add any sequence corruption tasks.
        prefix_function (callable, optional): A function that maps denoising
            task parameters (e.g. mean_length=3, mask_ratio=0.15) to a prefix
            that will be added to sequences when the associated "noiser" is
            applied.
            To disable these prefixes, use a value of ``None``.
            Default: :func:`ul2_prefix_function` applies the prefix scheme
            suggested in the UL2 paper: http://arxiv.org/abs/2205.05131.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        max_seq_length: int,
        decoder_only_format: bool = False,
        span_mean_lengths_and_ratios: Optional[List] = None,
        sequence_mask_ratios: Optional[Union[List[float], float]] = None,
        prefix_function: Optional[PREFIX_FUNCTION] = ul2_prefix_function,
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.decoder_only_format = decoder_only_format
        self._sentinel_token_ids = np.array(self.tokenizer.sentinel_token_ids)

        # Prepare the tokenizer for denoising tasks
        utils.adapt_tokenizer_for_denoising(self.tokenizer)

        # Process the span_mean_lengths_and_ratios argument
        if span_mean_lengths_and_ratios is None:
            # In this case, there are no span corruption tasks
            self.span_mean_lengths_and_ratios = []
        elif isinstance(span_mean_lengths_and_ratios[0], (int, float)):
            # In this case, there is one span corruption task
            if not len(span_mean_lengths_and_ratios) == 2:
                raise ValueError('`span_mean_lengths_and_ratios` must be a ' + \
                                 'pair of [mean_length, mask_ratio], a list ' + \
                                 f'of such pairs, or None. Got {span_mean_lengths_and_ratios}.')
            self.span_mean_lengths_and_ratios = [span_mean_lengths_and_ratios]
        else:
            # In this case, there are one or more span corruption tasks
            span_mean_lengths_and_ratios = list(span_mean_lengths_and_ratios)
            for spec_pair in span_mean_lengths_and_ratios:
                if len(spec_pair) != 2:
                    raise ValueError('`span_mean_lengths_and_ratios` must be a ' + \
                                     'pair of [mean_length, mask_ratio], a list ' + \
                                     f'of such pairs, or None. Got {span_mean_lengths_and_ratios}.')
            self.span_mean_lengths_and_ratios = span_mean_lengths_and_ratios

        # Process the sequence_mask_ratios argument
        if sequence_mask_ratios is None:
            # In this case, there are no sequence corruption tasks
            self.sequence_mask_ratios = []
        elif isinstance(sequence_mask_ratios, float):
            # In this case, there is one sequence corruption task
            self.sequence_mask_ratios = [sequence_mask_ratios]
        else:
            # In this case, there is one or more sequence corruption tasks
            for ratio in sequence_mask_ratios:
                if not (0 < ratio < 0.5):
                    raise ValueError('`sequence_mask_ratios` must be a float (or list '+\
                                    'of floats) between 0.0 and 0.5, or None. '+\
                                    f'Got {sequence_mask_ratios}.')
            self.sequence_mask_ratios = sequence_mask_ratios

        # Populate the noisers so we can learn to denoise them!
        self._noisers = []
        self._smallest_max_raw_length = self.max_seq_length * 100
        self._largest_max_raw_length = 0
        self._uses_span_corruption = False

        # Add "noisers" for any span corruption denoising tasks
        # Each mean_length / mask_ratio combo becomes one of the span
        # corruption denoising tasks
        for span_mean_length, span_mask_ratio in self.span_mean_lengths_and_ratios:
            self._uses_span_corruption = True
            if span_mean_length < 0:
                raise ValueError('All span mean lengths must be positive.')
            if not 0 < span_mask_ratio < 1.0:
                raise ValueError(
                    'All span masking ratios must be between 0.0 and 1.0.')

            if prefix_function is not None:
                prefix_tokens = prefix_function(span_mask_ratio,
                                                span_mean_length,
                                                self.tokenizer)
            else:
                prefix_tokens = None

            max_raw_length = _get_max_starting_length(
                max_length=self.max_seq_length,
                mask_ratio=span_mask_ratio,
                mean_span_length=span_mean_length,
                n_prefix_tokens=len(prefix_tokens or []),
                decoder_only_format=self.decoder_only_format)
            if max_raw_length < self._smallest_max_raw_length:
                self._smallest_max_raw_length = max_raw_length
            if max_raw_length > self._largest_max_raw_length:
                self._largest_max_raw_length = max_raw_length

            kwargs = {
                'mean_span_length': span_mean_length,
                'mask_ratio': span_mask_ratio,
                'prefix_tokens': prefix_tokens,
                'max_raw_length': max_raw_length,
            }
            self._noisers.append(kwargs)

        # Add "noisers" for any sequential denoising tasks
        for sequence_mask_ratio in self.sequence_mask_ratios:
            if prefix_function is not None:
                prefix_tokens = prefix_function(sequence_mask_ratio, None,
                                                self.tokenizer)
            else:
                prefix_tokens = None

            max_raw_length = self.max_seq_length - len(prefix_tokens or []) - 1

            if not self._uses_span_corruption and (
                    max_raw_length < self._smallest_max_raw_length):
                # We choose not to count sequence denoising in the smallest
                # unless there is only sequence denoising.
                self._smallest_max_raw_length = max_raw_length
            if max_raw_length > self._largest_max_raw_length:
                self._largest_max_raw_length = max_raw_length

            kwargs = {
                'mean_span_length': None,
                'mask_ratio': sequence_mask_ratio,
                'prefix_tokens': prefix_tokens,
                'max_raw_length': max_raw_length,
            }
            self._noisers.append(kwargs)

        if not self._noisers:
            raise ValueError(
                'No denoising tasks were included. Make sure to set ' + \
                '`span_mean_lengths_and_ratios` and/or `sequence_mask_ratios`.')

    @property
    def smallest_max_raw_length(self):
        return int(self._smallest_max_raw_length)

    @property
    def largest_max_raw_length(self):
        return int(self._largest_max_raw_length)

    def __call__(self, examples: List[Dict[str,
                                           Any]]) -> Dict[str, torch.Tensor]:
        """Batch examples processed by the span corrupter."""
        processed_examples = []
        for example in examples:
            # Randomly pick a "noiser" to apply to this example
            noiser = random.choice(self._noisers)
            # Apply it
            processed_examples.append(
                noise_token_sequence(
                    example,
                    mask_ratio=noiser['mask_ratio'],
                    mean_span_length=noiser['mean_span_length'],
                    prefix_tokens=noiser['prefix_tokens'],
                    max_raw_length=noiser['max_raw_length'],
                    max_seq_length=self.max_seq_length,
                    tokenizer=self.tokenizer,
                    sentinel_token_ids=self._sentinel_token_ids,
                    decoder_only_format=self.decoder_only_format))
        batch = self.tokenizer.pad(processed_examples)

        # Truncate portions of the inputs that are purely padding
        # (up to a multiple of 8)
        multiple_of = 8
        n_examples_per_length = batch['attention_mask'].sum(0)
        keep_tokens = torch.sum(n_examples_per_length > 0)
        keep_tokens = int(multiple_of * torch.ceil(keep_tokens / multiple_of))

        # Note: EncDec formatting will always produce a right-padded batch
        if self.tokenizer.padding_side == 'left' and self.decoder_only_format:
            batch['input_ids'] = batch['input_ids'][:, -keep_tokens:]
            batch['attention_mask'] = batch['attention_mask'][:, -keep_tokens:]
        else:
            batch['input_ids'] = batch['input_ids'][:, :keep_tokens]
            batch['attention_mask'] = batch['attention_mask'][:, :keep_tokens]

        if self.decoder_only_format:
            if self.tokenizer.padding_side == 'left':
                batch['labels'] = batch['labels'][:, -keep_tokens:]
                batch['bidirectional_mask'] = batch[
                    'bidirectional_mask'][:, -keep_tokens:]
            else:
                batch['labels'] = batch['labels'][:, :keep_tokens]
                batch['bidirectional_mask'] = batch[
                    'bidirectional_mask'][:, :keep_tokens]

        else:
            # Truncate portions of the decoder inputs that are purely padding
            n_examples_per_length = batch['decoder_attention_mask'].sum(0)
            keep_tokens = torch.sum(n_examples_per_length > 0)
            keep_tokens = int(multiple_of *
                              torch.ceil(keep_tokens / multiple_of))

            batch['labels'] = batch['labels'][:, :keep_tokens]
            batch['decoder_attention_mask'] = batch[
                'decoder_attention_mask'][:, :keep_tokens]
            batch['decoder_input_ids'] = batch[
                'decoder_input_ids'][:, :keep_tokens]

        # This slicing can produce non-contiguous tensors, so use .contiguous
        # to prevent related problems
        batch = {k: v.contiguous() for k, v in batch.items()}

        return batch


def build_text_denoising_dataloader(cfg: DictConfig,
                                    device_batch_size: int) -> DataLoader:
    """Constructor function for a Mixture of Denoisers dataloader.

    This function constructs a dataloader that can be used to train an
    encoder-decoder model or a (prefix LM) decoder-only model on a text
    denoising task mixture (e.g. span corruption, or UL2).

    The underlying dataset is a :class:`StreamingTextDataset`, allowing you to
    stream raw text data or pre-tokenized text data.

    The dataloader uses a :class:`MixtureOfDenoisersCollator` to prepare the
    tokenized examples into training batches.

    Args:
        cfg (DictConfig): An omegaconf dictionary used to configure the loader:
            cfg.name (str): The type of dataloader to build. Must = "text_denoising".
            ---
            cfg.dataset.tokenizer_name (str): The name of the tokenizer to use
                for tokenizing raw text. Or, if using pre-tokenized data, the
                name of the tokenizer used to prepare the data.
            cfg.dataset.max_seq_len (int): The maximum length of sequences
                in the batch. See :class:`MixtureOfDenoisersCollator` docstring
                for details.
            See :class:`StreamingTextDataset` for info on other standard config
                options within `cfg.dataset`.
            ---
            cfg.mixture_of_denoisers.decoder_only_format (bool): Whether the
                batches should use the format required for training a decoder-only
                model (if ``True``) or an encoder-decoder model (if ``False``).
            cfg.mixture_of_denoisers.span_mean_lengths_and_ratios (optiona): The
                parameters for any span corruption denoising tasks to include in
                the task mixture.
                See :class:`MixtureOfDenoisersCollator` docstring for details.
            cfg.mixture_of_denoisers.sequence_mask_ratios (optiona): The
                parameters for any sequence denoising tasks to include in the
                task mixture.
                See :class:`MixtureOfDenoisersCollator` docstring for details.
            cfg.mixture_of_denoisers.prefix_function (optiona): Set to ``None``
                to disable the UL2-style prefixes that will be automatically
                added by default.
            ---
            See :class:`DataLoader` for standard argument options to the pytorch
                dataloader, such as `cfg.drop_last`, `cfg.num_workers`, etc.
        device_batch_size (int): The size of the batches (number of examples)
            that the dataloader will produce.
    """
    assert cfg.name == 'text_denoising', f'Tried to build_denoising text dataloader with cfg.name={cfg.name}'

    collate_fn = MixtureOfDenoisersCollator(
        tokenizer=utils.AutoTokenizerForMOD.from_pretrained(
            cfg.dataset.tokenizer_name),
        max_seq_length=cfg.dataset.max_seq_len,
        decoder_only_format=cfg.mixture_of_denoisers.decoder_only_format,
        span_mean_lengths_and_ratios=cfg.mixture_of_denoisers.get(
            'span_mean_lengths_and_ratios'),
        sequence_mask_ratios=cfg.mixture_of_denoisers.get(
            'sequence_mask_ratios'),
        prefix_function=cfg.mixture_of_denoisers.get('prefix_function',
                                                     ul2_prefix_function))

    truncate_to = cfg.mixture_of_denoisers.get('truncate_raw_tokens_to')
    if truncate_to is None:
        # By default, truncate to the largest max raw length of the denoisers
        truncate_to = collate_fn.largest_max_raw_length
    elif isinstance(truncate_to, str):
        if truncate_to.lower() == 'min':
            # Truncate to the smallest max raw length of the denoisers
            truncate_to = collate_fn.smallest_max_raw_length
        elif truncate_to.lower() == 'max':
            # Truncate to the largest max raw length of the denoisers
            truncate_to = collate_fn.largest_max_raw_length
        else:
            raise ValueError(
                f'truncate_raw_tokens_to(="{truncate_to.lower()}") must be "min", "max", a positive int, or None.'
            )
    else:
        if not isinstance(truncate_to, int):
            ValueError(
                f'truncate_raw_tokens_to(={truncate_to}) must be "min", "max", a positive int, or None.'
            )
        if truncate_to < 0:
            ValueError(
                f'truncate_raw_tokens_to(={truncate_to}) must be "min", "max", a positive int, or None.'
            )

    dataset = StreamingTextDataset(
        local=cfg.dataset.local,
        tokenizer_name=cfg.dataset.tokenizer_name,
        max_seq_len=truncate_to,
        remote=cfg.dataset.get('remote'),
        split=cfg.dataset.get('split'),
        shuffle=cfg.dataset.get('shuffle', False),
        predownload=cfg.dataset.get('predownload', 100_000),
        keep_zip=cfg.dataset.get('keep_zip', False),
        download_retry=cfg.dataset.get('download_retry', 2),
        download_timeout=cfg.dataset.get('download_timeout', 60),
        validate_hash=cfg.dataset.get('validate_hash'),
        shuffle_seed=cfg.dataset.get('shuffle_seed'),
        num_canonical_nodes=cfg.dataset.get('num_canonical_nodes'),
        batch_size=device_batch_size)

    return DataLoader(
        dataset,
        collate_fn=collate_fn,
        batch_size=device_batch_size,
        drop_last=cfg.drop_last,
        num_workers=cfg.num_workers,
        pin_memory=cfg.get('pin_memory', True),
        prefetch_factor=cfg.get('prefetch_factor', 2),
        persistent_workers=cfg.get('persistent_workers', False),
        timeout=cfg.get('timeout', 0),
    )


def noise_token_sequence(
    example: Union[torch.Tensor, Mapping[str, Any]],
    mask_ratio: float,
    mean_span_length: Optional[float],
    prefix_tokens: Optional[Sequence[int]],
    max_raw_length: int,
    max_seq_length: int,
    tokenizer: Tokenizer,
    sentinel_token_ids: np.ndarray,
    decoder_only_format: bool,
) -> Dict[str, torch.Tensor]:
    """Span corruption applicable to all UL2 denoising tasks."""
    # Extract the raw text tokens (trim if we need to)
    if isinstance(example, torch.Tensor):
        # If the example is a tensor, assume is the raw tokens with no padding
        tokens = example
        length = len(tokens)
    else:
        tokens = example['input_ids']
        length = sum(example['attention_mask'])
    if length > max_raw_length:
        length = max_raw_length
    if tokenizer.padding_side == 'left':
        tokens = tokens[-length:]
    else:
        tokens = tokens[:length]

    prefix_tokens = prefix_tokens or []

    # mean_span_length==None is a special case for "sequential" denoising
    # (where a single span at the end of the sequence is masked)
    if mean_span_length is None:
        # This ensures that exactly 1 span will be produced and that
        # trimming to max_seq_length won't cut off any <EOS> token.
        # In the decoder-only case, this won't insert new tokens.
        min_span_length = np.maximum(
            1, length + len(prefix_tokens) - max_seq_length)
        max_span_length = np.maximum(
            min_span_length, np.minimum(length - 1, 2 * mask_ratio * length))
        mean_span_length = float(
            np.floor(
                np.random.uniform(low=min_span_length, high=max_span_length)))
        mask_ratio = mean_span_length / length
        use_sentinels = False
    else:
        use_sentinels = True
    ensure_input_eos = False if decoder_only_format else True

    # Generate the mask
    # Note: this function can be used for all the UL2 noising functions
    mask = _sample_mask_array(length, mask_ratio, mean_span_length)
    # The sequence should always be unmasked at the beginning
    assert mask[0] == 0

    # Generate the input/label sequences given the raw tokens and the mask
    tokens_inputs = _apply_mask(tokens,
                                mask,
                                use_sentinels,
                                tokenizer.eos_token_id,
                                sentinel_token_ids,
                                ensure_eos=ensure_input_eos)
    tokens_labels = _apply_mask(tokens,
                                1 - mask,
                                use_sentinels,
                                tokenizer.eos_token_id,
                                sentinel_token_ids,
                                ensure_eos=True)

    # Tag the inputs with any prefix
    if prefix_tokens:
        tokens_inputs = np.concatenate([prefix_tokens, tokens_inputs])

    # Trim if necessary
    if len(tokens_inputs) > max_seq_length:
        raise ValueError('This should not exceed the max length')
    if len(tokens_labels) > max_seq_length:
        raise ValueError('This should not exceed the max length')

    tokens_inputs = torch.LongTensor(tokens_inputs)
    tokens_labels = torch.LongTensor(tokens_labels)

    if decoder_only_format:
        return _format_tokens_for_decoder_only(tokens_inputs, tokens_labels,
                                               max_seq_length,
                                               tokenizer.pad_token_id,
                                               tokenizer.padding_side)
    return _format_tokens_for_encoder_decoder(tokens_inputs, tokens_labels,
                                              max_seq_length,
                                              tokenizer.pad_token_id)


def _get_max_starting_length(max_length: int, mask_ratio: float,
                             mean_span_length: float, n_prefix_tokens: int,
                             decoder_only_format: bool):
    """Get max num raw tokens that will fit max_length."""

    def sequence_stats(length: int):
        length = np.maximum(length, 2)
        num_noise_tokens = int(np.round(mask_ratio * float(length)))
        num_noise_tokens = np.minimum(np.maximum(num_noise_tokens, 1),
                                      length - 1)
        num_spans = int(np.round(float(num_noise_tokens) / mean_span_length))
        num_noise_spans = np.maximum(num_spans, 1)
        num_nonnoise_tokens = length - num_noise_tokens
        # Prefix, sentinel, and EOS added to input for Enc-Dec
        extra_inp_tokens = n_prefix_tokens + num_noise_spans + int(
            not decoder_only_format)
        # Sentinel and EOS added to target
        extra_targ_tokens = num_noise_spans + 1
        # Sequence totals after corruption
        total_inp_tokens = num_nonnoise_tokens + extra_inp_tokens
        total_targ_tokens = num_noise_tokens + extra_targ_tokens
        return total_inp_tokens, total_targ_tokens

    def length_fits(length: int) -> bool:
        total_inp_tokens, total_targ_tokens = sequence_stats(length)
        if decoder_only_format:
            return (total_inp_tokens + total_targ_tokens) <= max_length
        return (total_inp_tokens <= max_length) and (total_targ_tokens <=
                                                     max_length)

    # Start with a definitely too-long sequence and reduce until it fits
    num_raw_tokens = max_length * 2
    while num_raw_tokens > 0:
        if length_fits(num_raw_tokens):
            return num_raw_tokens
        num_raw_tokens -= 1
    raise ValueError(
        'Unable to find a starting sequence length that can fit given the corruption and max_length parameters.'
    )


def _sample_mask_array(length: int, mask_ratio: float,
                       mean_span_length: float) -> np.ndarray:
    """Samples a span corruption mask."""
    if mask_ratio == 0.0:
        return np.zeros(length)
    # This first block computes the number of noise/non-noise spans and the
    # total tokens in each. Extra steps are taken to handle edge cases that
    # cause degeneracy.
    starting_length = length
    length = np.maximum(length, 2)
    num_noise_tokens = int(np.round(mask_ratio * float(length)))
    num_noise_tokens = np.minimum(np.maximum(num_noise_tokens, 1), length - 1)
    num_spans = int(np.round(float(num_noise_tokens) / mean_span_length))
    num_noise_spans = np.maximum(num_spans, 1)
    num_nonnoise_tokens = length - num_noise_tokens

    # Sample the noise/non-noise span lengths and interleave them to
    # generate the mask array.
    # Note: We always start with a non-noise span.
    def _sample_span_lengths(total_tokens: int, num_spans: int) -> np.ndarray:
        """Samples lengths of num_spans segments.

        Note: the combined length of segments equals total_tokens.
        """
        span_markers = np.less(np.arange(total_tokens - 1), num_spans -
                               1)[np.random.permutation(total_tokens - 1)]
        span_start_indicator = np.concatenate([[0], span_markers])
        span_id = np.cumsum(span_start_indicator).reshape(-1, 1)
        spans = np.arange(num_spans).reshape(1, -1)
        span_lengths = np.sum(span_id == spans, axis=0)
        return span_lengths

    noise_span_lengths = _sample_span_lengths(num_noise_tokens, num_noise_spans)
    nonnoise_span_lengths = _sample_span_lengths(num_nonnoise_tokens,
                                                 num_noise_spans)
    interleaved_span_lengths = np.reshape(
        np.stack([nonnoise_span_lengths, noise_span_lengths], axis=1),
        [num_noise_spans * 2])

    span_starts = np.cumsum(interleaved_span_lengths)[:-1]
    span_start_indicator = np.zeros(length)
    span_start_indicator[span_starts] = 1
    span_id = np.cumsum(span_start_indicator)
    is_noise = np.equal(np.mod(span_id, 2), 1)

    mask = is_noise[:starting_length]

    return mask


def _apply_mask(tokens: Union[torch.Tensor, Sequence[int], np.ndarray],
                mask: np.ndarray,
                use_sentinels: bool,
                eos_token_id: int,
                sentinel_token_ids: np.ndarray,
                ensure_eos: bool = True) -> np.ndarray:
    """Remove or replace masked portions from token sequence."""
    if not use_sentinels:
        # The logic is simple if we don't use sentinel tokens
        noised_tokens = np.array(tokens)[np.logical_not(mask)]

        # Ensure there's an end-of-sentence token at the end
        if ensure_eos and (noised_tokens[-1] != eos_token_id):
            noised_tokens = np.concatenate([noised_tokens, [eos_token_id]])

        return noised_tokens

    # Masking at previous token
    prev_token_mask = np.concatenate([[0], mask[:-1]])

    # Decompose mask into start-of-span mask and non-start-of-span mask
    start_of_noise_span_token = np.logical_and(mask,
                                               np.logical_not(prev_token_mask))
    nonstart_noise_span_token = np.logical_and(mask, prev_token_mask)

    # Replace tokens at the start of each noise span with its corresponding
    # sentinel token
    sentinel_idx = np.minimum(len(sentinel_token_ids),
                              np.cumsum(start_of_noise_span_token)) - 1
    tokens = np.where(start_of_noise_span_token,
                      sentinel_token_ids[sentinel_idx], tokens)

    # Remove masked tokens (but preserving the sentinel tokens)
    noised_tokens = tokens[np.logical_not(nonstart_noise_span_token)]

    # Ensure there's an end-of-sentence token at the end
    if ensure_eos and (noised_tokens[-1] != eos_token_id):
        noised_tokens = np.concatenate([noised_tokens, [eos_token_id]])
    return noised_tokens


def _format_tokens_for_encoder_decoder(
    tokens_inputs: torch.LongTensor,
    tokens_labels: torch.LongTensor,
    max_seq_length: int,
    pad_token_id: int,
) -> Dict[str, torch.Tensor]:
    """Package the input/label sequence for an EncDec model."""
    example = {}
    # Re-populate with an empty, padded example
    example['input_ids'] = torch.full((max_seq_length,),
                                      pad_token_id,
                                      dtype=torch.int32)
    example['labels'] = torch.full((max_seq_length,),
                                   _HF_IGNORE_INDEX,
                                   dtype=torch.int32)
    example['attention_mask'] = torch.zeros_like(example['input_ids'])
    example['decoder_attention_mask'] = torch.zeros_like(example['labels'])

    # Fill in with processed results (Note: EncDec format is right-padded)
    example['input_ids'][:len(tokens_inputs)] = tokens_inputs
    example['labels'][:len(tokens_labels)] = tokens_labels
    example['attention_mask'][:len(tokens_inputs)] = 1
    example['decoder_attention_mask'][:len(tokens_labels)] = 1

    # Best practice is to include decoder_input_ids (= right-shifted labels)
    example['decoder_input_ids'] = torch.full_like(example['labels'],
                                                   pad_token_id)
    example['decoder_input_ids'][1:len(tokens_labels)] = tokens_labels[:-1]
    return example


def _format_tokens_for_decoder_only(
    tokens_inputs: torch.LongTensor,
    tokens_labels: torch.LongTensor,
    max_seq_length: int,
    pad_token_id: int,
    padding_side: str,
) -> Dict[str, torch.Tensor]:
    """Package the input/label sequence for an decoder-only model."""
    example = {}
    # Re-populate with an empty, padded example
    example['input_ids'] = torch.full((max_seq_length,),
                                      pad_token_id,
                                      dtype=torch.int32)
    example['labels'] = torch.full((max_seq_length,),
                                   _HF_IGNORE_INDEX,
                                   dtype=torch.int32)
    example['attention_mask'] = torch.full((max_seq_length,),
                                           0,
                                           dtype=torch.bool)
    example['bidirectional_mask'] = torch.full((max_seq_length,),
                                               0,
                                               dtype=torch.bool)

    n_input = len(tokens_inputs)
    n_label = len(tokens_labels)
    n_concat = n_input + n_label
    assert n_concat <= max_seq_length

    tokens_concat = torch.concat([tokens_inputs, tokens_labels], dim=0)

    # Fill in with the processed results
    if padding_side == 'left':
        example['input_ids'][-n_concat:] = tokens_concat
        # `labels` copies `input_ids` but with -100 at
        # non-loss-generating tokens. `labels` will be shifted in the
        # model code when computing loss.
        example['labels'][-n_concat:] = tokens_concat
        example['labels'][-n_concat:-n_label] = _HF_IGNORE_INDEX
        example['attention_mask'][-n_concat:] = 1
        example['bidirectional_mask'][-n_concat:-n_label] = 1
    else:
        example['input_ids'][:n_concat] = tokens_concat
        # See above comment regarding `labels`
        example['labels'][:n_concat] = tokens_concat
        example['labels'][:n_input] = _HF_IGNORE_INDEX
        example['attention_mask'][:n_concat] = 1
        example['bidirectional_mask'][:n_input] = 1
    return example


# Helpful to test if your dataloader is working locally
# Run `python data_denoising.py [remote] [local, optional]` and verify that batches
# are printed out
if __name__ == '__main__':
    remote = sys.argv[1]
    if len(sys.argv) > 2:
        local = sys.argv[2]
    else:
        local = remote
    print(f'Reading val split from {remote} -> {local}')

    decoder_only = True

    cfg = {
        'name': 'text_denoising',
        'dataset': {
            'local': local,
            'remote': remote,
            'split': 'val_small',
            'shuffle': False,
            'tokenizer_name': 'gpt2' if decoder_only else 't5-base',
            'max_seq_len': 256 if decoder_only else 128,
            'predownload': 1000,
            'keep_zip': True,  # in case we need compressed files after testing
        },
        'mixture_of_denoisers': {
            'decoder_only_format': decoder_only,
            'span_mean_lengths_and_ratios': [[3, .15], [8, .5]],
            'sequence_mask_ratios': 0.25,
        },
        'drop_last': False,
        'num_workers': 0,
    }
    cfg = om.create(cfg)
    device_batch_size = 2

    loader = build_text_denoising_dataloader(cfg, device_batch_size)

    print(
        f'\n\nTRUNCATING TO: {loader.dataset.max_seq_len}\n\n')  # type: ignore

    tokenizer = loader.collate_fn.tokenizer
    batch_ix = 0
    for batch in loader:
        print('\n')
        print('#' * 20, f'Batch {batch_ix}', '#' * 20)
        for k, v in batch.items():
            print(k, v.shape, v.dtype)
        for sample_ix, token_sample in enumerate(batch['input_ids']):
            if cfg.mixture_of_denoisers.decoder_only_format:
                labels = batch['labels'][sample_ix]
                attn_inputs = batch['bidirectional_mask'][sample_ix].to(
                    torch.bool)
                attn_full = batch['attention_mask'][sample_ix].to(torch.bool)
                attn_labels = torch.logical_xor(attn_inputs, attn_full)
                print('-' * 20, f' Sample {sample_ix} ', '-' * 20)
                print('Input:  ', tokenizer.decode(token_sample[attn_inputs]))
                print('Target: ', tokenizer.decode(labels[attn_labels]))
            else:
                labels = batch['labels'][sample_ix]
                attn_inputs = batch['attention_mask'][sample_ix].to(torch.bool)
                attn_labels = batch['decoder_attention_mask'][sample_ix].to(
                    torch.bool)
                print('-' * 20, f' Sample {sample_ix} ', '-' * 20)
                print('Input:  ', tokenizer.decode(token_sample[attn_inputs]))
                print('Target: ', tokenizer.decode(labels[attn_labels]))
        batch_ix += 1
        if batch_ix >= 5:
            break
