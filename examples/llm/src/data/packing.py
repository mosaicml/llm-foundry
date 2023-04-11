# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

from typing import Callable, Dict, List, Literal, Optional, Tuple

import torch


class BinPackWrapper:
    """Utility collator for packing to reduce padding."""

    def __init__(self,
                 collator: Callable,
                 target_batch_size: int,
                 max_seq_len: int,
                 pad_token_id: int,
                 padding_side: Literal['left', 'right'],
                 max_leftover_bins_to_keep: Optional[int] = None):
        self.base_collator = collator
        self.out_size = int(target_batch_size)
        self.max_seq_len = int(max_seq_len)
        self.pad_token_id = int(pad_token_id)
        self.padding_side = padding_side

        if self.out_size <= 0:
            raise ValueError(f'{target_batch_size=} must be >0.')
        if self.max_seq_len <= 0:
            raise ValueError(f'{max_seq_len=} must be >0.')
        if self.pad_token_id < 0:
            raise ValueError(f'{pad_token_id=} must be >=0.')

        if max_leftover_bins_to_keep is None:
            self.max_leftover_bins_to_keep = int(10 * self.out_size)
        elif max_leftover_bins_to_keep < 0:
            raise ValueError(
                f'{max_leftover_bins_to_keep=} must be >=0 or None.')
        else:
            self.max_leftover_bins_to_keep = int(max_leftover_bins_to_keep)

        self.n_packed_tokens = 0
        self.n_total_tokens = 0
        self.n_packed_examples = 0

        self._leftover_bins: List[Tuple[int, Dict[str, torch.Tensor]]] = []

    @property
    def waste(self):
        return 1 - (self.n_packed_tokens / self.n_total_tokens)

    @property
    def efficiency(self):
        return self.n_packed_tokens / (self.max_seq_len *
                                       self.n_packed_examples)

    def __call__(
            self,
            examples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        batch = self.base_collator(examples)

        assert 'attention_mask' in batch
        assert 'input_ids' in batch

        for key in batch.keys():
            assert key in [
                'input_ids',
                'labels',
                'attention_mask',
                'bidirectional_mask',
            ]

        # Cut everything down to size
        sizes, trimmed_examples = [], []
        for idx in range(batch['attention_mask'].shape[0]):
            size, trimmed_example = extract_trim_batch_idx(batch, idx)
            sizes.append(size)
            trimmed_examples.append(trimmed_example)

        # Apply our CS 101 bin packing algorithm.
        packed_examples, n_packed_tokens, n_total_tokens, leftover_bins = first_fit_bin_packing(
            sizes=sizes,
            examples=trimmed_examples,
            num_bins=self.out_size,
            max_bin_size=self.max_seq_len,
            existing_bins=self._leftover_bins,
        )
        self.n_packed_tokens += n_packed_tokens
        self.n_total_tokens += n_total_tokens
        self.n_packed_examples += self.out_size
        self._leftover_bins = leftover_bins[:self.max_leftover_bins_to_keep]

        # Re-pad to max_seq_len and batch
        batch = repad(packed_examples,
                      max_seq_len=self.max_seq_len,
                      pad_token_id=self.pad_token_id,
                      padding_side=self.padding_side)
        return batch


def extract_trim_batch_idx(batch: Dict[str, torch.Tensor], idx: int):
    example = {k: v[idx] for k, v in batch.items()}

    keep = example['attention_mask'] == 1
    size = int(keep.sum())
    trim_example = {k: v[keep] for k, v in example.items()}
    trim_example['sequence_id'] = torch.zeros_like(trim_example['input_ids'])

    return size, trim_example


def combine_in_place(example: Dict[str, torch.Tensor],
                     add_on: Dict[str, torch.Tensor]):
    if 'labels' in add_on:
        # Prevents the last token in example from being trained to
        # predict the first token in add_on, which would make no sense.
        add_on['labels'][0] = -100

    for k in example.keys():
        if k == 'sequence_id':
            example[k] = torch.cat(
                [example[k], add_on[k] + 1 + torch.max(example[k])])
        else:
            example[k] = torch.cat([example[k], add_on[k]])
    return example


def first_fit_bin_packing(
    sizes: List[int], examples: List[Dict[str, torch.Tensor]], num_bins: int,
    max_bin_size: int, existing_bins: List[Tuple[int, Dict[str, torch.Tensor]]]
) -> Tuple[List[Dict[str, torch.Tensor]], int, int, List[Tuple[int, Dict[
        str, torch.Tensor]]]]:

    # Will contain tuples (bin_size_size, packed_example)
    bins: List[Tuple[int, Dict[str, torch.Tensor]]] = existing_bins

    starting_total_bin_sizes = sum([bin_size for bin_size, _ in bins])

    required_num_examples = max(0, num_bins - len(bins))
    num_examples = len(sizes)
    if num_examples < required_num_examples:
        raise ValueError(
            f'Cannot pack {num_examples} examples into {num_bins} bins, starting from {len(bins)} existing bins.'
        )

    sizes_and_examples = [
        (size, example) for size, example in zip(sizes, examples)
    ]
    sorted_sizes_and_examples = sorted(sizes_and_examples,
                                       key=lambda x: x[0],
                                       reverse=True)

    # Go through each item from longest to shortest.
    # Note: all items will either go into an existing or new bin.
    for i, (size, example) in enumerate(sorted_sizes_and_examples):
        # If we can't keep packing, all remaining items get their own bin.
        required_num_examples = max(0, num_bins - len(bins))
        n_remaining = num_examples - i
        assert n_remaining >= required_num_examples
        if n_remaining == required_num_examples:
            # Can't keep packing. All remaining items get their own bin.
            bins.append((size, example))
            continue

        # Add it to the first bin it fits in
        added = False
        for bidx in range(len(bins)):
            if bins[bidx][0] + size <= max_bin_size:
                bin_size, packed_example = bins.pop(bidx)
                bin_size = bin_size + size
                packed_example = combine_in_place(packed_example, example)
                bins.append((bin_size, packed_example))
                added = True
                break
        # If it didn't fit anywhere, open a new bin
        if not added:
            bins.append((size, example))

    total_bin_sizes = sum([bin_size for bin_size, _ in bins])
    total_new_bin_sizes = total_bin_sizes - starting_total_bin_sizes
    total_example_sizes = sum(sizes)
    if total_new_bin_sizes != total_example_sizes:
        raise AssertionError(
            f'Error in packing. {total_example_sizes=} does not equal {total_new_bin_sizes=}.'
        )

    sorted_bins = sorted(bins, key=lambda x: x[0], reverse=True)
    bin_sizes, packed_examples = [], []
    for bin_size, packed_example in sorted_bins:
        bin_sizes.append(bin_size)
        packed_examples.append(packed_example)

    # Return:
    #  - the num_bins largest packed examples
    #  - the total tokens in those examples
    #  - the total size of all new examples
    #  - leftover bins
    return packed_examples[:num_bins], sum(
        bin_sizes[:num_bins]), sum(sizes), sorted_bins[num_bins:]


def repad(packed_examples: List[Dict[str, torch.Tensor]], max_seq_len: int,
          pad_token_id: int, padding_side: str) -> Dict[str, torch.Tensor]:

    def pad_tensor(tensor: torch.Tensor, pad_value: int):
        if len(tensor) == max_seq_len:
            return tensor
        t = torch.full((max_seq_len,),
                       pad_value,
                       dtype=tensor.dtype,
                       device=tensor.device)
        if padding_side == 'left':
            t[-len(tensor):] = tensor
        elif padding_side == 'right':
            t[:len(tensor)] = tensor
        else:
            raise ValueError(f'Unknown {padding_side=}')
        return t

    pad_vals = {
        'input_ids': pad_token_id,
        'labels': -100,
        'attention_mask': 0,
        'bidirectional_mask': 0,
        'sequence_id': -1,
    }
    keys = packed_examples[0].keys()
    batch = {}
    for key in keys:
        batch[key] = torch.stack([
            pad_tensor(example[key], pad_vals[key])
            for example in packed_examples
        ])
    return batch
