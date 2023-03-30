# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

"""Build a StreamingTextDataset dataset and dataloader for training."""

import os
from itertools import islice
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
import transformers
from omegaconf import DictConfig
from omegaconf import OmegaConf as om
from streaming import StreamingDataset
from torch.utils.data import DataLoader


class StreamingTextDataset(StreamingDataset):
    """Generic text dataset using MosaicML's StreamingDataset.

    Args:
        local (str): Local dataset directory where shards are cached by split.
        tokenizer_name (str): The name of the HuggingFace tokenizer to use to
            tokenize samples.
        max_seq_len (int): The max sequence length of each sample.
        remote (str, optional): Download shards from this remote path or directory. If None, this
            rank and worker's partition of the dataset must all exist locally. Defaults to ``None``.
        split (str, optional): Which dataset split to use, if any. Defaults to ``None``.
        shuffle (bool): Whether to iterate over the samples in randomized order. Defaults to ``False``.
        predownload (int, optional): Target number of samples ahead to download the shards of while
            iterating. Defaults to ``100_000``.
        keep_zip (bool, optional): Whether to keep or delete the compressed file when
            decompressing downloaded shards. If set to None, keep if remote is local. Defaults to
            ``None``.
        download_retry (int): Number of download re-attempts before giving up. Defaults to ``2``.
        download_timeout (float): Number of seconds to wait for a shard to download before raising
            an exception. Defaults to ``60``.
        validate_hash (str, optional): Optional hash or checksum algorithm to use to validate
            shards. Defaults to ``None``.
        shuffle_seed (int): Seed for Deterministic data shuffling. Defaults to ``9176``.
        num_canonical_nodes (int, optional): Canonical number of nodes for shuffling with resumption.
            If ``None``, defaults to the number of nodes of the initial run. Defaults to 128.
        batch_size (int, optional): Batch size of its DataLoader, which affects how the dataset is
            partitioned over the workers. Defaults to ``None``.
    """

    def __init__(self,
                 local: str,
                 tokenizer_name: str,
                 max_seq_len: int,
                 remote: Optional[str] = None,
                 split: Optional[str] = None,
                 shuffle: bool = False,
                 predownload: Optional[int] = 100_000,
                 keep_zip: Optional[bool] = None,
                 download_retry: int = 2,
                 download_timeout: float = 60,
                 validate_hash: Optional[str] = None,
                 shuffle_seed: int = 9176,
                 num_canonical_nodes: Optional[int] = 128,
                 batch_size: Optional[int] = None,
                 **kwargs: Dict[str, Any]):

        group_method = kwargs.pop('group_method', None)
        if group_method is not None:
            raise NotImplementedError(
                'group_method is deprecated and has been removed.\nTo ' +
                'concatenate, use the --concat_tokens ' +
                'argument when creating your MDS dataset with concat_c4.py')

        if kwargs is not None and len(kwargs) > 0:
            raise ValueError(
                f'StreamingTextDataset() got an unexpected keyword argument: {kwargs}'
            )

        if remote is None or (local == remote):
            if os.path.isdir(local):
                contents = set(os.listdir(local))
                if split not in contents:
                    raise ValueError(
                        f'local directory {local} does not contain split {split}'
                    )

        # Build Dataset
        super().__init__(local=local,
                         remote=remote,
                         split=split,
                         shuffle=shuffle,
                         predownload=predownload,
                         keep_zip=keep_zip,
                         download_retry=download_retry,
                         download_timeout=download_timeout,
                         validate_hash=validate_hash,
                         shuffle_seed=shuffle_seed,
                         num_canonical_nodes=num_canonical_nodes,
                         batch_size=batch_size)
        self.max_seq_len = max_seq_len

        # Build tokenizer
        os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            tokenizer_name)

        # suppress warnings when using longer 'max_seq_len'
        self.tokenizer.model_max_length = int(1e30)

    # How to tokenize a text sample to a token sample
    def _tokenize(self, text_sample):
        if self.tokenizer._pad_token is None:
            # Some tokenizers (e.g. GPT2 tokenizer) have no padding token which causes bugs
            raise RuntimeError(
                'If tokenizing on-the-fly, tokenizer must have a pad_token_id')

        return self.tokenizer(text_sample['text'],
                              truncation=True,
                              padding='max_length',
                              max_length=self.max_seq_len)

    def _read_binary_tokenized_sample(self, sample):
        return torch.from_numpy(
            np.frombuffer(sample['tokens'],
                          dtype=np.int64)[:self.max_seq_len].copy())

    # How to process a sample
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = super().__getitem__(idx)
        if 'text' in sample:
            token_sample = self._tokenize(sample)
        elif 'tokens' in sample:
            token_sample = self._read_binary_tokenized_sample(sample)
        else:
            raise RuntimeError(
                'StreamingTextDataset needs samples to have a `text` or `tokens` column'
            )
        return token_sample


class ConcatenatedSequenceCollatorWrapper:
    """Collator wrapper to add sequence_id to batch."""

    def __init__(self,
                 base_collator: Callable,
                 eos_token_id: Optional[int] = None,
                 bos_token_id: Optional[int] = None):
        self.base_collator = base_collator
        if (eos_token_id is None) and (bos_token_id is None):
            raise ValueError(
                'Must supply a value for either eos_token_id or bos_token_id, but got None for both.'
            )
        if (eos_token_id is not None) and (bos_token_id is not None):
            raise ValueError(
                'Cannot use *both* EOS and BOS tokens for detecting sequence boundaries. ' +\
                'Please supply `eos_token_id` if sequences end with an EOS token, or use ' +\
                '`bos_token_id` if sequences start with a BOS token.'
            )
        if eos_token_id is None:
            self.split_token_id = bos_token_id
            self.bos_mode = True
        else:
            self.split_token_id = eos_token_id
            self.bos_mode = False

    def __call__(self, examples: List[Any]) -> Dict[str, torch.Tensor]:
        batch = self.base_collator(examples)
        batch['sequence_id'] = self.get_sequence_id_from_batch(batch)
        return batch

    def get_sequence_id_from_batch(
            self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        is_separator = torch.eq(batch['input_ids'], self.split_token_id)
        cumulative_sep = torch.cumsum(is_separator,
                                      dim=1).to(batch['input_ids'].dtype)
        # If separator token is bos, we're already done
        if self.bos_mode:
            return cumulative_sep

        # If separator token is eos, right shift 1 space
        left_zeros = cumulative_sep.new_zeros((cumulative_sep.shape[0], 1))
        return torch.cat([left_zeros, cumulative_sep[:, :-1]], dim=1)


def build_text_dataloader(cfg: DictConfig, device_batch_size: int):
    assert cfg.name == 'text', f'Tried to build text dataloader with cfg.name={cfg.name}'
    if cfg.dataset.get('group_method', None) is not None:
        raise NotImplementedError(
            'group_method is deprecated and has been removed.\nTo ' +
            'concatenate, use the --concat_tokens ' +
            'argument when creating your MDS dataset with convert_dataset.py')
    dataset = StreamingTextDataset(
        local=cfg.dataset.local,
        tokenizer_name=cfg.dataset.tokenizer_name,
        max_seq_len=cfg.dataset.max_seq_len,
        remote=cfg.dataset.get('remote', None),
        split=cfg.dataset.get('split', None),
        shuffle=cfg.dataset.get('shuffle', False),
        predownload=cfg.dataset.get('predownload', 100_000),
        keep_zip=cfg.dataset.get('keep_zip', None),
        download_retry=cfg.dataset.get('download_retry', 2),
        download_timeout=cfg.dataset.get('download_timeout', 60),
        validate_hash=cfg.dataset.get('validate_hash', None),
        shuffle_seed=cfg.dataset.get('shuffle_seed', 9176),
        num_canonical_nodes=cfg.dataset.get('num_canonical_nodes', 128),
        batch_size=device_batch_size)

    mlm_probability = cfg.dataset.get('mlm_probability', None)
    collate_fn = transformers.DataCollatorForLanguageModeling(
        tokenizer=dataset.tokenizer,
        mlm=mlm_probability is not None,
        mlm_probability=mlm_probability)

    eos_token_id = cfg.dataset.get('eos_token_id')
    bos_token_id = cfg.dataset.get('bos_token_id')
    if (eos_token_id is not None) or (bos_token_id is not None):
        # Note: Will raise an error if both are non-None
        collate_fn = ConcatenatedSequenceCollatorWrapper(
            base_collator=collate_fn,
            eos_token_id=eos_token_id,
            bos_token_id=bos_token_id)

    return DataLoader(
        dataset,
        collate_fn=collate_fn,
        batch_size=device_batch_size,
        drop_last=cfg.drop_last,
        num_workers=cfg.num_workers,
        pin_memory=cfg.get('pin_memory', True),
        prefetch_factor=cfg.get('prefetch_factor', 2),
        persistent_workers=cfg.get('persistent_workers', True),
        timeout=cfg.get('timeout', 0),
    )


# Helpful to test if your dataloader is working locally
# Run `python data.py  --local_path [local] [--remote_path remote, optional]` and verify that batches are printed out
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer',
                        type=str,
                        default='gpt2',
                        help='the name of the tokenizer to use')
    parser.add_argument('--local_path',
                        type=str,
                        required=True,
                        help='the path to the local copy of the dataset')
    parser.add_argument(
        '--remote_path',
        type=str,
        default=None,
        help='the path to the remote copy to stream from (optional)')
    parser.add_argument('--split',
                        type=str,
                        default='val',
                        help='which split of the dataset to use')
    parser.add_argument('--max_seq_len',
                        type=int,
                        default=32,
                        help='max sequence length to test')

    args = parser.parse_args()

    if args.remote_path is not None:
        print(
            f'Reading {args.split} split from {args.local_path} <- streamed from <- {args.remote_path}'
        )
    else:
        print(f'Reading {args.split} split from {args.local_path}')

    cfg = {
        'name': 'text',
        'dataset': {
            'local': args.local_path,
            'remote': args.remote_path,
            'split': args.split,
            'shuffle': False,
            'tokenizer_name': args.tokenizer,
            'max_seq_len': args.max_seq_len,
            'keep_zip': True,  # in case we need compressed files after testing
        },
        'drop_last': False,
        'num_workers': 4,
    }
    cfg = om.create(cfg)
    device_batch_size = 2

    loader = build_text_dataloader(cfg, device_batch_size)
    tokenizer = loader.dataset.tokenizer  # type: ignore
    for batch_ix, batch in enumerate(islice(loader, 5)):
        print('\n')
        print('#' * 20, f'Batch {batch_ix}', '#' * 20)
        for k, v in batch.items():
            print(k, v.shape, v.dtype)
        for sample_ix, token_sample in enumerate(batch['input_ids']):
            print('-' * 20, f' Sample {sample_ix} ', '-' * 20)
            print(tokenizer.decode(token_sample))
