# Copyright 2022 MosaicML Benchmarks authors
# SPDX-License-Identifier: Apache-2.0

"""Build a StreamingC4 dataset and dataloader for training."""

import os
import sys
from itertools import islice
from typing import Any, Dict, Iterator, Optional

import transformers
from omegaconf import DictConfig
from omegaconf import OmegaConf as om
from streaming import Dataset
from torch.utils.data import DataLoader


class StreamingC4(Dataset):
    """Colossal Cleaned Common Crawl dataset using streaming datasets V2.

    Args:
        remote (str): Remote directory (S3 or local filesystem) where dataset is stored.
        local (str): Local filesystem directory where dataset is cached during operation.
        split (str): The dataset split to use, either 'train' or 'val'.
        shuffle (bool): Whether to shuffle the samples in this dataset.
        prefetch (int): Target number of samples remaining to prefetch while iterating.
        tokenizer_name (str): The name of the HuggingFace tokenizer to use to tokenize samples.
        max_seq_len (int): The max sequence length of each token sample.
        group_method (str): How to group text samples into token samples. Supports 'truncate' or 'concat'.
        retry (int): Number of download re-attempts before giving up. Default: 2.
        timeout (float): How long to wait for shard to download before raising an exception. Default: 120 sec.
        batch_size (Optional[int]): Hint batch_size that will be used on each device's DataLoader. Default: ``None``.
    """

    def __init__(self,
                 remote: str,
                 local: str,
                 split: str,
                 shuffle: bool,
                 prefetch: int,
                 tokenizer_name: str,
                 max_seq_len: int,
                 group_method: str = 'truncate',
                 retry: int = 2,
                 timeout: float = 120,
                 batch_size: Optional[int] = None):
        # Validation
        if split not in ['train', 'val']:
            raise ValueError(
                f"split='{split}' must be one of ['train', 'val'].")
        if group_method not in ['truncate', 'concat']:
            raise ValueError(
                f"group_method='{group_method}' must be one of ['truncate', 'concat']."
            )

        # Build Dataset
        super().__init__(remote=remote,
                         local=local,
                         split=split,
                         shuffle=shuffle,
                         prefetch=prefetch,
                         keep_zip=False,
                         retry=retry,
                         timeout=timeout,
                         hash=None,
                         batch_size=batch_size)
        self.tokenizer_name = tokenizer_name
        self.max_seq_len = max_seq_len
        self.group_method = group_method

        # Build tokenizer
        os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.tokenizer_name)
        if self.tokenizer.pad_token is None:
            # Some tokenizers (e.g. GPT2 tokenizer) have no padding token which causes bugs
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # suppress warnings when using group_method='concat' and no truncation
        self.tokenizer.model_max_length = int(1e30)

    # How to tokenize a text sample to a token sample
    def _tokenize(self, text_sample):
        if self.group_method == 'truncate':
            truncation = True
            padding = 'max_length'
            max_length = self.max_seq_len
        elif self.group_method == 'concat':
            truncation = False
            padding = False
            max_length = None
        else:
            raise ValueError(f"Got unknown group_method='{self.group_method}'.")
        return self.tokenizer(text_sample['text'],
                              truncation=truncation,
                              padding=padding,
                              max_length=max_length)

    # How to process a sample
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        text_sample = super().__getitem__(idx)
        token_sample = self._tokenize(text_sample)
        return token_sample

    # Define iterable over samples
    # Usually this can be left alone and inherited directly from super() class StreamingDataset, but concatenating samples is custom behavior.
    # If group_method=='truncate', we simply return the token sample.
    # If group_method=='concat', then we keep fetching token samples until we fill up max_seq_len.
    def __iter__(self) -> Iterator[Any]:
        if self.group_method == 'truncate':
            iterator = super().__iter__()
            yield from iterator

        elif self.group_method == 'concat':
            buffer = {}
            while True:
                iterator = super().__iter__()
                for sample in iterator:

                    for k, v in sample.items():
                        buffer[k] = buffer.get(k, []) + v
                    while len(buffer['input_ids']) >= self.max_seq_len:
                        concat_sample = {}
                        for k, v in buffer.items():
                            concat_sample[k] = v[:self.max_seq_len]
                            buffer[k] = v[self.max_seq_len:]
                        yield concat_sample
        else:
            raise ValueError(f"Got unknown group_method='{self.group_method}'.")

    # Define length
    # Usually this can be left alone and inherited directly from superclass
    # Dataset, but concatenating samples is custom behavior.
    # If group_method=='truncate', we simply return the # samples.
    # If group_method=='concat', we repeat forever and have no defined length.
    def __len__(self) -> Optional[int]:
        if self.group_method == 'truncate':
            return super().__len__()
        elif self.group_method == 'concat':
            return None
        else:
            raise ValueError(f"Got unknown group_method='{self.group_method}'.")


def build_c4_dataloader(cfg: DictConfig, device_batch_size: int):
    assert cfg.name == 'c4', f'Tried to build c4 dataloader with cfg.name={cfg.name}'
    dataset = StreamingC4(split=cfg.dataset.split,
                          remote=cfg.dataset.remote,
                          local=cfg.dataset.local,
                          shuffle=cfg.dataset.shuffle,
                          prefetch=cfg.dataset.prefetch,
                          tokenizer_name=cfg.dataset.tokenizer_name,
                          max_seq_len=cfg.dataset.max_seq_len,
                          group_method=cfg.dataset.group_method,
                          batch_size=device_batch_size)

    collate_fn = transformers.DataCollatorForLanguageModeling(
        tokenizer=dataset.tokenizer,
        mlm=True,
        mlm_probability=cfg.dataset.mlm_probability)

    return DataLoader(
        dataset,
        collate_fn=collate_fn,
        batch_size=device_batch_size,
        drop_last=cfg.drop_last,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        prefetch_factor=cfg.prefetch_factor,
        persistent_workers=cfg.persistent_workers,
        timeout=cfg.timeout,
    )


# Helpful to test if your dataloader is working locally
# Run `python data.py [remote] [local, optional]` and verify that batches are printed out
if __name__ == '__main__':
    remote = sys.argv[1]
    if len(sys.argv) > 2:
        local = sys.argv[2]
    else:
        local = remote
    print(f'Reading val split from {remote} -> {local}')

    cfg = {
        'name': 'c4',
        'dataset': {
            'remote': remote,
            'local': local,
            'split': 'val',
            'shuffle': True,
            'prefetch': 1000,
            'tokenizer_name': 'bert-base-uncased',
            'max_seq_len': 32,
            'group_method': 'truncate',
            'mlm_probability': 0.15,
        },
        'drop_last': False,
        'num_workers': 4,
        'pin_memory': True,
        'prefetch_factor': 2,
        'persistent_workers': True,
        'timeout': 30,
    }
    cfg = om.create(cfg)
    device_batch_size = 2

    loader = build_c4_dataloader(cfg, device_batch_size)
    tokenizer = loader.dataset.tokenizer  # type: ignore
    for batch_ix, batch in enumerate(islice(loader, 5)):
        print('\n')
        print('#' * 20, f'Batch {batch_ix}', '#' * 20)
        for k, v in batch.items():
            print(k, v.shape, v.dtype)
        for sample_ix, token_sample in enumerate(batch['input_ids']):
            print('-' * 20, f' Sample {sample_ix} ', '-' * 20)
            print(tokenizer.decode(token_sample))
