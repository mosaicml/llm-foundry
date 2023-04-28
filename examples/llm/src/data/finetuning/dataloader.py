# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Union

import torch
from composer.utils import dist
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from examples.llm.src.data.finetuning.collator import Seq2SeqFinetuningCollator
from examples.llm.src.data.finetuning.tasks import dataset_constructor
from examples.llm.src.data.packing import BinPackWrapper

log = logging.getLogger(__name__)

# HuggingFace hardcodes the ignore index to -100
_HF_IGNORE_INDEX = -100

Tokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]


def build_finetuning_dataloader(cfg: DictConfig, tokenizer: Tokenizer,
                                device_batch_size: int) -> DataLoader:
    """Builds a finetuning dataloader for training or evaluating.

    Args:
        cfg (DictConfig): An omegaconf dictionary used to configure the loader:
            cfg.name (str): The type of dataloader to build. Must = "finetuning".
            ---
            cfg.dataset.name (str): The name of the HuggingFace dataset to use
                and/or the name of the tokenization function to use -- the
                tokenization function must be registered in `tasks.py`.
                See README for details.
            cfg.dataset.kwargs (DictConfig, optional): Additional kwargs to
                pass to `datasets.load_dataset`, which can be used to load
                a dataset from local files.
            cfg.dataset.remote (str, optional): Location of a MDS-formatted
                streaming dataset to use. Setting this will tell the builder
                to create a streaming dataset rather than a HuggingFace dataset.
            cfg.dataset.local (str, optional): Local path where remote data
                will be streamed to. Only valid if `cfg.dataset.remote` has
                also been set.
            cfg.dataset.max_seq_len (int): The maximum length of sequences
                in the batch. See :class:`Seq2SeqFinetuningCollator` docstring
                for details.
            cfg.dataset.decoder_only_format (bool): Whether to format the
                examples for a decoder-only model. See :class:`Seq2SeqFinetuningCollator`
                docstring for details.
            cfg.dataset.allow_pad_trimming (bool, optional): Whether to allow
                the collator to trim padding. See :class:`Seq2SeqFinetuningCollator`
                docstring for details. Default: ``False``.
            cfg.dataset.packing_ratio (float, optional): If provided, this invokes
                a collator wrapper that packs `device_batch_size*packing_ratio`
                raw examples into `device_batch_size` packed examples. This helps
                minimize padding while preserving sequence integrity.
                This adds `sequence_id` to the batch, which indicates which unique
                sequence each token belongs to.
                Note: Using this feature will not change device_batch_size but it
                    will determine the number of raw examples consumed by the dataloader
                    per batch. Some examples may be discarded if they do not fit when
                    packing.
                    Select `packing_ratio` **carefully** based on the dataset
                    statistics, `max_seq_len`, and tolerance for discarding samples!
                    The packing code in `../packing.py` provides a script that can help
                    you choose the best `packing_ratio`.
            cfg.dataset.shuffle (bool): Whether to shuffle the dataset.
            ___
            See :class:`StreamingTextDataset` for info on other standard config
                options within `cfg.dataset` that will be passed as kwargs if
                using the streaming codepath.
            ---
            See :class:`DataLoader` for standard argument options to the pytorch
                dataloader, such as `cfg.drop_last`, `cfg.num_workers`, etc.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer used to
            prepare the data from raw text. Any missing sentinel tokens will
            be added by the collator.
        device_batch_size (int): The size of the batches (number of examples)
            that the dataloader will produce.

    Returns:
        A pytorch dataloader

    Note:
        You can run the script inside `../packing.py` to quickly test the
        padding/waste rates for different `cfg.dataset.packing_ratio` choices,
        given a starting workload YAML.
    """
    # Use EOS as the pad token if none exists
    if tokenizer.pad_token is None:  # type: ignore
        tokenizer.pad_token = tokenizer.eos_token

    if cfg.dataset.get('local') is not None:
        dataset = dataset_constructor.build_from_streaming(
            cfg.dataset.name,
            tokenizer=tokenizer,
            local=cfg.dataset.local,
            remote=cfg.dataset.get('remote', None),
            split=cfg.dataset.get('split'),
            shuffle=cfg.dataset.get('shuffle', False),
            predownload=cfg.dataset.get('predownload', 100_000),
            keep_zip=cfg.dataset.get('keep_zip', None),
            download_retry=cfg.dataset.get('download_retry', 2),
            download_timeout=cfg.dataset.get('download_timeout', 60),
            validate_hash=cfg.dataset.get('validate_hash', None),
            shuffle_seed=cfg.dataset.get('shuffle_seed', 9176),
            num_canonical_nodes=cfg.dataset.get('num_canonical_nodes', 128),
            batch_size=device_batch_size,
        )

        collate_fn, dataloader_batch_size = _build_collate_fn(
            cfg, tokenizer, device_batch_size)

        return DataLoader(
            dataset,
            collate_fn=collate_fn,
            batch_size=dataloader_batch_size,
            drop_last=cfg.drop_last,
            num_workers=cfg.num_workers,
            pin_memory=cfg.get('pin_memory', True),
            prefetch_factor=cfg.get('prefetch_factor', 2),
            persistent_workers=cfg.get('persistent_workers', True),
            timeout=cfg.get('timeout', 0),
        )

    else:
        if cfg.dataset.get('remote') is not None:
            raise ValueError(
                f'{cfg.dataset.remote=} but cfg.dataset.local is None.')

        dataset = dataset_constructor.build(cfg.dataset, tokenizer)

        collate_fn, dataloader_batch_size = _build_collate_fn(
            cfg.dataset, tokenizer, device_batch_size)

        return DataLoader(
            dataset,
            collate_fn=collate_fn,
            batch_size=dataloader_batch_size,
            sampler=dist.get_sampler(dataset,
                                     drop_last=cfg.drop_last,
                                     shuffle=cfg.dataset.shuffle),
            num_workers=cfg.num_workers,
            pin_memory=cfg.get('pin_memory', True),
            prefetch_factor=cfg.get('prefetch_factor', 2),
            persistent_workers=cfg.get('persistent_workers', True),
            timeout=cfg.get('timeout', 0),
        )


def _build_collate_fn(dataset_cfg: DictConfig, tokenizer: Tokenizer,
                      device_batch_size: int):
    collate_fn = Seq2SeqFinetuningCollator(
        tokenizer=tokenizer,
        max_seq_len=dataset_cfg.max_seq_len,
        decoder_only_format=dataset_cfg.decoder_only_format,
        allow_pad_trimming=dataset_cfg.get('allow_pad_trimming', False),
    )

    packing_ratio = dataset_cfg.get('packing_ratio')
    if packing_ratio is None:
        if dataset_cfg.get('max_leftover_bins_to_keep') is not None:
            raise ValueError(
                'dataset.max_leftover_bins_to_keep has been defined, ' +\
                'but dataset.packing_ratio has not been set. Please set ' +\
                'the latter to turn on packing or remove the former from the config.')
        return collate_fn, device_batch_size

    if packing_ratio == 1.0:
        return collate_fn, device_batch_size
    elif packing_ratio < 1.0:
        raise ValueError('packing_ratio must be >= 1, if supplied')

    if not dataset_cfg.decoder_only_format:
        raise NotImplementedError(
            'On-the-fly packing is currently only supported for decoder-only formats.'
        )

    collate_fn = BinPackWrapper(
        collator=collate_fn,
        target_batch_size=device_batch_size,
        max_seq_len=dataset_cfg.max_seq_len,
        pad_token_id=tokenizer.pad_token_id,
        padding_side=tokenizer.padding_side,
        max_leftover_bins_to_keep=dataset_cfg.get('max_leftover_bins_to_keep'),
    )
    n_examples_to_pack = int(device_batch_size * packing_ratio)
    return collate_fn, n_examples_to_pack


if __name__ == '__main__':
    import torch
    from omegaconf import OmegaConf as om

    from examples.common import build_tokenizer
    cfg = om.create({
        'dataset': {
            'name': 'tatsu-lab/alpaca',
            'split': 'train',
            'packing_ratio': 18.0,
            'max_seq_len': 2048,
            'decoder_only_format': True,
            'separator_text': False,
            'allow_pad_trimming': False,
            'num_canonical_nodes': 472,
            'shuffle': True,
        },
        'drop_last': False,
        'num_workers': 0,
        'pin_memory': False,
        'prefetch_factor': 2,
        'persistent_workers': False,
        'timeout': 0
    })

    tokenizer_cfg = {'name': 'gpt2', 'kwargs': {}}
    tokenizer_cfg['kwargs'] = {'model_max_length': cfg.dataset.max_seq_len}
    tokenizer_cfg = om.create(tokenizer_cfg)
    tokenizer = build_tokenizer(tokenizer_cfg)

    device_batch_size = 2
    dataloader = build_finetuning_dataloader(cfg, tokenizer, device_batch_size)

    packing = cfg.dataset.get('packing_ratio') is not None

    for i, batch in enumerate(dataloader):
        if i >= 5:
            break
        print(f'-----Batch {i}-----')
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                print(k, v.shape)
            else:
                print(k, v)
        for j in range(device_batch_size):
            print(f'--- Sample {j} ---')
            if cfg.dataset.decoder_only_format:
                if packing:
                    for subseq in range(int(batch['sequence_id'][j].max()) + 1):
                        is_subseq = batch['sequence_id'][j] == subseq
                        print(
                            '\033[93m{}\033[00m\n'.format('INPUT IDS:'),
                            tokenizer.decode(batch['input_ids'][
                                j,
                                torch.logical_and(
                                    is_subseq, batch['attention_mask'][j] ==
                                    1)],
                                             skip_special_tokens=False))
                        print(
                            '\033[92m{}\033[00m\n'.format('CONTEXT:  '),
                            tokenizer.decode(batch['input_ids'][
                                j,
                                torch.logical_and(
                                    is_subseq, batch['bidirectional_mask'][j] ==
                                    1)],
                                             skip_special_tokens=False))
                        print(
                            '\033[91m{}\033[00m\n'.format('TARGET:   '),
                            tokenizer.decode(batch['input_ids'][
                                j,
                                torch.logical_and(
                                    is_subseq,
                                    batch['labels'][j] != _HF_IGNORE_INDEX)],
                                             skip_special_tokens=False))
                else:
                    print(
                        '\033[93m{}\033[00m\n'.format('INPUT IDS:'),
                        tokenizer.decode(
                            batch['input_ids'][j,
                                               batch['attention_mask'][j] == 1],
                            skip_special_tokens=False))
                    print(
                        '\033[92m{}\033[00m\n'.format('CONTEXT:  '),
                        tokenizer.decode(batch['input_ids'][
                            j, batch['bidirectional_mask'][j] == 1],
                                         skip_special_tokens=False))
                    print(
                        '\033[91m{}\033[00m\n'.format('TARGET:   '),
                        tokenizer.decode(batch['input_ids'][
                            j, batch['labels'][j] != _HF_IGNORE_INDEX],
                                         skip_special_tokens=False))
            else:
                print(
                    '\033[92m{}\033[00m\n'.format('CONTEXT:  '),
                    tokenizer.decode(
                        batch['input_ids'][j, batch['attention_mask'][j] == 1],
                        skip_special_tokens=False))
                print(
                    '\033[91m{}\033[00m\n'.format('TARGET:   '),
                    tokenizer.decode(batch['labels'][
                        j, batch['decoder_attention_mask'][j] == 1],
                                     skip_special_tokens=False))
        print('   ')
