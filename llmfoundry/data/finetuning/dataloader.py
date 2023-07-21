# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import logging
import os
from typing import Union

import torch
from composer.utils import dist, get_file, parse_uri
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from llmfoundry.data.finetuning.collator import Seq2SeqFinetuningCollator
from llmfoundry.data.finetuning.tasks import dataset_constructor
from llmfoundry.data.packing import BinPackWrapper

log = logging.getLogger(__name__)

# HuggingFace hardcodes the ignore index to -100
_HF_IGNORE_INDEX = -100

Tokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]


def build_finetuning_dataloader(cfg: DictConfig, tokenizer: Tokenizer,
                                device_batch_size: int) -> DataLoader:
    """Builds a finetuning dataloader for training or evaluating.

    The underlying dataset can be built through one of two code paths:
        1. As a HuggingFace dataset, via `datasets.load_dataset(...)`
        2. As a streaming dataset
    You will need to set slightly different dataset config fields depending
    on which you intend to use, as explained below.

    Args:
        cfg (DictConfig): An omegaconf dictionary used to configure the loader:
            cfg.name (str): The type of dataloader to build. Must = "finetuning".
            ---
            *** HuggingFace dataset config fields ***
            cfg.dataset.hf_name (str, optional): The name of the HuggingFace dataset
                to use. Can also be a remote http(s) directory or object store bucket
                containing the file {split}.jsonl in the format (prompt, response),
                in which case the builder will create a HuggingFace dataset.
            cfg.dataset.hf_kwargs (DictConfig, optional): Additional kwargs to
                pass to `datasets.load_dataset`, which can be used to load
                a dataset from local files.
            cfg.dataset.preprocessing_fn (str, optional): The name/import path of
                the preprocessing function to use for formatting the data examples.
                If ``None`` (default), the builder will use the preprocessing function
                    registered under `hf_name` (see `tasks.py`), if one exists,
                    otherwise it will skip preprocessing.
                If `preprocessing_fn` corresponds to a registered preprocessing
                    function in `tasks.py`, the builder will use that.
                Otherwise, it will interpret `preprocessing_fn` as a
                    "import.path:function_name" import path; e.g., it will call
                    `from import.path import function_name` and use the imported
                    function as the preprocessing function.
            *** Streaming dataset config fields ***
            cfg.dataset.remote (str, optional): Location of a MDS-formatted
                streaming dataset to use. Setting this will tell the builder
                to create a streaming dataset rather than a HuggingFace dataset.
            cfg.dataset.local (str, optional): Local path where remote data
                will be streamed to. Only valid if `cfg.dataset.remote` has
                also been set.
            *** Shared dataset configs fields ***
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
            See :class:`StreamingFinetuningDataset` for info on other standard config
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
    _validate_config(cfg.dataset)

    # Use EOS as the pad token if none exists
    if tokenizer.pad_token is None:  # type: ignore
        tokenizer.pad_token = tokenizer.eos_token

    if cfg.dataset.get('remote') is not None:
        dataset = dataset_constructor.build_from_streaming(
            tokenizer=tokenizer,
            local=cfg.dataset.local,
            remote=cfg.dataset.get('remote', None),
            split=cfg.dataset.get('split'),
            shuffle=cfg.dataset.get('shuffle', False),
            predownload=cfg.dataset.get('predownload', 100_000),
            keep_zip=cfg.dataset.get('keep_zip', False),
            download_retry=cfg.dataset.get('download_retry', 2),
            download_timeout=cfg.dataset.get('download_timeout', 60),
            validate_hash=cfg.dataset.get('validate_hash', None),
            shuffle_seed=cfg.dataset.get('shuffle_seed', 9176),
            num_canonical_nodes=cfg.dataset.get('num_canonical_nodes', 128),
            batch_size=device_batch_size,
        )

        collate_fn, dataloader_batch_size = _build_collate_fn(
            cfg.dataset, tokenizer, device_batch_size)

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
        backend, _, _ = parse_uri(cfg.dataset.hf_name)
        if backend not in ['', None]:
            if cfg.dataset.get('split') is None:
                raise ValueError(
                    'When using a HuggingFace dataset from a URL, you must set the ' + \
                    '`split` key in the dataset config.'
                )
            dataset = _build_hf_dataset_from_remote(cfg, tokenizer)
        else:
            dataset = dataset_constructor.build_from_hf(
                cfg.dataset,
                max_seq_len=cfg.dataset.max_seq_len,
                tokenizer=tokenizer,
            )

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


def _validate_config(dataset_cfg: DictConfig):
    """Validates the dataset configuration.

    Makes sure that the dataset is properly configured for either
    a HuggingFace dataset or a streaming dataset. Must be valid for one or
    the other.

    Args:
        dataset_cfg (DictConfig): The dataset configuration to be validated.

    Raises:
        ValueError: If the dataset configuration does not meet the requirements.
    """
    if dataset_cfg.get('hf_name') is not None:
        # Using the HuggingFace dataset codepath
        illegal_keys = ['local', 'remote']
        discovered_illegal_keys = []
        for key in illegal_keys:
            if dataset_cfg.get(key) is not None:
                discovered_illegal_keys.append('`' + key + '`')
        if discovered_illegal_keys:
            raise ValueError(
                'The dataset config sets a value for `hf_name` as well as the ' +\
                f'following keys: {", ".join(discovered_illegal_keys)}.\n' +\
                'Those keys are used when building from a streaming dataset, but ' +\
                'setting `hf_name` instructs the dataset to build from a HuggingFace dataset.'
            )
    elif dataset_cfg.get('remote') is not None:
        # Using the streaming dataset codepath
        illegal_keys = ['hf_name', 'hf_kwargs', 'preprocessing_fn']
        discovered_illegal_keys = []
        for key in illegal_keys:
            if dataset_cfg.get(key) is not None:
                discovered_illegal_keys.append('`' + key + '`')
        if discovered_illegal_keys:
            raise ValueError(
                'The dataset config sets a value for `remote` as well as the ' +\
                f'following keys: {", ".join(discovered_illegal_keys)}.\n' +\
                'Those keys are used when building from a HuggingFace dataset, but ' +\
                'setting `remote` instructs the dataset to build from a streaming dataset.'
            )
        if dataset_cfg.get('local') is None:
            raise ValueError(
                'Using a streaming dataset requires setting both `remote` and `local`, ' +\
                'but dataset.local is None.'
            )
    else:
        raise ValueError(
            'In the dataset config, you must set either `hf_name` to use a ' +\
            'HuggingFace dataset or set `remote` to use a streaming ' +\
            'dataset, but both were None.'
        )


def _build_hf_dataset_from_remote(cfg: DictConfig, tokenizer: Tokenizer):
    """Builds a dataset from a remote object store.

    This function supports 'jsonl', 'csv', and 'parquet' file formats for the dataset. It will attempt to download
    the dataset, then once it is downloaded, convert it into HuggingFace ``datasets`` format, and then return this
    dataset.

    The function also ensures synchronicity across multiple processes during the file download. It creates a signal
    file that is used to synchronize the start of the download across different processes. Once the download is
    completed, the function removes the signal file.

    Args:
        cfg (DictConfig): The configuration dictionary containing the necessary parameters to load the dataset.
            This includes:
                - dataset.hf_name: The path of the HuggingFace dataset to download.
                - dataset.split: The dataset split to download (e.g., 'train', 'validation', 'test').
                - dataset.max_seq_len: The maximum sequence length for tokenizing the dataset.

        tokenizer (Tokenizer): The tokenizer to be used to tokenize the dataset.

    Returns:
        Dataset: A HuggingFace dataset built from the remote file, prepared and tokenized for fine-tuning the model.

    Raises:
        FileNotFoundError: Raised if the dataset file cannot be found with any of the supported extensions.
    """
    supported_extensions = ['jsonl', 'csv', 'parquet']
    finetune_dir = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        f'downloaded_finetuning_data/{cfg.dataset.split}')
    os.makedirs(finetune_dir, exist_ok=True)
    for extension in supported_extensions:
        name = f'{cfg.dataset.hf_name.strip("/")}/{cfg.dataset.split}.{extension}'
        destination = str(
            os.path.abspath(f'{finetune_dir}/{cfg.dataset.split}.{extension}'))
        # Since we don't know exactly what the extension will be, since it is one of a list
        # use a signal file to wait for instead of the desired file
        signal_file_path = os.path.join(finetune_dir, '.the_eagle_has_landed')
        if dist.get_local_rank() == 0:
            try:
                get_file(name, destination, overwrite=True)
            except FileNotFoundError as e:
                if extension == supported_extensions[-1]:
                    raise FileNotFoundError(
                        f'Could not find a {cfg.dataset.split} file with any of ' + \
                        f'the supported extensions: {supported_extensions}\n' + \
                        f'at {cfg.dataset.hf_name}/{cfg.dataset.split}'
                    ) from e
                else:
                    print(
                        f'Could not find {name}, looking for another extension')
                continue

            os.makedirs(os.path.dirname(signal_file_path), exist_ok=True)
            with open(signal_file_path, 'wb') as f:
                f.write(b'local_rank0_completed_download')

        # Avoid the collective call until the local rank zero has finished trying to download the checkpoint
        # so that we don't timeout for large downloads. This syncs all processes on the node
        with dist.local_rank_zero_download_and_wait(signal_file_path):
            # Then, wait to ensure every node has finished downloading the checkpoint
            dist.barrier()

        # clean up signal file
        if dist.get_local_rank() == 0:
            os.remove(signal_file_path)
        dist.barrier()

        cfg.dataset.hf_name = finetune_dir
        print(cfg.dataset)
        dataset = dataset_constructor.build_from_hf(
            cfg.dataset,
            max_seq_len=cfg.dataset.max_seq_len,
            tokenizer=tokenizer,
        )
        return dataset


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

    from llmfoundry.utils import build_tokenizer
    cfg = om.create({
        'dataset': {
            'hf_name':
                'tatsu-lab/alpaca',
            'preprocessing_fn':
                'llmfoundry.data.finetuning.tasks:alpaca_preprocessing_function',
            'split':
                'train',
            'packing_ratio':
                18.0,
            'max_seq_len':
                2048,
            'decoder_only_format':
                True,
            'separator_text':
                False,
            'allow_pad_trimming':
                False,
            'num_canonical_nodes':
                472,
            'shuffle':
                True,
        },
        'drop_last': False,
        'num_workers': 0,
        'pin_memory': False,
        'prefetch_factor': 2,
        'persistent_workers': False,
        'timeout': 0
    })

    tokenizer_cfg = {'name': 'EleutherAI/gpt-neox-20b', 'kwargs': {}}
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
