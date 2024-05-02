# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0
import logging
import os
from typing import Tuple, Union

import torch
from composer.core.data_spec import DataSpec
from composer.utils import dist, get_file, parse_uri
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

from llmfoundry import registry
from llmfoundry.data.finetuning.collator import (Seq2SeqFinetuningCollator,
                                                 validate_target_settings)
from llmfoundry.data.finetuning.tasks import (DOWNLOADED_FT_DATASETS_DIRPATH,
                                              SUPPORTED_EXTENSIONS,
                                              dataset_constructor)
from llmfoundry.data.packing import BinPackCollator, auto_packing_ratio
from llmfoundry.data.text_data import build_streams
from llmfoundry.utils.exceptions import (MissingHuggingFaceURLSplitError,
                                         NotEnoughDatasetSamplesError)
from llmfoundry.utils.registry_utils import construct_from_registry

log = logging.getLogger(__name__)

__all__ = [
    'build_finetuning_dataloader',
]

# HuggingFace hardcodes the ignore index to -100
_HF_IGNORE_INDEX = -100

# Default settings to use for target responses and target prompts
_DEFAULT_TARGET_RESPONSES = 'last'
_DEFAULT_TARGET_PROMPTS = 'none'


def build_finetuning_dataloader(
        cfg: DictConfig, tokenizer: PreTrainedTokenizerBase,
        device_batch_size: Union[int, float]) -> DataSpec:
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
            cfg.dataset.target_responses (str): Which responses are used as training targets.
                Defaults to "last", meaning only the final response in multi-turn examples
                will serve as training targets. See :class:`Seq2SeqFinetuningCollator` docstring for
                details.
            cfg.dataset.target_prompts (str): Which prompts are used as training targets.
                Defaults to "none", meaning prompts are never used as training targets.
                See :class:`Seq2SeqFinetuningCollator` docstring for details.
            cfg.dataset.allow_pad_trimming (bool, optional): Whether to allow
                the collator to trim padding. See :class:`Seq2SeqFinetuningCollator`
                docstring for details. Default: ``False``.
            cfg.dataset.packing_ratio (Optional[float, Literal['auto']]): If provided, this invokes
                a collator wrapper that packs device_batch_size*packing_ratio
                raw examples into device_batch_size packed examples. This helps
                minimize padding while preserving sequence integrity.
                This adds `sequence_id` to the batch, which indicates which unique
                sequence each token belongs to.

                If set to 'auto', packing_ratio is profiled and the highest observed packing ratio with
                zero waste is selected.
                In practice, this may result in > 0 waste because profiling is done on only a portion
                of the dataset.

                Note: Using this feature will not change device_batch_size but it
                    will determine the number of raw examples consumed by the dataloader
                    per batch. Some examples may be discarded if they do not fit when
                    packing.
                    Select packing_ratio **carefully** based on the dataset
                    statistics, max_seq_len, and tolerance for discarding samples!
                    The script `scripts/misc/profile_packing.py` can help
                    you choose the best packing_ratio.
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
        device_batch_size (int, float): The size of the batches (number of examples)
            that the dataloader will produce.

    Returns:
        A pytorch dataloader

    Note:
        You can run the script inside `scripts/misc/profile_packing.py` to quickly test the
        padding/waste rates for different `cfg.dataset.packing_ratio` choices,
        given a starting workload YAML.
    """
    _validate_config(cfg.dataset)

    # Use EOS as the pad token if none exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    replication_factor, ds_batch_size = construct_from_registry(
        name='dataset_replication_validator',
        registry=registry.dataset_replication_validators,
        partial_function=False,
        kwargs={
            'cfg': cfg,
            'tokenizer': tokenizer,
            'device_batch_size': device_batch_size
        },
    )

    collate_fn, dataloader_batch_size = construct_from_registry(
        name='finetuning_collator',
        registry=registry.collators,
        partial_function=False,
        kwargs={
            'cfg': cfg,
            'tokenizer': tokenizer,
            'ds_batch_size': ds_batch_size,
        },
    )

    dataset = None  # for pyright
    sampler = None
    if cfg.dataset.get('remote') is not None or cfg.dataset.get(
            'streams') is not None:
        # Build streaming dataloader
        streams = build_streams(cfg.dataset)
        dataset = dataset_constructor.build_from_streaming(
            tokenizer=tokenizer,
            streams=streams,
            local=cfg.dataset.get('local', None),
            remote=cfg.dataset.get('remote', None),
            split=cfg.dataset.get('split', None),
            download_retry=cfg.dataset.get('download_retry', 2),
            download_timeout=cfg.dataset.get('download_timeout', 60),
            validate_hash=cfg.dataset.get('validate_hash', None),
            keep_zip=cfg.dataset.get('keep_zip', False),
            epoch_size=cfg.dataset.get('epoch_size', None),
            predownload=cfg.dataset.get('predownload', None),
            cache_limit=cfg.dataset.get('cache_limit', None),
            partition_algo=cfg.dataset.get('partition_algo', 'relaxed'),
            num_canonical_nodes=cfg.dataset.get('num_canonical_nodes', None),
            batch_size=ds_batch_size,
            shuffle=cfg.dataset.get('shuffle', False),
            shuffle_algo=cfg.dataset.get('shuffle_algo', 'py1e'),
            shuffle_seed=cfg.dataset.get('shuffle_seed', 9176),
            shuffle_block_size=cfg.dataset.get('shuffle_block_size', None),
            sampling_method=cfg.dataset.get('sampling_method', 'balanced'),
            sampling_granularity=cfg.dataset.get('sampling_granularity', 1),
            batching_method=cfg.dataset.get('batching_method', 'random'),
            max_seq_len=cfg.dataset.max_seq_len,
            allow_unsafe_types=cfg.dataset.get('allow_unsafe_types', False),
            replication=replication_factor,
        )

    else:
        # Build HF dataloader
        dataset_name_or_path = cfg.dataset.hf_name
        split = cfg.dataset.get('split')
        if split is None:
            raise MissingHuggingFaceURLSplitError()

        # If dataset is a remote path, download it first.
        backend, _, _ = parse_uri(dataset_name_or_path)
        if backend not in ['', None]:
            dataset_name_or_path = _download_remote_hf_dataset(
                remote_path=dataset_name_or_path, split=split)
            split = split.replace('-', '_')

        # Get the preprocessing function.
        proto_preprocessing_fn = cfg.dataset.get('preprocessing_fn')
        if isinstance(proto_preprocessing_fn, (dict, DictConfig)):
            preprocessing_fn = dataset_constructor.get_preprocessing_fn_from_dict(
                dict(proto_preprocessing_fn))
        else:
            preprocessing_fn = dataset_constructor.get_preprocessing_fn_from_str(
                proto_preprocessing_fn, dataset_name_or_path)

        # Build dataset from HF.
        dataset = dataset_constructor.build_from_hf(
            dataset_name=dataset_name_or_path,
            split=split,
            safe_load=cfg.dataset.get('safe_load', False),
            max_seq_len=cfg.dataset.max_seq_len,
            preprocessing_fn=preprocessing_fn,
            tokenizer=tokenizer,
            target_prompts=cfg.dataset.get('target_prompts',
                                           _DEFAULT_TARGET_PROMPTS),
            target_responses=cfg.dataset.get('target_responses',
                                             _DEFAULT_TARGET_RESPONSES),
            decoder_only_format=cfg.dataset.decoder_only_format,
            hf_kwargs=cfg.dataset.get('hf_kwargs', {}))

        # Ensure dataset is large enough.
        if cfg.drop_last:
            world_size = dist.get_world_size() // replication_factor
            minimum_dataset_size = world_size * dataloader_batch_size
            if hasattr(dataset, '__len__'):
                full_dataset_size = len(dataset)
                if full_dataset_size < minimum_dataset_size:
                    raise NotEnoughDatasetSamplesError(
                        dataset_name=cfg.dataset.hf_name,
                        split=split,
                        dataloader_batch_size=dataloader_batch_size,
                        world_size=world_size,
                        full_dataset_size=full_dataset_size,
                        minimum_dataset_size=minimum_dataset_size)
        # Initialize sampler.
        sampler = dist.get_sampler(
            dataset,
            drop_last=cfg.drop_last,
            shuffle=cfg.dataset.shuffle,
            num_replicas=dist.get_world_size() //
            replication_factor if replication_factor > 1 else None,
            rank=dist.get_global_rank() //
            replication_factor if replication_factor > 1 else None,
        )

    assert dataset is not None  # for pyright
    dl = DataLoader(
        dataset,
        collate_fn=collate_fn,
        batch_size=dataloader_batch_size,
        drop_last=cfg.drop_last,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=cfg.get('pin_memory', True),
        prefetch_factor=cfg.get('prefetch_factor', 2),
        persistent_workers=cfg.get('persistent_workers', True),
        timeout=cfg.get('timeout', 0),
    )

    return construct_from_registry(
        name='data_spec',
        registry=registry.data_specs,
        partial_function=False,
        kwargs={
            'dl': dl,
            'dataset_cfg': cfg.dataset
        },
    )


def _validate_config(dataset_cfg: DictConfig) -> None:
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
        illegal_keys = ['hf_name', 'hf_kwargs', 'preprocessing_fn', 'safe_load']
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
    elif dataset_cfg.get('streams') is not None:
        # Using the streaming dataset codepath
        illegal_keys = ['hf_name', 'hf_kwargs', 'preprocessing_fn', 'safe_load']
        discovered_illegal_keys = []
        for key in illegal_keys:
            if dataset_cfg.get(key) is not None:
                discovered_illegal_keys.append('`' + key + '`')
        if discovered_illegal_keys:
            raise ValueError(
                'The dataset config sets a value for `streams` as well as the ' +\
                f'following keys: {", ".join(discovered_illegal_keys)}.\n' +\
                'Those keys are used when building from a HuggingFace dataset, but ' +\
                'setting `streams` instructs the dataset to build from a streaming dataset.'
            )
        illegal_keys = ['remote', 'local']
        discovered_illegal_keys = []
        for key in illegal_keys:
            if dataset_cfg.get(key) is not None:
                discovered_illegal_keys.append('`' + key + '`')
        if discovered_illegal_keys:
            raise ValueError(
                'The dataset config sets a value for `streams` as well as the ' +\
                f'following keys: {", ".join(discovered_illegal_keys)}.\n' +\
                'Please either use single stream (set remote/local only) ' +\
                'or put remote/local under streams'
            )

    else:
        raise ValueError(
            'In the dataset config, you must set `hf_name` to use a HuggingFace ' +\
            'dataset, or set `remote` to use a streaming dataset, or set ' +\
            '`streams` to use multiple streaming datasets,  but all were None.'
        )
    max_seq_len = dataset_cfg.get('max_seq_len')
    if max_seq_len is None:
        raise ValueError(
            'In the dataset config, you must set the `max_seq_len`')

    if max_seq_len != int(max_seq_len):
        raise ValueError('max_seq_len must be an integer')
    dataset_cfg['max_seq_len'] = int(max_seq_len)

    # Raise an error if the target_prompts + target_responses + decoder_only_format settings
    # are invalid
    target_responses = str(
        dataset_cfg.get('target_responses', _DEFAULT_TARGET_RESPONSES)).lower()
    target_prompts = str(
        dataset_cfg.get('target_prompts', _DEFAULT_TARGET_PROMPTS)).lower()
    decoder_only_format = dataset_cfg.decoder_only_format
    validate_target_settings(target_prompts, target_responses,
                             decoder_only_format)


def _download_remote_hf_dataset(remote_path: str, split: str) -> str:
    """Downloads a dataset from a remote object store.

    This function supports 'jsonl', 'csv', and 'parquet' file formats for the dataset. It will attempt to download
    the dataset, then once it is downloaded, convert it into HuggingFace ``datasets`` format, and then return this
    dataset.

    The function also ensures synchronicity across multiple processes during the file download. It creates a signal
    file that is used to synchronize the start of the download across different processes. Once the download is
    completed, the function removes the signal file.

    Args:
        hf_name (str): The path of the HuggingFace dataset to download.
        split (str): The dataset split to download (e.g., 'train', 'validation', 'test').

    Returns:
        A local directory path where the dataset files are stored.

    Raises:
        FileNotFoundError: Raised if the dataset file cannot be found with any of the supported extensions.
    """
    # HF datasets does not support a split with dashes, so we replace dashes with underscores.
    hf_formatted_split = split.replace('-', '_')
    finetune_dir = os.path.join(
        DOWNLOADED_FT_DATASETS_DIRPATH,
        hf_formatted_split if hf_formatted_split != 'data' else 'data_not',
    )
    os.makedirs(finetune_dir, exist_ok=True)
    for extension in SUPPORTED_EXTENSIONS:
        name = f'{remote_path.strip("/")}/{split}{extension}'
        destination = str(
            os.path.abspath(
                os.path.join(
                    finetune_dir, 'data',
                    f'{hf_formatted_split}-00000-of-00001{extension}')))

        # Since we don't know exactly what the extension will be, since it is one of a list
        # use a signal file to wait for instead of the desired file
        signal_file_path = os.path.join(
            finetune_dir, f'.node_{dist.get_node_rank()}_local_rank0_completed')
        if dist.get_local_rank() == 0:
            try:
                get_file(path=name, destination=destination, overwrite=True)
            except FileNotFoundError as e:
                if extension == SUPPORTED_EXTENSIONS[-1]:
                    files_searched = [
                        f'{cfg.dataset.hf_name}/{cfg.dataset.split}{ext}'
                        for ext in SUPPORTED_EXTENSIONS
                    ]
                    raise FileNotFoundError(
                        f'Could not find a file with any of ' + \
                        f'the supported extensions: {SUPPORTED_EXTENSIONS}\n' + \
                        f'at {files_searched}'
                    ) from e
                else:
                    log.debug(
                        f'Could not find {name}, looking for another extension')
                continue

            os.makedirs(os.path.dirname(signal_file_path), exist_ok=True)
            with open(signal_file_path, 'wb') as f:
                f.write(b'local_rank0_completed_download')

        # Avoid the collective call until the local rank zero has finished trying to download the dataset
        # so that we don't timeout for large downloads. This syncs all processes on the node
        with dist.local_rank_zero_download_and_wait(signal_file_path):
            # Then, wait to ensure every node has finished trying to download the dataset
            dist.barrier()

        # clean up signal file
        if dist.get_local_rank() == 0:
            os.remove(signal_file_path)
        dist.barrier()
        break
    return finetune_dir


def build_collate_fn(
    dataloader_cfg: DictConfig, tokenizer: PreTrainedTokenizerBase,
    device_batch_size: int
) -> Tuple[Union[Seq2SeqFinetuningCollator, BinPackCollator], int]:
    dataset_cfg = dataloader_cfg.dataset
    max_seq_len = dataset_cfg.max_seq_len

    collate_fn = Seq2SeqFinetuningCollator(
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        decoder_only_format=dataset_cfg.decoder_only_format,
        target_responses=dataset_cfg.get('target_responses',
                                         _DEFAULT_TARGET_RESPONSES),
        target_prompts=dataset_cfg.get('target_prompts',
                                       _DEFAULT_TARGET_PROMPTS),
        allow_pad_trimming=dataset_cfg.get('allow_pad_trimming', False),
    )

    packing_ratio = dataset_cfg.get('packing_ratio')
    max_leftover_bins_to_keep = dataset_cfg.get('max_leftover_bins_to_keep')
    if packing_ratio is None:
        if max_leftover_bins_to_keep is not None:
            raise ValueError(
                'dataset.max_leftover_bins_to_keep has been defined, ' +\
                'but dataset.packing_ratio has not been set. Please set ' +\
                'the latter to turn on packing or remove the former from the config.')
        return collate_fn, device_batch_size

    if packing_ratio == 'auto':
        packing_ratio = auto_packing_ratio(dataloader_cfg, tokenizer,
                                           device_batch_size)

    if isinstance(packing_ratio, str):
        raise ValueError(
            'dataset.packing_ratio must be a float or "auto", but it was set to '
            + f'{packing_ratio}.')

    log.info(f'Using packing ratio {packing_ratio}')

    if packing_ratio == 1.0:
        return collate_fn, device_batch_size
    elif packing_ratio < 1.0:
        raise ValueError('packing_ratio must be >= 1, if supplied')

    if not dataset_cfg.decoder_only_format:
        raise NotImplementedError(
            'On-the-fly packing is currently only supported for decoder-only formats.'
        )

    collate_fn = BinPackCollator(
        collator=collate_fn,
        target_batch_size=device_batch_size,
        max_seq_len=max_seq_len,
        pad_token_id=tokenizer.pad_token_id,
        padding_side=tokenizer.padding_side,
        max_leftover_bins_to_keep=max_leftover_bins_to_keep,
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
            'allow_pad_trimming':
                False,
            'num_canonical_nodes':
                472,
            'shuffle':
                True,
            'target_responses':
                'last',
            'target_prompts':
                'none',
        },
        'drop_last': False,
        'num_workers': 0,
        'pin_memory': False,
        'prefetch_factor': None,
        'persistent_workers': False,
        'timeout': 0
    })

    tokenizer_name = 'EleutherAI/gpt-neox-20b'
    tokenizer_kwargs = {'model_max_length': cfg.dataset.max_seq_len}
    tokenizer = build_tokenizer(tokenizer_name, tokenizer_kwargs)

    device_batch_size = 1
    dataloader = build_finetuning_dataloader(cfg, tokenizer,
                                             device_batch_size).dataloader

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
                                             skip_special_tokens=False,
                                             clean_up_tokenization_spaces=True))
                        context = torch.logical_and(
                            batch['attention_mask'][j] == 1,
                            batch['labels'][j] == _HF_IGNORE_INDEX)
                        print(
                            '\033[92m{}\033[00m\n'.format('CONTEXT:  '),
                            tokenizer.decode(batch['input_ids'][
                                j, torch.logical_and(is_subseq, context)],
                                             skip_special_tokens=False,
                                             clean_up_tokenization_spaces=True))
                        print(
                            '\033[91m{}\033[00m\n'.format('TARGET:   '),
                            tokenizer.decode(batch['input_ids'][
                                j,
                                torch.logical_and(
                                    is_subseq,
                                    batch['labels'][j] != _HF_IGNORE_INDEX)],
                                             skip_special_tokens=False,
                                             clean_up_tokenization_spaces=True))
                else:
                    print(
                        '\033[93m{}\033[00m\n'.format('INPUT IDS:'),
                        tokenizer.decode(
                            batch['input_ids'][j,
                                               batch['attention_mask'][j] == 1],
                            skip_special_tokens=False,
                            clean_up_tokenization_spaces=True))
                    context = torch.logical_and(
                        batch['attention_mask'][j] == 1,
                        batch['labels'][j] == _HF_IGNORE_INDEX)
                    print(
                        '\033[92m{}\033[00m\n'.format('CONTEXT:  '),
                        tokenizer.decode(batch['input_ids'][j, context],
                                         skip_special_tokens=False,
                                         clean_up_tokenization_spaces=True))
                    print(
                        '\033[91m{}\033[00m\n'.format('TARGET:   '),
                        tokenizer.decode(batch['input_ids'][
                            j, batch['labels'][j] != _HF_IGNORE_INDEX],
                                         skip_special_tokens=False,
                                         clean_up_tokenization_spaces=True))
            else:
                print(
                    '\033[92m{}\033[00m\n'.format('CONTEXT:  '),
                    tokenizer.decode(
                        batch['input_ids'][j, batch['attention_mask'][j] == 1],
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=True))
                print(
                    '\033[91m{}\033[00m\n'.format('TARGET:   '),
                    tokenizer.decode(batch['labels'][
                        j, batch['decoder_attention_mask'][j] == 1],
                                     skip_special_tokens=False,
                                     clean_up_tokenization_spaces=True))
        print('   ')
