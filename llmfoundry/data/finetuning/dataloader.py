# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0
import inspect
import logging
import os
from typing import Any, Optional, Union

import torch
from composer.core.data_spec import DataSpec
from composer.utils import dist, get_file, parse_uri
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

from llmfoundry import registry
from llmfoundry.data.finetuning.collator import (
    Seq2SeqFinetuningCollator,
    validate_target_settings,
)
from llmfoundry.data.finetuning.tasks import (
    DEFAULT_TARGET_PROMPTS,
    DEFAULT_TARGET_RESPONSES,
    SUPPORTED_EXTENSIONS,
    dataset_constructor,
)
from llmfoundry.data.packing import BinPackCollator, auto_packing_ratio
from llmfoundry.data.text_data import build_streams
from llmfoundry.utils.config_utils import to_dict_container
from llmfoundry.utils.consts import CROSS_ENTROPY_IGNORE_INDEX
from llmfoundry.utils.exceptions import (
    FinetuningFileNotFoundError,
    MissingHuggingFaceURLSplitError,
    NotEnoughDatasetSamplesError,
)
from llmfoundry.utils.file_utils import dist_mkdtemp
from llmfoundry.utils.registry_utils import construct_from_registry

log = logging.getLogger(__name__)

__all__ = [
    'build_finetuning_dataloader',
]

# Extra keys present in the dataset config dictionary beyond the constructor keys
_ALLOWED_DATASET_KEYS = {
    'shuffle',
    'packing_ratio',
    'allow_pad_trimming',
    'seq_parallel_replication',
    'auto_packing_replication',
    'max_leftover_bins_to_keep',
    'pad_to_longest',
}


def build_finetuning_dataloader(
    tokenizer: PreTrainedTokenizerBase,
    device_batch_size: Union[int, float],
    dataset: dict[str, Any],
    num_workers: int,
    drop_last: bool = False,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    persistent_workers: bool = True,
    timeout: int = 0,
) -> DataSpec:
    """Builds a finetuning dataloader for training or evaluating.

    The underlying dataset can be built through one of two code paths:
        1. As a HuggingFace dataset, via `datasets.load_dataset(...)`
        2. As a streaming dataset
    You will need to set slightly different dataset config fields depending
    on which you intend to use, as explained below.

    Args:
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer used to
            prepare the data from raw text. Any missing sentinel tokens will
            be added by the collator.
        device_batch_size (int, float): The size of the batches (number of examples)
            that the dataloader will produce.
        dataset (Dict[str, Any]): A HuggingFace dataset config which contains the following fields:
            dataset.hf_name (str, optional): The name of the HuggingFace dataset
                to use. Can also be a remote http(s) directory or object store bucket
                containing the file {split}.jsonl in the format (prompt, response),
                in which case the builder will create a HuggingFace dataset.
            dataset.hf_kwargs (DictConfig, optional): Additional kwargs to
                pass to `datasets.load_dataset`, which can be used to load
                a dataset from local files.
            dataset.preprocessing_fn (str, optional): The name/import path of
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
            dataset.remote (str, optional): Location of a MDS-formatted
                streaming dataset to use. Setting this will tell the builder
                to create a streaming dataset rather than a HuggingFace dataset.
            dataset.local (str, optional): Local path where remote data
                will be streamed to. Only valid if `cfg.dataset.remote` has
                also been set.
            *** Shared dataset configs fields ***
            dataset.max_seq_len (int): The maximum length of sequences
                in the batch. See :class:`Seq2SeqFinetuningCollator` docstring
                for details.
            dataset.decoder_only_format (bool): Whether to format the
                examples for a decoder-only model. See :class:`Seq2SeqFinetuningCollator`
                docstring for details.
            dataset.target_responses (str): Which responses are used as training targets.
                Defaults to "last", meaning only the final response in multi-turn examples
                will serve as training targets. See :class:`Seq2SeqFinetuningCollator` docstring for
                details.
            dataset.target_prompts (str): Which prompts are used as training targets.
                Defaults to "none", meaning prompts are never used as training targets.
                See :class:`Seq2SeqFinetuningCollator` docstring for details.
            dataset.allow_pad_trimming (bool, optional): Whether to allow
                the collator to trim padding. See :class:`Seq2SeqFinetuningCollator`
                docstring for details. Default: ``False``.
            dataset.packing_ratio (Optional[float, Literal['auto']]): If provided, this invokes
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
            dataset.shuffle (bool): Whether to shuffle the dataset.
            See :class:`StreamingFinetuningDataset` for info on other standard config
                options within `dataset` that will be passed as kwargs if
                using the streaming codepath.
        num_workers (int, optional): How many subprocesses to use for data loading.
            0 means that the data will be loaded in the main process. The default is 0.
            This argument is passed directly to the pytorch :class:`DataLoader`.
        drop_last (bool, optional): If true, drop the last incomplete batch, if the dataset
            size is not divisible by the batch size. If False and the size of dataset is
            not divisible by the batch size, then the last batch will be smaller. The
            default is False. This argument is passed directly to the pytorch :class:`DataLoader`.
        pin_memory (bool, optional): If True, the data loader will copy Tensors into device/CUDA
            pinned memory before returning them. If your data elements are a custom type, or your
            `collate_fn` returns a batch that is a custom type. This argument is passed directly to
            the pytorch :class:`DataLoader`.
        prefetch_factor (int, optional): Number of batches loaded in advance by each worker.
            2 means there will be a total of 2 * num_workers batches prefetched across all workers.
            (default value depends on the set value for num_workers. If value of num_workers=0 default
            is None. Otherwise, if value of num_workers > 0 default is 2). This argument is passed
            directly to the pytorch :class:`DataLoader`.
        persistent_workers (bool, optional): If True, the data loader will not shut down the worker
            processes after a dataset has been consumed once. This allows to maintain the workers
            Dataset instances alive. The default is False. This argument is passed directly to the
            pytorch :class:`DataLoader`.
        timeout (int, optional): If positive, the timeout value for collecting a batch from workers.
            Should always be non-negative. The default is 0. This argument is passed directly to the
            pytorch :class:`DataLoader`.
        See :class:`DataLoader` for standard argument options to the pytorch
            dataloader, such as `drop_last`, `num_workers`, etc.

    Returns:
        A pytorch dataloader

    Note:
        You can run the script inside `scripts/misc/profile_packing.py` to quickly test the
        padding/waste rates for different `cfg.dataset.packing_ratio` choices,
        given a starting workload YAML.
    """
    dataset_cfg = dataset
    is_streaming = (
        dataset_cfg.get('remote') is not None or
        dataset_cfg.get('streams') is not None
    )
    if is_streaming:
        dataset_constructor_keys = inspect.signature(
            dataset_constructor.streaming_dataset_class,
        ).parameters.keys()
    else:
        dataset_constructor_keys = inspect.signature(
            dataset_constructor.build_from_hf,
        ).parameters.keys()

    allowed_dataset_config_keys = set(
        dataset_constructor_keys,
    ).union(_ALLOWED_DATASET_KEYS)

    extraneous_keys = _validate_config(
        **dataset_cfg,
        allowed_dataset_keys=allowed_dataset_config_keys,
    )

    # Use EOS as the pad token if none exists
    if tokenizer.pad_token is None:  # type: ignore (sometimes it's none and that's ok)
        tokenizer.pad_token = tokenizer.eos_token

    # this full config is necessary for properly profiling the packing ratio
    dataloader_cfg = {
        'name': 'finetuning',
        'dataset': dataset_cfg,
        'drop_last': drop_last,
        'num_workers': num_workers,
        'pin_memory': pin_memory,
        'prefetch_factor': prefetch_factor,
        'persistent_workers': persistent_workers,
        'timeout': timeout,
    }

    replication_factor, dataset_batch_size = construct_from_registry(
        name='dataset_replication_validator',
        registry=registry.dataset_replication_validators,
        partial_function=False,
        kwargs={
            'dataset_cfg': dataset_cfg,
            'tokenizer': tokenizer,
            'device_batch_size': device_batch_size,
        },
    )

    collate_fn, dataloader_batch_size = construct_from_registry(
        name='finetuning_collator',
        registry=registry.collators,
        partial_function=False,
        kwargs={
            'dataloader_cfg': dataloader_cfg,
            'tokenizer': tokenizer,
            'dataset_batch_size': dataset_batch_size,
        },
    )

    streaming_dataset = None  # for pyright
    sampler = None
    if is_streaming:
        # Build streaming dataloader
        streams_cfg = dataset_cfg.get('streams', None)
        streams_cfg = to_dict_container(
            streams_cfg,
        ) if streams_cfg is not None else None
        streams = build_streams(
            streams_cfg,
        ) if streams_cfg is not None else None

        dataset_constructor_args = {
            k: v
            for k, v in dataset_cfg.items()
            if k in set(dataset_constructor_keys).union(extraneous_keys) and
            k not in {'streams', 'packing_ratio', 'replication'}
        }

        streaming_dataset = dataset_constructor.build_from_streaming(
            tokenizer=tokenizer,
            streams=streams,
            batch_size=dataloader_batch_size,
            replication=replication_factor,
            packing_ratio=dataloader_batch_size / dataset_batch_size,
            **dataset_constructor_args,
        )

    else:
        # Build HF dataloader
        dataset_name_or_path = dataset_cfg['hf_name']
        split = dataset_cfg.get('split')
        if split is None:
            raise MissingHuggingFaceURLSplitError()

        # If dataset is a remote path, download it first.
        backend, _, _ = parse_uri(dataset_name_or_path)
        if backend not in ['', None]:
            dataset_name_or_path = _download_remote_hf_dataset(
                remote_path=dataset_name_or_path,
                split=split,
            )
            split = split.replace('-', '_')

        # Get the preprocessing function.
        proto_preprocessing_fn = dataset_cfg.get('preprocessing_fn')
        if isinstance(proto_preprocessing_fn, (dict, DictConfig)):
            preprocessing_fn = dataset_constructor.get_preprocessing_fn_from_dict(
                dict(proto_preprocessing_fn),
            )
        else:
            preprocessing_fn = dataset_constructor.get_preprocessing_fn_from_str(
                proto_preprocessing_fn,
                dataset_name_or_path,
            )

        # Take the constructor args from above, minus args that have been created separately
        dataset_constructor_args = {
            k: v
            for k, v in dataset_cfg.items()
            if k in dataset_constructor_keys and
            k not in {'split', 'preprocessing_fn'}
        }
        streaming_dataset = dataset_constructor.build_from_hf(
            dataset_name=dataset_name_or_path,
            split=split,
            preprocessing_fn=preprocessing_fn,
            tokenizer=tokenizer,
            **dataset_constructor_args,
        )

        # Ensure dataset is large enough.
        if drop_last:
            world_size = dist.get_world_size() // replication_factor
            minimum_dataset_size = world_size * dataloader_batch_size
            if hasattr(streaming_dataset, '__len__'):
                full_dataset_size = len(streaming_dataset)
                if full_dataset_size < minimum_dataset_size:
                    raise NotEnoughDatasetSamplesError(
                        dataset_name=dataset_cfg['hf_name'],
                        split=split,
                        dataloader_batch_size=dataloader_batch_size,
                        world_size=world_size,
                        full_dataset_size=full_dataset_size,
                        minimum_dataset_size=minimum_dataset_size,
                    )

        # Initialize sampler.
        sampler = dist.get_sampler(
            streaming_dataset,
            drop_last=drop_last,
            shuffle=dataset_cfg['shuffle'],
            num_replicas=dist.get_world_size() //
            replication_factor if replication_factor > 1 else None,
            rank=dist.get_global_rank() //
            replication_factor if replication_factor > 1 else None,
            seed=dataset_cfg.get('shuffle_seed', 0),
        )

    assert streaming_dataset is not None  # for pyright
    dl = DataLoader(
        streaming_dataset,
        collate_fn=collate_fn,
        batch_size=dataloader_batch_size,
        drop_last=drop_last,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        timeout=timeout,
    )

    return construct_from_registry(
        name='data_spec',
        registry=registry.data_specs,
        partial_function=False,
        kwargs={
            'dl': dl,
            'dataset_cfg': dataset_cfg,
        },
    )


def _validate_config(
    max_seq_len: int,
    decoder_only_format: Optional[bool] = None,
    hf_name: Optional[str] = None,
    local: Optional[str] = None,
    remote: Optional[str] = None,
    hf_kwargs: Optional[dict[str, Any]] = None,
    preprocessing_fn: Optional[str] = None,
    safe_load: Optional[bool] = None,
    streams: Optional[dict[str, Any]] = None,
    target_prompts: Optional[str] = None,
    target_responses: Optional[str] = None,
    allowed_dataset_keys: set[str] = _ALLOWED_DATASET_KEYS,
    **kwargs: dict[str, Any],
) -> set[str]:
    """Validates the dataset configuration.

    Makes sure that the dataset is properly configured for either
    a HuggingFace dataset or a streaming dataset. Must be valid for one or
    the other.

    Args:
        max_seq_len (int): The maximum length of sequences
            in the batch. See :class:`Seq2SeqFinetuningCollator` docstring
            for details.
        decoder_only_format (bool, optional): Whether to format the
            examples for a decoder-only model. See :class:`Seq2SeqFinetuningCollator`
            docstring for details.
        hf_name (str, optional): The name of the HuggingFace dataset
            to use. Can also be a remote http(s) directory or object store bucket
            containing the file {split}.jsonl in the format (prompt, response),
            in which case the builder will create a HuggingFace dataset.
        local (str, optional): Local path where remote data
            will be streamed to. Only valid if `cfg.dataset.remote` has
            also been set.
        remote (str, optional): Location of a MDS-formatted
            streaming dataset to use. Setting this will tell the builder
            to create a streaming dataset rather than a HuggingFace dataset.
        hf_kwargs (DictConfig, optional): Additional kwargs to
            pass to `datasets.load_dataset`, which can be used to load
            a dataset from local files.
        preprocessing_fn (str, optional): The name/import path of
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
        safe_load (bool, optional): Whether to enforce safe loading of the dataset.
            If `None`, will default to not applying any safe loading.
        streams (Dict[str, Any], optional): A dictionary with multiple data streams.
            If `None`, will assume no streams.
        target_prompts (str): Which prompts are used as training targets.
            Defaults to "none", meaning prompts are never used as training targets.
            See :class:`Seq2SeqFinetuningCollator` docstring for details.
        target_responses (str): Which responses are used as training targets.
            Defaults to "last", meaning only the final response in multi-turn examples
            will serve as training targets. See :class:`Seq2SeqFinetuningCollator` docstring for
            details.
        allowed_dataset_keys (set[str], optional): The set of allowed keys for the dataset config.
        kwargs (DictConfig, optional): Additional kwargs to
                pass to `datasets.load_dataset`, which can be used to load
                a dataset from local files.

    Raises:
        ValueError: If the dataset configuration does not meet the requirements.

    Returns:
        set[str]: Return the extraneous keys.
    """
    if decoder_only_format is None:
        raise ValueError(
            f'decoder_only_format must be set to either True or False, but it was {decoder_only_format}.',
        )

    extraneous_keys = set()
    if not set(kwargs.keys()).issubset(allowed_dataset_keys):
        extraneous_keys = set(kwargs.keys()) - allowed_dataset_keys
        log.warning(
            'The dataset config contains the following extraneous keys: ' +\
            ', '.join(extraneous_keys),
        )

    if hf_name is not None:
        # Using the HuggingFace dataset codepath
        illegal_keys = ['local', 'remote']
        discovered_illegal_keys = []
        if local is not None:
            discovered_illegal_keys.append('`local`')
        if remote is not None:
            discovered_illegal_keys.append('`remote`')
        if discovered_illegal_keys:
            raise ValueError(
                'The dataset config sets a value for `hf_name` as well as the ' +\
                f'following keys: {", ".join(discovered_illegal_keys)}.\n' +\
                'Those keys are used when building from a streaming dataset, but ' +\
                'setting `hf_name` instructs the dataset to build from a HuggingFace dataset.',
            )
    elif remote is not None or local is not None:
        # Using the streaming dataset codepath
        illegal_keys = {
            'hf_name': hf_name,
            'hf_kwargs': hf_kwargs,
            'preprocessing_fn': preprocessing_fn,
            'safe_load': safe_load,
        }
        discovered_illegal_keys = []
        for key, value in illegal_keys.items():
            if value is not None:
                discovered_illegal_keys.append('`' + key + '`')
        if discovered_illegal_keys:
            raise ValueError(
                'The dataset config sets a value for `remote` as well as the ' +\
                f'following keys: {", ".join(discovered_illegal_keys)}.\n' +\
                'Those keys are used when building from a HuggingFace dataset, but ' +\
                'setting `remote` instructs the dataset to build from a streaming dataset.',
            )
        if local is None:
            raise ValueError(
                'Using a streaming dataset requires setting both `remote` and `local`, ' +\
                'but dataset.local is None.',
            )
    elif streams is not None:
        # Using the streaming dataset codepath
        illegal_keys = {
            'hf_name': hf_name,
            'hf_kwargs': hf_kwargs,
            'preprocessing_fn': preprocessing_fn,
            'safe_load': safe_load,
        }
        discovered_illegal_keys = []
        for key, value in illegal_keys.items():
            if value is not None:
                discovered_illegal_keys.append('`' + key + '`')
        if discovered_illegal_keys:
            raise ValueError(
                'The dataset config sets a value for `streams` as well as the ' +\
                f'following keys: {", ".join(discovered_illegal_keys)}.\n' +\
                'Those keys are used when building from a HuggingFace dataset, but ' +\
                'setting `streams` instructs the dataset to build from a streaming dataset.',
            )
        illegal_keys = {'remote': remote, 'local': local}
        discovered_illegal_keys = []
        for key, value in illegal_keys.items():
            if value is not None:
                discovered_illegal_keys.append('`' + key + '`')
        if discovered_illegal_keys:
            raise ValueError(
                'The dataset config sets a value for `streams` as well as the ' +\
                f'following keys: {", ".join(discovered_illegal_keys)}.\n' +\
                'Please either use single stream (set remote/local only) ' +\
                'or put remote/local under streams',
            )

    else:
        raise ValueError(
            'In the dataset config, you must set `hf_name` to use a HuggingFace ' +\
            'dataset, or set `remote` to use a streaming dataset, or set ' +\
            '`streams` to use multiple streaming datasets,  but all were None.',
        )

    # Raise an error if the target_prompts + target_responses + decoder_only_format settings
    # are invalid
    if target_prompts is None:
        target_prompts = DEFAULT_TARGET_PROMPTS
    if target_responses is None:
        target_responses = DEFAULT_TARGET_RESPONSES
    target_prompts, target_responses = target_prompts.lower(
    ), target_responses.lower()
    validate_target_settings(
        target_prompts,
        target_responses,
        decoder_only_format,
    )

    return extraneous_keys


def _download_remote_hf_dataset(remote_path: str, split: str) -> str:
    """Downloads a dataset from a remote object store.

    This function supports 'jsonl', 'csv', and 'parquet' file formats for the dataset. It will attempt to download
    the dataset, then once it is downloaded, convert it into HuggingFace ``datasets`` format, and then return this
    dataset.

    The function also ensures synchronicity across multiple processes during the file download. It creates a signal
    file that is used to synchronize the start of the download across different processes. Once the download is
    completed, the function removes the signal file.

    Args:
        remote_path (str): The path of the HuggingFace dataset to download.
        split (str): The dataset split to download (e.g., 'train', 'validation', 'test').

    Returns:
        A local directory path where the dataset files are stored.

    Raises:
        FileNotFoundError: Raised if the dataset file cannot be found with any of the supported extensions.
    """
    # HF datasets does not support a split with dashes, so we replace dashes with underscores.
    hf_formatted_split = split.replace('-', '_')
    finetune_dir = os.path.join(
        dist_mkdtemp(),
        hf_formatted_split if hf_formatted_split != 'data' else 'data_not',
    )
    os.makedirs(finetune_dir, exist_ok=True)
    for extension in SUPPORTED_EXTENSIONS:
        name = f'{remote_path.strip("/")}/{split}{extension}'
        destination = str(
            os.path.abspath(
                os.path.join(
                    finetune_dir,
                    'data',
                    f'{hf_formatted_split}-00000-of-00001{extension}',
                ),
            ),
        )

        # Since we don't know exactly what the extension will be, since it is one of a list
        # use a signal file to wait for instead of the desired file
        signal_file_path = os.path.join(
            finetune_dir,
            f'.node_{dist.get_node_rank()}_local_rank0_completed',
        )

        log.debug(f'Downloading dataset {name} to {destination}.')
        if dist.get_local_rank() == 0:
            try:
                get_file(path=name, destination=destination, overwrite=True)
            except FileNotFoundError as e:
                if extension == SUPPORTED_EXTENSIONS[-1]:
                    files_searched = [
                        f'{name}/{split}{ext}' for ext in SUPPORTED_EXTENSIONS
                    ]
                    raise FinetuningFileNotFoundError(
                        files_searched=files_searched,
                        supported_extensions=SUPPORTED_EXTENSIONS,
                    ) from e
                else:
                    log.debug(
                        f'Could not find {name}, looking for another extension',
                    )
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
    dataloader_cfg: dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    device_batch_size: int,
) -> tuple[Union[Seq2SeqFinetuningCollator, BinPackCollator], int]:
    # These `.get` calls are safe because the dataset_cfg is validated for extra keys
    dataset_cfg = dataloader_cfg['dataset']
    target_responses = dataset_cfg.get(
        'target_responses',
        DEFAULT_TARGET_RESPONSES,
    )
    target_prompts = dataset_cfg.get('target_prompts', DEFAULT_TARGET_PROMPTS)
    max_seq_len = dataset_cfg['max_seq_len']
    decoder_only_format = dataset_cfg['decoder_only_format']
    allow_pad_trimming = dataset_cfg.get('allow_pad_trimming', False)
    pad_to_longest = dataset_cfg.get('pad_to_longest', False)

    collate_fn = Seq2SeqFinetuningCollator(
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        decoder_only_format=decoder_only_format,
        target_responses=target_responses,
        target_prompts=target_prompts,
        allow_pad_trimming=allow_pad_trimming,
        pad_to_longest=pad_to_longest,
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
        packing_ratio = auto_packing_ratio(
            dataloader_cfg=dataloader_cfg,
            tokenizer=tokenizer,
            device_batch_size=device_batch_size,
        )

    if isinstance(packing_ratio, str):
        raise ValueError(
            'dataset.packing_ratio must be a float or "auto", but it was set to '
            + f'{packing_ratio}.',
        )

    log.info(f'Using packing ratio {packing_ratio}')

    if packing_ratio == 1.0:
        return collate_fn, device_batch_size
    elif packing_ratio < 1.0:
        raise ValueError('packing_ratio must be >= 1, if supplied')

    if not decoder_only_format:
        raise NotImplementedError(
            'On-the-fly packing is currently only supported for decoder-only formats.',
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
        'timeout': 0,
    })

    tokenizer_name = 'EleutherAI/gpt-neox-20b'
    tokenizer_kwargs = {'model_max_length': cfg.dataset.max_seq_len}
    tokenizer = build_tokenizer(tokenizer_name, tokenizer_kwargs)

    device_batch_size = 1
    dataloader = build_finetuning_dataloader(
        **cfg,
        tokenizer=tokenizer,
        device_batch_size=device_batch_size,
    ).dataloader

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
                            tokenizer.decode(
                                batch['input_ids'][
                                    j,
                                    torch.logical_and(
                                        is_subseq,
                                        batch['attention_mask'][j] == 1,
                                    )],
                                skip_special_tokens=False,
                                clean_up_tokenization_spaces=True,
                            ),
                        )
                        context = torch.logical_and(
                            batch['attention_mask'][j] == 1,
                            batch['labels'][j] == CROSS_ENTROPY_IGNORE_INDEX,
                        )
                        print(
                            '\033[92m{}\033[00m\n'.format('CONTEXT:  '),
                            tokenizer.decode(
                                batch['input_ids'][
                                    j, torch.logical_and(is_subseq, context)],
                                skip_special_tokens=False,
                                clean_up_tokenization_spaces=True,
                            ),
                        )
                        print(
                            '\033[91m{}\033[00m\n'.format('TARGET:   '),
                            tokenizer.decode(
                                batch['input_ids'][
                                    j,
                                    torch.logical_and(
                                        is_subseq,
                                        batch['labels'][j] !=
                                        CROSS_ENTROPY_IGNORE_INDEX,
                                    )],
                                skip_special_tokens=False,
                                clean_up_tokenization_spaces=True,
                            ),
                        )
                else:
                    print(
                        '\033[93m{}\033[00m\n'.format('INPUT IDS:'),
                        tokenizer.decode(
                            batch['input_ids'][j,
                                               batch['attention_mask'][j] == 1],
                            skip_special_tokens=False,
                            clean_up_tokenization_spaces=True,
                        ),
                    )
                    context = torch.logical_and(
                        batch['attention_mask'][j] == 1,
                        batch['labels'][j] == CROSS_ENTROPY_IGNORE_INDEX,
                    )
                    print(
                        '\033[92m{}\033[00m\n'.format('CONTEXT:  '),
                        tokenizer.decode(
                            batch['input_ids'][j, context],
                            skip_special_tokens=False,
                            clean_up_tokenization_spaces=True,
                        ),
                    )
                    print(
                        '\033[91m{}\033[00m\n'.format('TARGET:   '),
                        tokenizer.decode(
                            batch['input_ids']
                            [j,
                             batch['labels'][j] != CROSS_ENTROPY_IGNORE_INDEX],
                            skip_special_tokens=False,
                            clean_up_tokenization_spaces=True,
                        ),
                    )
            else:
                print(
                    '\033[92m{}\033[00m\n'.format('CONTEXT:  '),
                    tokenizer.decode(
                        batch['input_ids'][j, batch['attention_mask'][j] == 1],
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=True,
                    ),
                )
                print(
                    '\033[91m{}\033[00m\n'.format('TARGET:   '),
                    tokenizer.decode(
                        batch['labels'][j, batch['decoder_attention_mask'][j] ==
                                        1],
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=True,
                    ),
                )
        print('   ')
