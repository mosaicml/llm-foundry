
"""Build a StreamingTimeSeriesDataset dataset and dataloader for training."""

# import inspect
# from itertools import islice
from typing import (
    Any,
    Callable,
    Dict,
    # Mapping,
    Optional,
    Sequence,
    Union,
    Tuple,
    # cast,
)
import numpy as np
# import pandas as pd
import torch
from composer.core.data_spec import DataSpec
from streaming import Stream, StreamingDataset
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

from llmfoundry import registry
from llmfoundry.data import (
    # SUPPORTED_MDS_ENCODING_TYPES,
    stream_remote_local_validate,
)
from llmfoundry.utils.registry_utils import construct_from_registry

# New import statements
from llmfoundry.tokenizers import ChronosTokenizerWrapper
from llmfoundry.data.chronos_dataset import ChronosDataset, has_enough_observations
from gluonts.itertools import Filter
from gluonts.dataset.common import FileDataset
from functools import partial
from pathlib import Path

__all__ = [
    # 'StreamingTimeSeriesDataset',
    'build_timeseries_dataloader',
]


class StreamingTimeSeriesDataset(StreamingDataset):
    """Generic time series dataset using MosaicML's StreamingDataset.
    
    Args:
        tokenizer (Tokenizer): HuggingFace tokenizer to
            tokenize samples.
        ...
        max_seq_len (int): The number of time step values to consider for each sample (or time series).
        token_encoding_type (str): The encoding type of the tokenized samples. This is only used
            for legacy datasets that have been written directly as 'bytes' instead of numpy
            arrays. Types are auto-inferred for numpy arrays. Defaults to 'int64'.
        streams (Sequence[Stream], optional): One or more Streams to stream/cache samples from,
            which may be upsampled or downsampled. StreamingDataset uses either ``streams`` or
            ``remote``/``local``. Defaults to ``None``.
        remote (str, optional): Remote path or directory to download the dataset from. If ``None``,
            its data must exist locally. StreamingDataset uses either ``streams`` or
            ``remote``/``local``. Defaults to ``None``.
        local (str, optional): Local working directory to download shards to. This is where shards
            are cached while they are being used. Uses a temp directory if not set.
            StreamingDataset uses either ``streams`` or ``remote``/``local``. Defaults to ``None``.
        split (str, optional): Which dataset split to use, if any. If provided, we stream from/to
            the ``split`` subdirs of  ``remote`` and ``local``. Defaults to ``None``.
        download_retry (int): Number of download re-attempts before giving up. Defaults to ``2``.
        download_timeout (float): Number of seconds to wait for a shard to download before raising
            an exception. Defaults to ``60``.
        validate_hash (str, optional): Optional hash or checksum algorithm to use to validate
            shards. Defaults to ``None``.
        keep_zip (bool): Whether to keep or delete the compressed form when decompressing
            downloaded shards. If ``False``, keep iff remote is local or no remote. Defaults to
            `False``.
        epoch_size (Union[int, str], optional): Number of samples to draw per epoch balanced across all
            streams. If ``None``, takes its value from the total number of underlying samples.
            Provide this field if you are weighting streams relatively to target a larger or
            smaller epoch size. Defaults to ``None``.
        predownload (int, optional): Target number of samples ahead to download the shards of while
            iterating. If ``None``, its value is set to ``8 * batch_size``. Defaults to ``None``.
        cache_limit (Union[int, str], optional) - Maximum size in bytes of this StreamingDataset's
            shard cache. Before downloading a shard, the least recently used resident shard(s) may
            be evicted (deleted from the local cache) in order to stay under the limit. Set to None
            to disable shard eviction. Supports integer bytes as well as string human-readable
            bytes (e.g., 100b, 64kb, 77mb, and so on). Defaults to None.
        partition_algo (str): Which partitioning algorithm to use. Defaults to ``orig``.
        num_canonical_nodes (int, optional): Canonical number of nodes for shuffling with
            resumption. If ``None``, this is interpreted as 64 times the number of physical
            nodes of the initial run if ``shuffle_algo`` is ``py1s`` or ``py2s``, and simply the
            number of physical nodes of the initial run otherwise. Defaults to ``None``.
        batch_size (int, optional): Batch size of its DataLoader, which affects how the dataset is
            partitioned over the workers. Defaults to ``None``.
        shuffle (bool): Whether to iterate over the samples in randomized order. Defaults to
            ``False``.
        shuffle_algo (str): Which shuffling algorithm to use. Defaults to ``py1e``.
        shuffle_seed (int): Seed for Deterministic data shuffling. Defaults to ``9176``.
        shuffle_block_size (int, optional): Unit of shuffle. A canonical node's samples are split
            into blocks of this size, and samples within each block are shuffled. If ``None``, its
            value is calculated as ``max(4_000_000 // num_canonical_nodes), 1 << 18)``. Defaults to
            ``None``.
        sampling_method (str): Which sampling method to use, either ``balanced`` or ``fixed``.
            Defaults to ``balanced``.
        sampling_granularity (int): When picking samples for a stream's final partial repeat,
            how many samples to pick from the same shard at a time (``1`` for evenly balanced
            across shards, ``1000`` to pick 1000 samples from the same shard at a time, etc).
            Defaults to ``1``.
        batching_method (str): Which batching method to use, either ``random``, ``stratified``, or
            ``per_stream``. Defaults to ``random``.
        allow_unsafe_types (bool): If a shard contains Pickle, which allows arbitrary code
            execution during deserialization, whether to keep going if ``True`` or raise an error
            if ``False``. Defaults to ``False``.
        replication (int, optional): Determines how many consecutive devices will receive the same
            samples. Useful for training with tensor or sequence parallelism, where multiple
            devices need to see the same partition of the dataset. Defaults to ``None``.
    """
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_seq_len: int,
        # token_encoding_type: str = 'int64',  # data will be passed in as np.ndarrays
        streams: Optional[Sequence[Stream]] = None,
        remote: Optional[str] = None,
        local: Optional[str] = None,
        split: Optional[str] = None,
        download_retry: int = 2,
        download_timeout: float = 60,
        validate_hash: Optional[str] = None,
        keep_zip: bool = False,
        epoch_size: Optional[Union[int, str]] = None,
        predownload: Optional[int] = None,
        cache_limit: Optional[Union[int, str]] = None,
        partition_algo: str = 'relaxed',
        num_canonical_nodes: Optional[int] = None,
        batch_size: Optional[int] = None,
        shuffle: bool = False,
        shuffle_algo: str = 'py1e',
        shuffle_seed: int = 9176,
        shuffle_block_size: Optional[int] = None,
        sampling_method: str = 'balanced',
        sampling_granularity: int = 1,
        batching_method: str = 'random',
        allow_unsafe_types: bool = False,
        replication: Optional[int] = None,
        transform: Optional[Callable] = None,
        **kwargs: Any,
    ):
        
        if len(kwargs) > 0:
            raise ValueError(
                f'StreamingTimeSeriesDataset() got an unexpected keyword argument: {kwargs}',
            )

        # if token_encoding_type not in SUPPORTED_MDS_ENCODING_TYPES:
        #     raise ValueError(
        #         f'The token_encoding_type must be one of {SUPPORTED_MDS_ENCODING_TYPES}, but got {token_encoding_type}',
        #     )
        # self.token_encoding_type = token_encoding_type
        
        if streams is None:
            stream_remote_local_validate(remote, local, split)
        else:
            for stream in streams:
                stream_remote_local_validate(
                    stream.remote,
                    stream.local,
                    split,
                )
        
        # discover where yamls are being converted incorrect, but temporary workaround
        if isinstance(shuffle_block_size, float):
            shuffle_block_size = int(shuffle_block_size)
        
        # Build Dataset
        super().__init__(
            streams=streams,
            remote=remote,
            local=local,
            split=split,
            download_retry=download_retry,
            download_timeout=download_timeout,
            validate_hash=validate_hash,
            keep_zip=keep_zip,
            epoch_size=epoch_size,
            predownload=predownload,
            cache_limit=cache_limit,
            partition_algo=partition_algo,
            num_canonical_nodes=num_canonical_nodes,
            batch_size=batch_size,
            shuffle=shuffle,
            shuffle_algo=shuffle_algo,
            shuffle_seed=shuffle_seed,
            shuffle_block_size=shuffle_block_size,
            sampling_method=sampling_method,
            sampling_granularity=sampling_granularity,
            batching_method=batching_method,
            allow_unsafe_types=allow_unsafe_types,
            replication=replication,
        )
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.transform = transform
        
        # TODO: Check that max_seq_len < # columns in dataset from ``remote``
        # df = ...  # TODO: implement how to upload from ``remote``
        # if max_seq_len > df.shape[1]:
        #     raise ValueError(f'Cannot have max_seq_len = {max_seq_len} for dataset that has {df.shape[1]} columns')
        
        
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        sample = super().__getitem__(idx)
        sample_data = sample['text']  # Expects column 'data' in ``sample``
        sample_data_tensor = torch.from_numpy(sample_data)
        token_ids, _ = self.tokenizer.encode(context=sample_data_tensor)
        return token_ids.squeeze()
        # return sample_data
        
        # TODO: Perform tokenization on ``sample_data``
        # local_path = '/mnt/workdisk/kushal/llm-foundry/kushal-testing/hospital.csv'  # temporary
        # df = pd.read_csv(local_path)
        # context = df.iloc[idx, :self.max_seq_len].to_numpy()
        # token_ids, tokenizer_state = self.tokenizer.encode(context=context)
        
        # if self.transform:
        #     token_ids = self.transform(token_ids)
        
        # return token_ids, tokenizer_state
        


def build_timeseries_dataloader(
    tokenizer: PreTrainedTokenizerBase,
    device_batch_size: int,
    dataset: Dict[str, Any],
    drop_last: bool,
    num_workers: int,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    persistent_workers: bool = True,
    timeout: int = 0,
) -> DataSpec:
    
    assert isinstance(tokenizer, ChronosTokenizerWrapper), "tokenizer is not of type ChronosTokenizerWrapper"
    
    dataset_cfg = dataset
    _validate_config(**dataset_cfg)
    
    # Use EOS as the pad token if none exists
    if tokenizer.pad_token is None:  # type: ignore (sometimes it's none and that's ok)
        tokenizer.pad_token = tokenizer.eos_token
    
    # dataset_batch_size = device_batch_size  # TODO: maybe change
    # dataset_config_subset_for_streaming_text_dataset = dataset_cfg  # TODO: maybe change
    
    # build dataset (removed the streams part)
    # timeseries_dataset = StreamingTimeSeriesDataset(
    #     tokenizer=tokenizer,
    #     # streams=streams,
    #     batch_size=dataset_batch_size,
    #     **dataset_config_subset_for_streaming_text_dataset,
    # )
    
    dataloader_cfg = {
        'name': 'timeseries',
        'dataset': dataset_cfg,
        'drop_last': drop_last,
        'num_workers': num_workers,
        'pin_memory': pin_memory,
        'prefetch_factor': prefetch_factor,
        'persistent_workers': persistent_workers,
        'timeout': timeout,
    }
    
    _, dataset_batch_size = construct_from_registry(
        name='dataset_replication_validator',
        registry=registry.dataset_replication_validators,
        partial_function=False,
        kwargs={
            'dataset_cfg': dataset_cfg,
            'tokenizer': tokenizer,
            'device_batch_size': device_batch_size,
        },
    )
    dataloader_batch_size = dataset_batch_size  # TODO: Might need to change this
    
    context_length = dataset_cfg.get('context_length', 512)
    prediction_length = dataset_cfg.get('prediction_length', 64)
    min_past = dataset_cfg.get('min_past', 60)
    max_missing_prop = dataset_cfg.get('max_missing_prop', 0.9)
    shuffle_buffer_length = dataset_cfg.get('shuffle_buffer_length', 100_000)
    training_data_paths = [dataset_cfg.get('remote')]  # Used to create `train_datasets`
    train_datasets = [
        Filter(
            partial(
                has_enough_observations,
                min_length=min_past + prediction_length,
                max_missing_prop=max_missing_prop,
            ),
            FileDataset(path=Path(data_path), freq="h"),
        )
        for data_path in training_data_paths
    ]
    shuffled_train_dataset = ChronosDataset(
        datasets=train_datasets, 
        probabilities=[1.0], 
        tokenizer=tokenizer.chronos_tokenizer, 
        context_length=context_length,
        prediction_length=prediction_length,
        min_past=min_past,
        mode="training", 
    ).shuffle(shuffle_buffer_length=shuffle_buffer_length)
    
    dl = DataLoader(
        dataset=shuffled_train_dataset,
        # collate_fn=collate_fn,
        batch_size=dataloader_batch_size,
        drop_last=drop_last,
        # sampler=sampler,  # `sampler` not defined
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
    context_length: int,
    prediction_length: int,
    min_past: int,
    shuffle_buffer_length: int,
    max_missing_prop: float,
    decoder_only_format: bool = False,
    hf_name: Optional[str] = None,
    local: Optional[str] = None,
    remote: Optional[str] = None,
    hf_kwargs: Optional[Dict[str, Any]] = None,
    preprocessing_fn: Optional[str] = None,
    safe_load: Optional[bool] = None,
    streams: Optional[Dict[str, Any]] = None,
    target_prompts: Optional[str] = None,
    target_responses: Optional[str] = None,
    **kwargs: Dict[str, Any],
) -> None:
    """Validates the dataset configuration.

    Makes sure that the dataset is properly configured for either
    a HuggingFace dataset or a streaming dataset. Must be valid for one or
    the other.

    Args:
        dataset_cfg (DictConfig): The dataset configuration to be validated.

    Raises:
        ValueError: If the dataset configuration does not meet the requirements.
    """
    # TODO: Implement this function



# Helpful to test if your dataloader is working locally
# Run `python data.py  --local_path [local] [--remote_path remote, optional]` and verify that batches are printed out
if __name__ == '__main__':
    # TODO: implement this, using llmfoundry/data/text_data.py as an example
    
    pass
