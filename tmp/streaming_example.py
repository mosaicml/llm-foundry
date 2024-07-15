
# import pandas as pd
# import numpy as np
# import torch
from torch.utils.data import DataLoader#, Dataset
# import torchvision.transforms as transforms
from composer.core.data_spec import DataSpec
# from typing import Union, List, Optional
# import os
import inspect
# from itertools import islice
from typing import Any, Dict
# from typing import Callable, Mapping, Optional, Sequence, Union, Tuple, cast
from transformers import PreTrainedTokenizerBase#, PreTrainedTokenizer

from llmfoundry import registry
from llmfoundry.data import StreamingTextDataset#, StreamingTimeSeriesDataset
# from llmfoundry.data import SUPPORTED_MDS_ENCODING_TYPES, stream_remote_local_validate
from llmfoundry.data.text_data import build_streams
from llmfoundry.utils.registry_utils import construct_from_registry
from llmfoundry.tokenizers import ChronosTokenizerWrapper

# from streaming import Stream
# from streaming import Stream, StreamingDataset
# from streaming.base import MDSWriter




tokenizer = ChronosTokenizerWrapper('amazon/chronos-t5-small')
# dataset_cfg = {
#     'tokenizer': tokenizer,
#     'max_seq_len': 30,
#     'remote': '/mnt/workdisk/kushal/llm-foundry/kushal-testing/mds_single_col_small/',
# }

# # ts_dataset = StreamingTimeSeriesDataset(**dataset_cfg)
# text_ds = StreamingTextDataset(**dataset_cfg)
# # for i in range(text_ds.num_samples):
# #     print(text_ds[i])


# # TODO: See how to define these values (currently taken from t5-small_dolly_sft yaml)
# dataloader_cfg = {
#   'name': 'text',
#   'dataset': dataset_cfg,
#   'drop_last': True,
#   'num_workers': 8,
#   'pin_memory': False,
#   'prefetch_factor': 2,
#   'persistent_workers': True,
#   'timeout': 0,
# }
# # dl = DataLoader(
# #         text_dataset,
# #         collate_fn=collate_fn,
# #         batch_size=dataloader_batch_size,
# #         drop_last=drop_last,
# #         num_workers=num_workers,
# #         pin_memory=pin_memory,
# #         prefetch_factor=prefetch_factor,
# #         persistent_workers=persistent_workers,
# #         timeout=timeout,
# #     )

# # TODO: Figure out where to get ``dataset_batch_size`` from
# dataset_batch_size = 8
# collate_fn, dataloader_batch_size = construct_from_registry(
#     name='text_collator',
#     registry=registry.collators,
#     partial_function=False,
#     kwargs={
#         'dataloader_cfg': dataloader_cfg,
#         'tokenizer': tokenizer,
#         'dataset_batch_size': dataset_batch_size,
#     },
# )

# dl = DataLoader(dataset=text_ds)

# construct_from_registry(
#     name='data_spec',
#     registry=registry.data_specs,
#     partial_function=False,
#     kwargs={
#         'dl': dl,
#         'dataset_cfg': dataset_cfg,
#     },
# )

# print('SUCCESSFUL EXECUTION!!!')

import os
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'

text_ds = StreamingTextDataset(tokenizer=tokenizer, max_seq_len=30, 
                               remote='/mnt/workdisk/kushal/llm-foundry/kushal-testing/mds_single_col_small/', 
                               local='/mnt/workdisk/kushal/llm-foundry/kushal-testing/test-data/')
for i in range(text_ds.num_samples):
    print(text_ds[i])
    print(type(text_ds[i]))



# train_loader = build_dataloader(
#     train_loader_config,
#     tokenizer,
#     train_cfg.device_train_batch_size,
# )

# train_loader_config contains these entries:
#   name: finetuning
#   dataset:
#     hf_name: HuggingFaceH4/databricks_dolly_15k
#     split: train
#     max_seq_len: ${variables.max_seq_len}
#     allow_pad_trimming: false
#     decoder_only_format: false
#     shuffle: true
#   drop_last: true
#   num_workers: 8
#   pin_memory: false
#   prefetch_factor: 2
#   persistent_workers: true
#   timeout: 0

# When we want to build a text dataloader, the method build_text_dataloader() gets called from registry
def build_text_dataloader(
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

    dataset_cfg = dataset

    # get kwargs
    dataset_cfg['replication'], dataset_batch_size = construct_from_registry(
        name='dataset_replication_validator',
        registry=registry.dataset_replication_validators,
        partial_function=False,
        kwargs={
            'dataset_cfg': dataset_cfg,
            'tokenizer': tokenizer,
            'device_batch_size': device_batch_size,
        },
    )

    streams = build_streams(
        streams=dataset_cfg.pop('streams')
        if 'streams' in dataset_cfg else None,
    )

    valid_streaming_text_dataset_parameters = inspect.signature(
        StreamingTextDataset,
    ).parameters

    dataset_config_subset_for_streaming_text_dataset = {
        k: v
        for k, v in dataset_cfg.items()
        if k in valid_streaming_text_dataset_parameters
    }

    # build dataset potentially with streams
    text_dataset = StreamingTextDataset(
        tokenizer=tokenizer,
        streams=streams,
        batch_size=dataset_batch_size,
        **dataset_config_subset_for_streaming_text_dataset,
    )

    dataloader_cfg = {
        'name': 'text',
        'dataset': dataset_cfg,
        'drop_last': drop_last,
        'num_workers': num_workers,
        'pin_memory': pin_memory,
        'prefetch_factor': prefetch_factor,
        'persistent_workers': persistent_workers,
        'timeout': timeout,
    }

    collate_fn, dataloader_batch_size = construct_from_registry(
        name='text_collator',
        registry=registry.collators,
        partial_function=False,
        kwargs={
            'dataloader_cfg': dataloader_cfg,
            'tokenizer': tokenizer,
            'dataset_batch_size': dataset_batch_size,
        },
    )

    dl = DataLoader(
        text_dataset,
        collate_fn=collate_fn,
        batch_size=dataloader_batch_size,
        drop_last=drop_last,
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
