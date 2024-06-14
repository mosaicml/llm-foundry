# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any, Callable, Dict, Iterable, Mapping, Tuple, Union

import torch
import transformers
from composer.core.data_spec import DataSpec
from composer.core.types import Batch
from torch.utils.data import DataLoader as TorchDataloader
from transformers import PreTrainedTokenizerBase

from llmfoundry.data.finetuning.collator import Seq2SeqFinetuningCollator
from llmfoundry.data.finetuning.dataloader import build_collate_fn
from llmfoundry.data.packing import BinPackCollator
from llmfoundry.data.text_data import ConcatenatedSequenceCollatorWrapper

log = logging.getLogger(__name__)


def _validate_cfg(
    dataset_cfg: Dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
):
    eos_token_id = dataset_cfg.get('eos_token_id', None)
    bos_token_id = dataset_cfg.get('bos_token_id', None)

    tokenizer_eos_token_id = getattr(tokenizer, 'eos_token_id', None)
    if eos_token_id is not None and eos_token_id != tokenizer_eos_token_id:
        eos_mismatch_str = f'Provided {eos_token_id=} does not match the eos_token_id of the tokenizer={tokenizer_eos_token_id}.'
        if dataset_cfg.pop('override_eos_token_id_mismatch_error', False):
            log.warning(eos_mismatch_str)
        else:
            raise ValueError(
                eos_mismatch_str +
                ' To override this error, set the override_eos_token_id_mismatch_error flag to True in the dataset config section of the YAML.',
            )

    tokenizer_bos_token_id = getattr(tokenizer, 'bos_token_id', None)
    if bos_token_id is not None and bos_token_id != tokenizer_bos_token_id:
        bos_mismatch_str = f'Provided {bos_token_id=} does not match the bos_token_id of the tokenizer={tokenizer_bos_token_id}.'
        if dataset_cfg.pop('override_bos_token_id_mismatch_error', False):
            log.warning(bos_mismatch_str)
        else:
            raise ValueError(
                bos_mismatch_str +
                ' To override this error, set the override_bos_token_id_mismatch_error flag to True in the dataset config section of the YAML.',
            )

    max_seq_len = dataset_cfg.get('max_seq_len')
    if max_seq_len is not None:
        if max_seq_len != int(max_seq_len):
            raise ValueError('max_seq_len must be an integer')
        dataset_cfg['max_seq_len'] = int(max_seq_len)


def validate_ds_replication(
    dataset_cfg: Dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    device_batch_size: Union[int, float],
) -> Tuple[int, int]:
    _validate_cfg(dataset_cfg, tokenizer)
    if (dataset_cfg.get('seq_parallel_replication', 1) or 1) > 1:
        raise NotImplementedError('Sequence parallelism is not supported.')
    if not isinstance(device_batch_size, int):
        raise ValueError('device_batch_size should be integer.')
    return dataset_cfg.get('replication', 1) or 1, device_batch_size


def get_data_spec(
    dl: Union[Iterable, TorchDataloader],
    dataset_cfg: Dict[str, Any],
) -> DataSpec:
    del dataset_cfg
    token_counting_func = get_tokens_per_batch_func()

    return DataSpec(
        dataloader=dl,
        get_num_tokens_in_batch=token_counting_func,
    )


def get_tokens_per_batch_func(
    decoder_only: bool = True,
) -> Callable[[Batch], int]:
    """Returns a callable that counts the number of tokens in a batch.

    Args:
        pad_token_id (int): The id of the padding token.
        decoder_only (bool, optional): Whether to expect the batch to just contain ``input_ids`` (decoder only)
            or to also contain ``decoder_input_ids`` (encoder decoder). Defaults to ``True``.

    Returns:
        Callable[[Batch], int]: A callable that counts the number of tokens in a batch.
    """

    def get_num_tokens_in_batch(batch: Batch) -> int:
        if not isinstance(batch, Mapping) or (
            'attention_mask' not in batch and 'input_ids' not in batch
        ):
            raise ValueError(
                'get_tokens_per_batch_func() requires a batch with an attention_mask key or an input_ids key',
            )

        if not decoder_only and 'decoder_attention_mask' not in batch:
            raise ValueError(
                'get_tokens_per_batch_func() for encoder decoder requires a batch with a decoder_attention_mask key',
            )

        # Count number of non padding tokens in batch
        if 'attention_mask' in batch:
            input_ids_tokens = int(torch.sum(batch['attention_mask']).item())
        else:
            input_ids_tokens = batch['input_ids'].numel()

        # For encoder decoder models only
        decoder_input_ids_tokens = 0
        if not decoder_only:
            decoder_input_ids_tokens = int(
                torch.sum(batch['decoder_attention_mask']).item(),
            )

        return input_ids_tokens + decoder_input_ids_tokens

    return get_num_tokens_in_batch


def get_text_collator(
    dataloader_cfg: Dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    dataset_batch_size: int,
) -> Tuple[Union[transformers.DataCollatorForLanguageModeling,
                 ConcatenatedSequenceCollatorWrapper], int]:
    dataset_cfg = dataloader_cfg.get('dataset')
    assert isinstance(dataset_cfg, dict)
    eos_token_id = dataset_cfg.get('eos_token_id', None)
    bos_token_id = dataset_cfg.get('bos_token_id', None)
    mlm_probability = dataset_cfg.pop('mlm_probability', None)
    collate_fn = transformers.DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=mlm_probability is not None,
        mlm_probability=mlm_probability,
    )

    if (eos_token_id is not None) or (bos_token_id is not None):
        # Note: Will raise an error if both are non-None
        collate_fn = ConcatenatedSequenceCollatorWrapper(
            base_collator=collate_fn,
            eos_token_id=eos_token_id,
            bos_token_id=bos_token_id,
        )

    return collate_fn, dataset_batch_size


def get_finetuning_collator(
    dataloader_cfg: Dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    dataset_batch_size: int,
) -> Tuple[Union[Seq2SeqFinetuningCollator, BinPackCollator], int]:
    return build_collate_fn(dataloader_cfg, tokenizer, dataset_batch_size)
