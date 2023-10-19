

import os
from typing import Dict, List
from llmfoundry.data.packing import BinPackCollator
from omegaconf import DictConfig
from pytest import approx
import torch
from composer.utils import dist

from tests.data_utils import make_tiny_ft_dataset

from llmfoundry.data.finetuning.dataloader import build_finetuning_dataloader

from llmfoundry.utils.builders import build_tokenizer

def _data_to_batch(data: List[int], max_seq_len: int, pad_token_id: int, ) -> Dict[str, torch.Tensor]:
    input_ids = torch.stack([
        torch.tensor(d + [pad_token_id] * (max_seq_len - len(d)))
        for d in data
    ])

    attention_mask = torch.stack([
        torch.tensor([1] * len(d) + [pad_token_id] * (max_seq_len - len(d)))
        for d in data
    ])
    return { 'input_ids': input_ids, 'attention_mask': attention_mask }

def test_simple_packing():
    pad_token_id = 0
    max_seq_len = 5
    pack = BinPackCollator(
        collator=lambda x: x,
        target_batch_size=2,
        max_seq_len=max_seq_len, 
        pad_token_id=pad_token_id,
        padding_side = 'right'
    )

    batch = _data_to_batch([
        [1],
        [2] * 2,
        [4] * 4,
        [3] * 3,
    ], max_seq_len, pad_token_id)

    packed_samples = pack(batch)

    assert torch.equal(packed_samples['input_ids'], torch.Tensor([[3,3,3,2,2],[4,4,4,4,1]]))
    assert torch.all(packed_samples['attention_mask'] == 1)


def test_simple_packing_leftovers():
    pad_token_id = 0
    max_seq_len = 5
    pack = BinPackCollator(
        collator=lambda x: x,
        target_batch_size=2,
        max_seq_len=max_seq_len, 
        pad_token_id=pad_token_id,
        padding_side = 'right'
    )

    batch = _data_to_batch([
        [1],
        [2] * 2,
        [4] * 4,
        [4] * 4,
    ], max_seq_len, pad_token_id)

    packed_batch = pack(batch)

    assert torch.equal(packed_batch['input_ids'], torch.Tensor([[4,4,4,4,1],[4,4,4,4,0]]))
    assert torch.equal(packed_batch['attention_mask'], torch.Tensor([[1,1,1,1,1],[1,1,1,1,0]]))

    # Check leftovers and waste.
    assert len(pack._leftover_bins) == 1
    leftover_size, leftover = pack._leftover_bins[0]
    assert leftover_size == 2
    assert torch.equal(leftover['input_ids'], torch.Tensor([2,2]))
    assert torch.equal(leftover['attention_mask'], torch.Tensor([1,1]))
    assert pack.waste == approx(2/11) # 2 tokens wasted of 11 tokens total

    # Ensure that leftovers are used in the next batch if possible.
    batch = _data_to_batch([[1]], max_seq_len, pad_token_id)
    packed_batch = pack(batch)
    assert torch.equal(packed_batch['input_ids'], torch.Tensor([[2,2,0,0,0],[1,0,0,0,0]]))
    assert torch.equal(packed_batch['attention_mask'], torch.Tensor([[1,1,0,0,0],[1,0,0,0,0]]))

# def test_auto_packing():
#     max_seq_len = 2048
#     tiny_dataset_folder_path = os.path.join(os.getcwd(), 'test-ift-data-small')
#     tiny_dataset_path = os.path.join(tiny_dataset_folder_path, 'train.jsonl')
#     if dist.get_global_rank() == 0:
#         make_tiny_ft_dataset(path=tiny_dataset_path, size=100)

#     cfg = DictConfig({
#         'name': 'finetuning',
#         'dataset': {
#             'hf_name': tiny_dataset_folder_path,
#             'split': 'train',
#             'max_seq_len': max_seq_len,
#             'decoder_only_format': True,
#             'allow_pad_trimming': False,
#             'packing_ratio': 'auto',
#             'shuffle': True,
#         },
#         'drop_last': False,
#         'num_workers': 4,
#         'pin_memory': False,
#         'prefetch_factor': 2,
#         'persistent_workers': False,
#         'timeout': 0
#     })

#     tokenizer = build_tokenizer('mosaicml/mpt-7b', {})

#     dataloader = build_finetuning_dataloader(cfg, tokenizer, 2)


# def test_auto_packing():
#     dataloader_cfg = DictConfig({
#         'name': 'finetuning',
#         'dataset': {
#             'hf_name': 'mosaicml/dolly_hhrlhf',
#             'split': 'train',
#             'max_seq_len': 1024,
#             'allow_pad_trimming': False,
#             'decoder_only_format': True,
#             'packing_ratio': 'auto',
#             'shuffle': True,
#         },
#         'drop_last': False,
#         'num_workers': 8,
#         'pin_memory': False,
#         'prefetch_factor': 2,
#         'persistent_workers': True,
#         'timeout': 0,
#     })

#     tokenizer = build_tokenizer('mosaicml/mpt-7b', {})

#     dataloader = build_finetuning_dataloader(dataloader_cfg, tokenizer, 6)
