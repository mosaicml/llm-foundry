from typing import Dict, List
from unittest.mock import Mock, patch
from llmfoundry.data.packing import BinPackCollator, auto_packing_ratio
from omegaconf import DictConfig
from pytest import approx
import torch

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

def test_packing():
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

def test_packing_with_leftovers():
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

@patch('llmfoundry.data.packing.profile_packing')
def test_auto_packing(profile_packing: Mock):
    # List of tuples of packing_ratio, padding, waste, sorted by packing ratio
    profile_packing.return_value = [(1, .9, 0), (2, .8, 0), (3, .7, .5)]

    packing_ratio = auto_packing_ratio(
        dataloader_cfg=DictConfig({'dataset': {'max_seq_len': 2048 }}), 
        tokenizer=None, 
        device_batch_size=1,
    ) # Dummy values, profiling results are already set.

    # auto packing ratio should choose 2 because packing ratio is maximized while waste is 0.
    assert packing_ratio == 2
