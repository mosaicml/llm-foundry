from typing import Optional

import pytest
import torch
from unittest.mock import MagicMock

from llmfoundry.models.layers import blocks
from llmfoundry.models.layers.blocks import MPTBlock

def test_default_attention_mask_slicing():
    attention_mask = torch.tensor([1, 1, 0, 1]).byte()
    assert isinstance(attention_mask, torch.ByteTensor)

    block = MPTBlock(
        d_model=4,
        n_heads=1,
        expansion_ratio=1,
    )

    output_mask = block.slice_attention_mask(
        attention_mask=attention_mask,
        seq_len=4,
    )
    
    assert torch.equal(output_mask, attention_mask)

def test_attention_mask_slicing_called(monkeypatch: pytest.MonkeyPatch):
    m = torch.randn(2, 4, 4)
    attention_mask = torch.tensor([1, 1, 1, 1]).byte()
    dummy_return_mask = torch.tensor([1, 1, 1, 0]).byte()
    assert isinstance(attention_mask, torch.ByteTensor)
    assert isinstance(dummy_return_mask, torch.ByteTensor)
    indices = torch.arange(4)

    unpad_mock = MagicMock(return_value=(m, indices, None, None))
    pad_mock = MagicMock(return_value=m)
    monkeypatch.setattr(blocks, 'unpad_input', unpad_mock)
    monkeypatch.setattr(blocks, 'pad_input', pad_mock)
    class MPTBlockTest(MPTBlock):
        def slice_attention_mask(
            self,
            attention_mask: Optional[torch.ByteTensor],
            seq_len: int,
        ) -> Optional[torch.ByteTensor]:
            del seq_len
            del attention_mask
            return dummy_return_mask  # type: ignore
        
    block = MPTBlockTest(
        d_model=4,
        n_heads=1,
        expansion_ratio=1,
        use_pad_tok_in_ffn=False,
    )

    block.apply_ffn(
        attention_mask=attention_mask,
        m=m,
    )

    assert unpad_mock.call_count == 1
    unpad_mock.assert_called_with(m, dummy_return_mask)