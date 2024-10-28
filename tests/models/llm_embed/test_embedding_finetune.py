# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from typing import Any
from unittest.mock import patch

import pytest
import torch
from transformers import AutoConfig, PreTrainedTokenizerBase
from transformers.modeling_outputs import \
    BaseModelOutputWithPastAndCrossAttentions

from llmfoundry.models.llm_embed import FinetuneEmbeddingModel


class MockTokenizer(PreTrainedTokenizerBase):

    def __init__(self) -> None:
        super().__init__()
        self.pad_token: str = '<pad>'
        self.eos_token: str = '</s>'
        self.bos_token: str = '<s>'
        self.unk_token: str = '<unk>'
        self._vocab_size: int = 30000

    def __len__(self) -> int:
        return self._vocab_size


@pytest.fixture
def mock_tokenizer() -> MockTokenizer:
    return MockTokenizer()


class MockAutoModel(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.config: AutoConfig = AutoConfig.from_pretrained(
            'bert-base-uncased',
        )
        self.config.hidden_size = 768
        self.config.num_hidden_layers = 12
        self.config.n_layers = 12
        self.config.vocab_size = 30000
        self.linear: torch.nn.Linear = torch.nn.Linear(
            self.config.hidden_size,
            self.config.hidden_size,
        )

    @classmethod
    def from_pretrained(cls, *args: Any, **kwargs: Any) -> 'MockAutoModel':
        return cls()

    def forward(
        self,
        **kwargs: Any,
    ) -> BaseModelOutputWithPastAndCrossAttentions:
        # Simulate forward pass
        input_ids: torch.Tensor = kwargs.get(
            'input_ids',
            torch.zeros(1, 10, dtype=torch.long),
        )
        batch_size: int = input_ids.size(0)
        seq_length: int = input_ids.size(1)
        last_hidden_state: torch.Tensor = torch.randn(
            batch_size,
            seq_length,
            self.config.hidden_size,
        )
        last_hidden_state = self.linear(last_hidden_state)
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=last_hidden_state,
            hidden_states=(last_hidden_state,) *
            (self.config.num_hidden_layers + 1),
        )


@pytest.fixture
def mock_auto_model() -> MockAutoModel:
    return MockAutoModel()


@pytest.fixture
def model(
    mock_tokenizer: MockTokenizer,
    mock_auto_model: MockAutoModel,
) -> FinetuneEmbeddingModel:
    with patch('transformers.AutoModel.from_pretrained', return_value=mock_auto_model), \
         patch('composer.utils.dist.get_global_rank', return_value=0), \
         patch('composer.utils.dist.barrier'), \
         patch('llmfoundry.models.llm_embed.FinetuneEmbeddingModel.construct_model', return_value=mock_auto_model):
        model_instance: FinetuneEmbeddingModel = FinetuneEmbeddingModel(
            tokenizer=mock_tokenizer,
            pretrained_model_name_or_path='bert-base-uncased',
            loss_fn='torch_crossentropy',
        )
        return model_instance


def test_construct_model(model: FinetuneEmbeddingModel) -> None:
    with patch(
        'transformers.AutoModel.from_pretrained',
        return_value=model.model,
    ):
        constructed_model = model.construct_model()
        assert constructed_model is not None
        assert isinstance(constructed_model, MockAutoModel)


def test_get_hidden_state(model: FinetuneEmbeddingModel) -> None:
    mock_outputs: BaseModelOutputWithPastAndCrossAttentions = BaseModelOutputWithPastAndCrossAttentions(
        last_hidden_state=torch.randn(1, 10, model.model.config.hidden_size),
    )
    hidden_state: torch.Tensor = model.get_hidden_state(mock_outputs)
    assert torch.equal(hidden_state, mock_outputs.last_hidden_state)


def test_handle_language_head(model: FinetuneEmbeddingModel) -> None:
    mock_outputs: BaseModelOutputWithPastAndCrossAttentions = BaseModelOutputWithPastAndCrossAttentions(
        last_hidden_state=torch.randn(1, 10, model.model.config.hidden_size),
    )
    result: torch.Tensor = model.handle_language_head(mock_outputs)
    assert isinstance(result, torch.Tensor)
    assert result.item() == 0
    assert result.dtype == torch.float32
    assert result.device == mock_outputs.last_hidden_state.device


def test_flops_per_batch(model: FinetuneEmbeddingModel) -> None:
    batch: dict[str, torch.Tensor] = {
        'input_ids': torch.randint(0, 1000, (2, 128)),
    }
    flops: int = model.flops_per_batch(batch)
    assert isinstance(flops, int)
    assert flops > 0


def test_get_attribute(model: FinetuneEmbeddingModel) -> None:
    config: AutoConfig = AutoConfig.from_pretrained('bert-base-uncased')
    config.hidden_size = 768
    config.d_model = 1024
    config.n_embd = 512

    attribute_value: Any = model._get_attribute(
        config,
        ['hidden_size', 'd_model', 'n_embd'],
    )
    assert attribute_value == 768
    attribute_value = model._get_attribute(config, ['d_model', 'n_embd'])
    assert attribute_value == 1024
    attribute_value = model._get_attribute(
        config,
        ['non_existent', 'also_non_existent'],
    )
    assert attribute_value is None


@pytest.mark.parametrize(
    'dist_initialized',
    [
        pytest.param(
            True,
            marks=[
                pytest.mark.gpu,
                pytest.mark.world_size(2),
            ],
        ),
        pytest.param(False),
    ],
)
def test_construct_model_distributed(
    mock_tokenizer: MockTokenizer,
    mock_auto_model: MockAutoModel,
    dist_initialized: bool,
) -> None:
    with patch('torch.distributed.is_initialized', return_value=dist_initialized), \
         patch('torch.distributed.get_rank', return_value=0), \
         patch('torch.distributed.barrier'), \
         patch('transformers.AutoModel.from_pretrained', return_value=mock_auto_model), \
         patch('llmfoundry.models.llm_embed.FinetuneEmbeddingModel.construct_model', return_value=mock_auto_model):
        model_instance: FinetuneEmbeddingModel = FinetuneEmbeddingModel(
            tokenizer=mock_tokenizer,
            pretrained_model_name_or_path='bert-base-uncased',
            loss_fn='torch_crossentropy',
        )
        constructed_model: torch.nn.Module = model_instance.construct_model()
        assert constructed_model is not None
        assert isinstance(constructed_model, torch.nn.Module)
