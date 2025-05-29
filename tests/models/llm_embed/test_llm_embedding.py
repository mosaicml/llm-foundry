# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Optional
from unittest.mock import patch

import pytest
import torch
from composer import Trainer
from composer.core import get_precision_context
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoConfig, PretrainedConfig
from transformers.modeling_outputs import \
    BaseModelOutputWithPastAndCrossAttentions

from llmfoundry.models.llm_embed import ContrastiveEvalLoss, ContrastiveModel
from llmfoundry.utils.builders import build_dataloader
from tests.data_utils import temporary_contrastive_streaming_dataset
from tests.test_utils import MockTokenizer


@pytest.fixture
def mock_tokenizer() -> MockTokenizer:
    return MockTokenizer()


class MockAutoModel(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.config: PretrainedConfig = AutoConfig.from_pretrained(
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
        last_hidden_state = torch.randn(
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
) -> ContrastiveModel:
    with patch('transformers.AutoModel.from_pretrained', return_value=mock_auto_model), \
         patch('torch.distributed.is_initialized', return_value=False), \
         patch('torch.distributed.get_rank', return_value=0), \
         patch('torch.distributed.barrier'), \
         patch('llmfoundry.models.llm_embed.FinetuneEmbeddingModel.construct_model', return_value=mock_auto_model):
        model_instance: ContrastiveModel = ContrastiveModel(
            tokenizer=mock_tokenizer,
            pretrained_model_name_or_path='bert-base-uncased',
            loss_fn='torch_crossentropy',
            use_flash_attention_2=False,
        )
        return model_instance


def build_lm_config(is_hf: bool, attn_impl: Optional[str]) -> dict[str, Any]:
    if is_hf:
        assert attn_impl is None
        return {
            'pretrained_model_name_or_path': 'facebook/opt-350m',
            'pretrained': False,
            'config_overrides': {
                'hidden_size': 2,
                'num_attention_heads': 2,
                'num_hidden_layers': 2,
            },
        }
    else:
        assert attn_impl is not None
        return {
            'num_layers': 2,
            'word_embed_proj_dim': 128,
            'd_model': 128,
            'n_heads': 2,
            'vocab_size': 4096,
            'attn_config': {
                'attn_impl': attn_impl,
            },
        }


def build_tokenizer_config(is_hf: bool) -> dict[str, Any]:
    return {'vocab_size': 50257 if is_hf else 4096}


@pytest.mark.gpu
@pytest.mark.parametrize('is_hf', [True, False])
@pytest.mark.parametrize('attn_impl', ['flash', 'torch'])
def test_mpt_embedding_lm(
    is_hf: bool,
    attn_impl: str,
    mock_tokenizer: MockTokenizer,
):
    maybe_attn_impl = None if is_hf else attn_impl
    lm_config = build_lm_config(is_hf, maybe_attn_impl)

    model = ContrastiveModel(**lm_config, tokenizer=mock_tokenizer).to('cuda')
    msl = 32
    model_inputs_batch = mock_tokenizer([['pair 1 a', 'pair 1 b'],
                                         ['pair 2 a', 'pair 2 b']],
                                        padding='max_length',
                                        truncation=True,
                                        max_length=msl,
                                        return_tensors='pt')
    if isinstance(model_inputs_batch, dict):
        model_inputs_batch = {
            k: v.to('cuda') for k, v in model_inputs_batch.items()
        }

    with get_precision_context('amp_bf16'):
        outputs = model(model_inputs_batch)

        assert isinstance(outputs, dict)
        assert 'hidden_states' in outputs

        hidden_states = outputs['hidden_states']
        assert isinstance(hidden_states, tuple)

        last_hidden_state = hidden_states[-1]
        proj_dim = model.model.config.word_embed_proj_dim
        assert last_hidden_state.shape == (
            4,
            msl,
            proj_dim,
        )  # 2 pairs * 2 texts per pair, 128 sequence length, word_embed_proj_dim dim
        assert last_hidden_state.dtype == torch.bfloat16
        assert last_hidden_state.device.type == 'cuda'


dataloader_config = lambda remote, local_ext: {
    'name': 'contrastive_pairs',
    'dataset': {
        'remote': remote,
        'local': remote + '_' + local_ext,
        'split': 'train',
        'max_seq_len': 1024,
    },
    'drop_last': False,
    'num_workers': 1,
    'max_hard_negatives': 1,
}


@pytest.mark.gpu
@pytest.mark.parametrize('is_hf', [True, False])
@pytest.mark.parametrize('attn_impl', ['flash', 'torch'])
@pytest.mark.parametrize(
    'ds_format',
    ['one_query_one_response', 'one_query_multiple_responses'],
)
def test_contrastive_loss(
    ds_format: str,
    is_hf: bool,
    attn_impl: str,
    mock_tokenizer: MockTokenizer,
):
    maybe_attn_impl = None if is_hf else attn_impl

    with temporary_contrastive_streaming_dataset(ds_format) as data_dir:
        lm_config = build_lm_config(is_hf, maybe_attn_impl)
        model = ContrastiveModel(**lm_config,
                                 tokenizer=mock_tokenizer).to('cuda')

        train_dataloader = build_dataloader(
            dataloader_config(data_dir, 'local'),
            mock_tokenizer,
            2,
        )

        with get_precision_context(
            'amp_bf16',
        ):
            trainer = Trainer(
                model=model,
                train_dataloader=train_dataloader,
                precision='amp_bf16',
                max_duration='1ba',
            )
            trainer.fit()


@pytest.mark.gpu
@pytest.mark.world_size(2)
@pytest.mark.parametrize(
    'use_legacy_gradient_passthrough',
    [
        pytest.param(
            True,
            marks=pytest.mark.xfail(reason='Does not backprop gradients.'),
        ),
        False,
    ],
)
def test_distributed_loss(
    use_legacy_gradient_passthrough: bool,
    mock_tokenizer: MockTokenizer,
):
    is_hf = False

    lm_config = build_lm_config(is_hf, 'flash')
    lm_config['contrastive_config'] = {
        'gather_in_batch_negatives': True,
        'use_legacy_gradient_passthrough': use_legacy_gradient_passthrough,
    }

    lm_config_single_device = lm_config.copy()
    lm_config_single_device['contrastive_config'] = lm_config[
        'contrastive_config'].copy()
    lm_config_single_device['contrastive_config']['gather_in_batch_negatives'
                                                 ] = False

    model_for_ddp = ContrastiveModel(mock_tokenizer, **lm_config)
    model_ddp = model_for_ddp.to('cuda').to(torch.bfloat16)
    model_ddp = DDP(model_ddp)

    model = ContrastiveModel(mock_tokenizer,
                             **lm_config_single_device).to('cuda').to(
                                 torch.bfloat16,
                             )
    model.load_state_dict(model_for_ddp.state_dict())

    input_batch = mock_tokenizer([['pair 1 a', 'pair 1 b'],
                                  ['pair 2 a', 'pair 2 b'],
                                  ['pair 3 a', 'pair 3 b'],
                                  ['pair 4 a', 'pair 4 b']],
                                 padding='max_length',
                                 truncation=True,
                                 max_length=128,
                                 return_tensors='pt')
    if isinstance(input_batch, dict):
        input_batch = {k: v.to('cuda') for k, v in input_batch.items()}


def test_contrastive_eval_loss_update_and_compute() -> None:
    metric = ContrastiveEvalLoss()

    # Mock outputs and labels
    outputs1 = {'loss': torch.tensor(1.0)}
    outputs2 = {'loss': torch.tensor(2.0)}
    outputs3 = {'loss': torch.tensor(3.0)}

    # Update metric
    metric.update(outputs1, None)
    metric.update(outputs2, None)
    metric.update(outputs3, None)

    # Compute average loss
    average_loss = metric.compute()
    assert average_loss == pytest.approx(2.0)


def test_contrastive_eval_loss_device_handling() -> None:
    metric = ContrastiveEvalLoss()

    # Mock outputs on a different device
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    loss_tensor = torch.tensor(1.5, device=device)
    outputs = {'loss': loss_tensor}

    metric.update(outputs, None)

    # Ensure loss is moved to metric's device
    assert metric.loss.device.type == device
    assert metric.loss == loss_tensor


def test_eval_forward_without_outputs(model: ContrastiveModel) -> None:
    # Create a mock batch
    batch = {
        'input_ids': torch.randint(0, 1000, (2, 128)),
        'attention_mask': torch.ones(2, 128, dtype=torch.long),
        'labels': torch.randint(0, 2, (2, 128)),
    }

    # Mock the forward method to return a mock output with 'loss'
    with patch.object(model, 'forward') as mock_forward, \
         patch.object(model, 'loss') as mock_loss:
        mock_forward.return_value = {
            'loss': torch.tensor(1.0),
            'hidden_states': None,
        }
        mock_loss.return_value = torch.tensor(1.0)

        result = model.eval_forward(batch)

        assert isinstance(result, dict)
        assert 'loss' in result
        assert 'outputs' in result
        assert result['loss'] == torch.tensor(1.0)
        assert result['outputs'] == mock_forward.return_value


def test_eval_forward_with_outputs(model: ContrastiveModel) -> None:
    # Create a mock batch and outputs
    batch = {
        'input_ids': torch.randint(0, 1000, (2, 128)),
        'attention_mask': torch.ones(2, 128, dtype=torch.long),
        'labels': torch.randint(0, 2, (2, 128)),
    }
    mock_outputs = {'loss': torch.tensor(2.0), 'hidden_states': None}

    # Mock the loss method
    with patch.object(model, 'loss') as mock_loss:
        mock_loss.return_value = torch.tensor(2.0)

        result = model.eval_forward(batch, outputs=mock_outputs)

        assert isinstance(result, dict)
        assert 'loss' in result
        assert 'outputs' in result
        assert result['loss'] == torch.tensor(2.0)
        assert result['outputs'] == mock_outputs


def test_eval_forward_returns_correct_structure(
    model: ContrastiveModel,
) -> None:
    # Create a mock batch
    batch = {
        'input_ids': torch.randint(0, 1000, (1, 50)),
        'attention_mask': torch.ones(1, 50, dtype=torch.long),
        'labels': torch.randint(0, 2, (1, 50)),
    }

    # Mock the forward and loss methods
    with patch.object(model, 'forward') as mock_forward, \
         patch.object(model, 'loss') as mock_loss:
        mock_forward.return_value = {
            'loss': torch.tensor(0.5),
            'hidden_states': None,
        }
        mock_loss.return_value = torch.tensor(0.5)

        result = model.eval_forward(batch)

        assert isinstance(result, dict)
        assert set(result.keys()) == {'loss', 'outputs'}
        assert isinstance(result['loss'], torch.Tensor)
        assert isinstance(result['outputs'], dict)
        assert 'loss' in result['outputs']
        assert result['outputs']['loss'] == torch.tensor(0.5)


def test_eval_forward_handles_missing_outputs(model: ContrastiveModel) -> None:
    # Create a mock batch without 'loss' in outputs
    batch = {
        'input_ids': torch.randint(0, 1000, (2, 128)),
        'attention_mask': torch.ones(2, 128, dtype=torch.long),
        'labels': torch.randint(0, 2, (2, 128)),
    }

    # Mock the forward method to return outputs without 'loss'
    with patch.object(model, 'forward') as mock_forward, \
         patch.object(model, 'loss') as mock_loss:
        mock_forward.return_value = {'hidden_states': None}
        mock_loss.return_value = torch.tensor(
            1.0,
        )  # Assume loss is computed elsewhere

        result = model.eval_forward(batch)

        assert isinstance(result, dict)
        assert 'loss' in result
        assert 'outputs' in result
        assert result['loss'] == torch.tensor(1.0)
        assert result['outputs'] == mock_forward.return_value
