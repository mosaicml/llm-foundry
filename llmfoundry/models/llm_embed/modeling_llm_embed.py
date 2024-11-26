# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""LLM Contrastive Embedding Model.

Implements InfoNCE Loss using the MPT architecture. The resulting model can
be used as a vector embedding model.

This is inspired by Microsoft Research's unilm repository
https://github.com/microsoft/unilm
"""

import logging
from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping, Optional, Union, cast

import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F
from composer.models import HuggingFaceModel
from composer.utils import dist
from einops import rearrange
from omegaconf import OmegaConf as om
from torch.distributed.nn.functional import all_gather
from torchmetrics import Metric
from transformers import PreTrainedTokenizerBase
from transformers.modeling_outputs import CausalLMOutputWithPast

from llmfoundry import registry
from llmfoundry.models.hf.hf_causal_lm import ComposerHFCausalLM
from llmfoundry.models.mpt.configuration_mpt import MPTConfig
from llmfoundry.models.mpt.modeling_mpt import MPTForCausalLM
from llmfoundry.models.utils.config_moe_args import create_set_process_group

log = logging.getLogger(__name__)


class ContrastiveEvalLoss(Metric):

    def __init__(self):
        super().__init__()
        self.add_state('loss', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, outputs: Any, labels: Any):
        loss = outputs['loss']
        if loss.device != self.loss.device:
            loss = loss.to(self.loss.device)
        self.loss += loss
        self.total += 1

    def compute(self):
        return self.loss / self.total


@dataclass
class ContrastiveConfig:
    """Configuration for the contrastive loss.

    Args:
        temperature (Union[int, float], optional): Temperature for InfoNCE Loss. Defaults to 1.
        vector_representation (str, optional): The vector representation to use. Defaults to 'avg'.
        normalize_output (bool, optional): Whether to normalize the output. Defaults to True.
        gather_in_batch_negatives (bool, optional): Whether to call all_gather on all samples in global batch
        use_legacy_gradient_passthrough (bool, optional): Whether to use the legacy gradient passthrough. Defaults to False.
        infonce_process_group_size (int, optional): The size of the process group for InfoNCE loss. Defaults to None.
    """
    temperature: Union[int, float] = 1
    vector_representation: str = 'avg'
    normalize_output: bool = True
    pos_step_size: int = -1  #keep for backwards compatibility
    gather_in_batch_negatives: bool = False
    use_legacy_gradient_passthrough: bool = False
    infonce_process_group_size: Optional[int] = None


class ContrastiveModel(HuggingFaceModel):
    """A contrastive loss function wrapping MPT or Huggingface architecture.

    This model applies a contrastive loss function to either a MPT (Mosaic Pretrained Transformer)
    or a Huggingface architecture. It allows for bidirectional encoding by modifying the attention mask.

    Args:
        tokenizer (PreTrainedTokenizerBase): The tokenizer used for tokenization.
        contrastive_config (Dict[str, Any], optional): Configuration for the contrastive loss. Defaults to None. See `ContrastiveConfig`.
        pretrained_model_name_or_path (Optional[str], optional): Pretrained model name or path. Defaults to None.
        pretrained_lora_id_or_path (Optional[str], optional): Pretrained LoRA (Low Rank Adaptation) ID or path. Defaults to None.
        trust_remote_code (bool, optional): Whether to trust remote code. Defaults to False.
        init_device (str, optional): The initial device. Defaults to 'cpu'.
        use_flash_attention_2 (bool, optional): Whether to use Flash Attention 2. Defaults to True.
        use_auth_token (bool, optional): Whether to use an authentication token. Defaults to False.
        config_overrides (Optional[Dict[str, Any]], optional): Overrides for the model configuration. Defaults to None.
        load_in_8bit (bool, optional): Whether to load the model in 8-bit mode. Defaults to False.
        loss_fn (str, optional): The loss function to use (either 'torch_crossentropy' or 'fused_crossentropy'). Defaults to 'fused_crossentropy'.
        pretrained (bool, optional): Whether to use a pretrained model when using a Hugging Face architecture. Defaults to True.
        **kwargs (Dict[str, Any]): Additional keyword arguments.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        contrastive_config: Optional[dict[str, Any]] = None,
        pretrained_model_name_or_path: Optional[str] = None,
        pretrained_lora_id_or_path: Optional[str] = None,
        trust_remote_code: bool = False,
        init_device: str = 'cpu',
        use_flash_attention_2: bool = True,
        use_auth_token: bool = False,
        config_overrides: Optional[dict[str, Any]] = None,
        load_in_8bit: bool = False,
        loss_fn: str = 'fused_crossentropy',
        pretrained: bool = True,
        **kwargs: dict[str, Any],
    ):
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.pretrained = pretrained
        self.pretrained_lora_id_or_path = pretrained_lora_id_or_path
        self.trust_remote_code = trust_remote_code
        self.init_device = init_device
        self.use_flash_attention_2 = use_flash_attention_2
        self.use_auth_token = use_auth_token
        self.config_overrides = config_overrides
        self.load_in_8bit = load_in_8bit
        self.kwargs = kwargs
        self.is_mpt = False

        contrastive_config = contrastive_config or {}
        contrastive_config_obj: ContrastiveConfig = om.structured(
            ContrastiveConfig(**contrastive_config),
        )
        if tokenizer.pad_token is None:  # type: ignore
            tokenizer.pad_token = tokenizer.eos_token

        model = self.construct_model()

        train_metrics: list[Metric] = [
        ]  # TODO: no train metrics for embedding models yet!

        self.eval_metrics = [
            ContrastiveEvalLoss(),
        ]

        super().__init__(
            model=model,
            tokenizer=tokenizer,
            use_logits=False,
            metrics=train_metrics,
            eval_metrics=self.eval_metrics, # type: ignore
            shift_labels=False,
            allow_embedding_resizing=True,
        )

        # Temperature for InfoNCELoss
        self.temperature = contrastive_config_obj.temperature

        # Set the vector representation to either be the average of all the individual token vectors,
        # or the EOS token at the end of the sequence
        self.vector_representation = contrastive_config_obj.vector_representation
        self.normalize_output = contrastive_config_obj.normalize_output

        self.step_size = None
        self.gather_in_batch_negatives = contrastive_config_obj.gather_in_batch_negatives
        self.use_legacy_gradient_passthrough = contrastive_config_obj.use_legacy_gradient_passthrough
        self.n_active_params = sum(p.numel() for p in self.parameters())
        if loss_fn == 'fused_crossentropy':
            try:
                from flash_attn.losses.cross_entropy import \
                    CrossEntropyLoss as FusedCrossEntropyLoss

                self.loss_fn = FusedCrossEntropyLoss(ignore_index=-100)
            except:
                raise ValueError(
                    'Fused Cross Entropy is not installed. Either (1) have a CUDA-compatible GPU '
                    +
                    'and `pip install .[gpu]` if installing from source or `pip install xentropy-cuda-lib@git+https://github.com/HazyResearch/flash-attention.git@v1.0.3#subdirectory=csrc/xentropy` '
                    +
                    'if installing from pypi, or (2) set your config model.loss_fn=torch_crossentropy.',
                )
        elif loss_fn == 'torch_crossentropy':
            self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        else:
            raise ValueError(
                f'Specified loss_fn={loss_fn} not recognized. `loss_fn` must be one of [`fused_crossentropy`, `torch_crossentropy`].',
            )

        self.infonce_process_group = None
        if contrastive_config_obj.infonce_process_group_size is not None:
            pg_size = contrastive_config_obj.infonce_process_group_size
            self.infonce_process_group = create_set_process_group(pg_size)

    def construct_model(self):
        if self.pretrained_model_name_or_path:
            model_class = registry.models.get('hf_causal_lm')
            model_class = cast(type[ComposerHFCausalLM], model_class)
            model = model_class.build_inner_model(
                pretrained=self.pretrained,
                pretrained_model_name_or_path=self.
                pretrained_model_name_or_path,
                pretrained_lora_id_or_path=self.pretrained_lora_id_or_path,
                trust_remote_code=self.trust_remote_code,
                init_device=self.init_device,
                use_flash_attention_2=self.use_flash_attention_2,
                use_auth_token=self.use_auth_token,
                config_overrides=self.config_overrides or {},
                load_in_8bit=self.load_in_8bit,
                **self.kwargs,
            )
        else:
            model = MPTForCausalLM(MPTConfig(**self.kwargs))
            self.is_mpt = True
        return model

    def _update_step_size_if_needed(self, batch: MutableMapping) -> None:
        """Update step size on first batch if we detect hard negatives."""
        if self.step_size:
            return

        input_shape = batch['input_ids'].shape
        if input_shape[1] > 2:
            # We have hard negatives, batch shape is [batch, sample of query+positive passage+negative passages, tokens].
            self.step_size = input_shape[1]
            log.info(
                f'Detected hard negatives, updated step_size to {self.step_size}',
            )
        else:
            self.step_size = 2

    def format_queries_batch(
        self,
        batch: MutableMapping,
        last_hidden_state: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Format `queries` by selecting every ``n``th entry from the batch.

        Here ``n`` is the step size, which represents the number of hard
        negatives per passage.
        """
        assert self.step_size
        queries = {}
        indices = list(range(0, batch['input_ids'].size(0), self.step_size))
        for key in batch:
            queries[key] = batch[key][indices, :]
        return queries, last_hidden_state[indices, :, :]

    def format_passages_batch(
        self,
        batch: MutableMapping,
        last_hidden_state: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Format `passages` by selecting every ``n``th entry from the batch.

        Here ``n`` is the step size, which represents the number of hard
        negatives per passage.
        """
        assert self.step_size
        passages = {}
        num_blocks = batch['input_ids'].size(0) // self.step_size
        index = torch.arange(
            1,
            num_blocks * self.step_size + 1,
            device=last_hidden_state.device,
        ).view(num_blocks, self.step_size)
        index = index[:, :self.step_size - 1].reshape(-1)
        for key in batch:
            passages[key] = batch[key][index]
        return passages, last_hidden_state[index, :, :]

    def forward(self, batch: MutableMapping) -> CausalLMOutputWithPast:
        # Collapse pairs into the batch dimension
        self._update_step_size_if_needed(batch)
        collapse_dims = lambda x: rearrange(x, 'b p d -> (b p) d') if \
            len(x.shape) > 2 else x

        for key in batch:
            batch[key] = collapse_dims(batch[key])

        return self.model(
            output_hidden_states=True,
            **batch,
        )

    def _cat_gather(self, t: torch.Tensor, group: Any = None) -> torch.Tensor:
        """Applies an all gather operation necessary for InfoNCELoss.

        See https://github.com/pytorch/pytorch/blob/63d5e9221bedd1546b7d364b5ce4171547db12a9/torch/distributed/nn/functional.py#L314
        as well as https://github.com/pytorch/pytorch/issues/121587#issuecomment-1989070351
        """
        if self.use_legacy_gradient_passthrough:
            all_tensors = list(dist.all_gather(t, group))
            all_tensors[dist.get_global_rank()] = t
            all_tensors = torch.cat(all_tensors)
        else:
            extra_kwargs = {'group': group} if group is not None else {}
            all_tensors = all_gather(t, **extra_kwargs)
            all_tensors = torch.cat(all_tensors)

        return all_tensors

    def get_hidden_state(self, outputs: CausalLMOutputWithPast) -> torch.Tensor:
        """Returns the hidden state to use for pooling."""
        return outputs.hidden_states[-1]

    def handle_language_head(
        self,
        outputs: CausalLMOutputWithPast,
    ) -> torch.Tensor:
        """Handles `zero` tensor to avoid DDP unused parameters error."""
        return torch.sum(
            outputs.logits,
        ) * 0  # This attaches the language head to the computation graph

    def _compute_scores(
        self,
        batch: MutableMapping,
        outputs: CausalLMOutputWithPast,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run pairs through the encoder separately in two passes.

        This function splits queries and passages based on the step size, which represents
        the number of hard negatives per passage. It then runs the queries and passages
        through the encoder separately to obtain the encoded representations. The encoded
        representations are used for further computations in the model.

        Args:
            batch (MutableMapping): The input batch containing queries and passages.
            outputs (CausalLMOutputWithPast): The model outputs.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The encoded representations of queries and passages.
        """
        hidden_state = self.get_hidden_state(outputs)
        zero = self.handle_language_head(outputs)
        (
            queries_batch,
            queries_last_hidden_state,
        ) = self.format_queries_batch(batch, hidden_state)
        (
            passages_batch,
            passages_last_hidden_state,
        ) = self.format_passages_batch(batch, hidden_state)

        query_attn_mask = queries_batch.get('attention_mask')
        passage_attn_mask = passages_batch.get('attention_mask')
        assert isinstance(query_attn_mask, torch.Tensor)
        assert isinstance(passage_attn_mask, torch.Tensor)
        if self.vector_representation == 'eos':

            def pool_fn(x: torch.Tensor, mask: torch.Tensor):
                row_indices = torch.arange(mask.shape[0])
                flipped_mask = ~mask.bool()
                last_true_indices = flipped_mask.int().argmax(dim=1) - 1
                pooled_outputs = x[row_indices, last_true_indices, :]
                return pooled_outputs
        elif self.vector_representation == 'avg':

            def pool_fn(x: torch.Tensor, mask: torch.Tensor):
                x = x.masked_fill(~mask[..., None].bool(), 0.0)
                pooled_outputs = x.sum(dim=1) / (mask.sum(dim=1)[..., None])
                return pooled_outputs
        else:
            raise ValueError(
                f'Specified vector_representation={self.vector_representation} not recognized. `vector_representation` must be one of [`avg`, `eos`].',
            )

        q_pooled_outputs = pool_fn(queries_last_hidden_state, query_attn_mask)
        p_pooled_outputs = pool_fn(
            passages_last_hidden_state,
            passage_attn_mask,
        )

        if self.normalize_output:
            q_pooled_outputs = F.normalize(q_pooled_outputs, dim=-1)
            p_pooled_outputs = F.normalize(p_pooled_outputs, dim=-1)

        # Use all_gather to include negatives across mini batch
        if self.gather_in_batch_negatives:
            all_q_pooled_outputs = self._cat_gather(
                q_pooled_outputs,
                group=self.infonce_process_group,
            )
            all_p_pooled_outputs = self._cat_gather(
                p_pooled_outputs,
                group=self.infonce_process_group,
            )
        else:
            all_q_pooled_outputs = q_pooled_outputs
            all_p_pooled_outputs = p_pooled_outputs

        assert all_q_pooled_outputs is not None
        assert all_p_pooled_outputs is not None

        all_scores = self._full_contrastive_scores(
            queries=all_q_pooled_outputs,
            passages=all_p_pooled_outputs,
        )
        all_scores = all_scores * (1 / self.temperature) + zero

        all_labels = torch.arange(
            all_scores.size(0),
            device=q_pooled_outputs.device,
            dtype=torch.long,
        )
        all_labels = all_labels * (
            p_pooled_outputs.size(0) // q_pooled_outputs.size(0)
        )

        return all_scores, all_labels

    def _full_contrastive_scores(
        self,
        queries: torch.Tensor,
        passages: torch.Tensor,
    ) -> torch.Tensor:

        # this calculates the inner product between query and passage pairs
        qp = torch.mm(queries, passages.t())

        return qp

    def loss(
        self,
        outputs: CausalLMOutputWithPast,
        batch: MutableMapping,
    ) -> torch.Tensor:
        scores, labels = self._compute_scores(batch, outputs)
        loss = self.loss_fn(scores, labels)
        return loss

    def eval_forward(
        self,
        batch: MutableMapping,
        outputs: Optional[Any] = None,
    ):
        if outputs is None:
            outputs = self.forward(batch)
        val_loss = self.loss(outputs, batch)
        return {'loss': val_loss, 'outputs': outputs}

    def flops_per_batch(self, batch: Mapping) -> int:
        # Note: this computation does not take into account padding, and assumes
        # that the dataset has been constructed without padding. Additionally, we
        # assume the backward pass is approximately 2x the forward pass

        bs, msl = batch['input_ids'].shape[0:2]
        params_flops_per_token = 2 * self.n_active_params
        params_flops_per_seq = params_flops_per_token * msl
        attn_flops_per_seq = (
            self.model.config.n_layers * 2 * 2 *
            (self.model.config.d_model * (msl**2))
        )

        return (params_flops_per_seq + attn_flops_per_seq) * 3 * bs
