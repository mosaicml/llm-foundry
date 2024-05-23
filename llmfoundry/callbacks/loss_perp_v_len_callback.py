# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, Mapping, Optional, Tuple

import torch
from composer.core import Callback, State
from composer.loggers import Logger, MLFlowLogger
from composer.utils import dist
from torchmetrics import Metric

from llmfoundry.models.mpt import ComposerMPTCausalLM
from llmfoundry.utils.warnings import experimental_class

__all__ = [
    'LossPerpVsContextLengthLogger',
]


@experimental_class('LossPerpVsContextLengthLogger')
class LossPerpVsContextLengthLogger(Callback):
    """Logs the average loss and perplexity for every context length.

    Note: Currently only works with MLFlow logger.

    Args:
        log_batch_interval (int): The interval for logging. Currently logging takes longer because MLFlow downloads the table, appends rows to it, and then re-uploads it. Once this is fixed, log_batch_interval will be removed and this will always log as soon as the metric is computed.
        compute_batch_interval (int): The interval for computing the metric.
        ignore_index (int): Specifies a target value that is ignored for computing loss.
    """

    def __init__(
        self,
        log_batch_interval: int,
        compute_batch_interval: int,
        ignore_index: int = -100,
    ):
        if compute_batch_interval > log_batch_interval:
            raise ValueError(
                'log_batch_interval is shorter than the compute_batch_interval for LossPerpVsContextLengthLogger.',
            )
        self.log_batch_interval = log_batch_interval
        self.compute_batch_interval = compute_batch_interval
        self.ignore_index = ignore_index
        self.metric_dict = {}
        self.loss_perp_v_len = LossPerpVLen(ignore_index)

    def init(self, state: State, logger: Logger) -> None:
        if not isinstance(state.model, ComposerMPTCausalLM):
            raise ValueError(
                'LossPerpVsContextLengthLogger only supported for ComposerMPTCausalLM models.',
            )
        if state.model.shift_labels is None:
            raise ValueError(
                'state.model.shift_labels should be set for LossPerpVsContextLengthLogger.',
            )
        if all(
            not isinstance(destination, MLFlowLogger)
            for destination in logger.destinations
        ):
            raise NotImplementedError(
                'Did not find MLflow in the list of loggers. LossPerpVsContextLengthLogger is only implemented for the MLflow logger.',
            )

    def after_backward(self, state: State, logger: Logger) -> None:
        if state.timestamp.batch.value % self.compute_batch_interval == 0:
            sequence_id = state.batch['sequence_id'
                                     ] if 'sequence_id' in state.batch else None
            labels = state.batch['labels']
            if state.model.shift_labels:
                labels[:, :-1] = labels[:, 1:].detach().clone()
                labels[:, -1] = -100
            seq_parallel_world_size = getattr(
                state.model.model.transformer,
                'seq_parallel_world_size',
                1,
            )
            seq_parallel_rank = state.model.model.transformer.seq_parallel_rank if seq_parallel_world_size > 1 else 0

            if isinstance(state.outputs, Mapping):
                logits = state.outputs['logits']  # type: ignore
            elif isinstance(state.outputs, torch.Tensor):
                logits = state.outputs
            else:
                raise Exception(
                    f'Type {type(state.outputs)} for the output is unsupported.',
                )

            if labels.shape[1] != logits.shape[1]:
                raise ValueError(
                    f'The length of labels, {labels.shape[1]=} does not match the length of logits {logits.shape[1]=}.',
                )

            labels, logits = self.preprocess_metric_inputs(
                sequence_id,
                labels,
                logits,
                seq_parallel_world_size,
                seq_parallel_rank,
            )

            self.loss_perp_v_len.update(
                labels,
                logits,
                sequence_id,
                state.model.loss_fn,
            )

    def batch_end(self, state: State, logger: Logger) -> None:
        if (
            state.timestamp.batch.value - 1
        ) % self.compute_batch_interval == 0:  # state.timestamp.batch.value - 1 because batch is incremented before batch_end (https://github.com/mosaicml/composer/blob/57c7b72b9df41b0c9777bad1c2bec17f3103c31f/composer/trainer/trainer.py#L2478C1-L2484C55)
            current_metric_dict = self.loss_perp_v_len.compute()
            if dist.get_global_rank() == 0:
                for k, v in current_metric_dict.items():
                    v = v.tolist()
                    v.append(
                        state.timestamp.batch.value,
                    )  # Add the current batch index as the last column
                    if k not in self.metric_dict:
                        self.metric_dict[k] = []
                    self.metric_dict[k].append(v)
        if (
            state.timestamp.batch.value - 1
        ) % self.log_batch_interval == 0 and dist.get_global_rank(
        ) == 0:  # state.timestamp.batch.value - 1 because batch is incremented before batch_end (https://github.com/mosaicml/composer/blob/57c7b72b9df41b0c9777bad1c2bec17f3103c31f/composer/trainer/trainer.py#L2478C1-L2484C55)
            for k, v in self.metric_dict.items():
                columns = []
                columns = [
                    f'context_length_{i}' for i in range(len(v[0]) - 1)
                ]  # len(v[0]) - 1 because the last column is the batch index
                columns.append(
                    'batch_index',
                )  # Add batch as the last column name
                for destination in logger.destinations:
                    if isinstance(destination, MLFlowLogger):
                        destination.log_table(
                            columns=columns,
                            rows=v,
                            name=f'metrics/train/LossPerpVLenTable/{k}',
                            step=state.timestamp.batch.value,
                        )
            self.metric_dict = {}

    def preprocess_metric_inputs(
        self,
        sequence_id: Optional[torch.Tensor],
        labels: torch.Tensor,
        logits: torch.Tensor,
        seq_parallel_world_size: int,
        seq_parallel_rank: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        del sequence_id, seq_parallel_rank
        if seq_parallel_world_size > 1:
            raise ValueError(
                'LossPerpVsContextLengthLogger does not support sequence parallelism',
            )

        return labels, logits


class LossPerpVLen(Metric):

    full_state_update = False

    def __init__(
        self,
        ignore_index: int,
        dist_sync_on_step: bool = False,
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.ignore_index = ignore_index
        self.add_state('sum_loss', default=torch.Tensor(), dist_reduce_fx='sum')
        self.add_state(
            'sum_perplexity',
            default=torch.Tensor(),
            dist_reduce_fx='sum',
        )
        self.add_state(
            'sum_length',
            default=torch.Tensor(),
            dist_reduce_fx='sum',
        )

        self.add_state(
            'sum_loss_seq_id',
            default=torch.Tensor(),
            dist_reduce_fx='sum',
        )
        self.add_state(
            'sum_perplexity_seq_id',
            default=torch.Tensor(),
            dist_reduce_fx='sum',
        )
        self.add_state(
            'sum_length_seq_id',
            default=torch.Tensor(),
            dist_reduce_fx='sum',
        )

    def update(
        self,
        labels: torch.Tensor,
        logits: torch.Tensor,
        sequence_id: Optional[torch.Tensor],
        loss_fn: Any,
    ) -> None:
        """Updates the internal state with results from a new batch.

        Args:
            labels (torch.Tensor): A Tensor of ground-truth values to compare against.
            logits (torch.Tensor): A Tensor of labels.
            sequence_id (torch.Tensor | None): The sequence ids for tokens.
            loss_fn (Any): The cross entropy loss to use.
        """
        valid_labels_mask = torch.where(
            labels != self.ignore_index,
            torch.ones_like(labels),
            torch.zeros_like(labels),
        )
        bsz, seq_len = labels.shape
        loss = loss_fn(logits.view(bsz * seq_len, -1), labels.view(-1))
        loss = loss.view(bsz, seq_len)
        perplexity = torch.exp(loss)

        if self.sum_loss.numel() == 0:
            self.sum_loss = torch.zeros(  # type: ignore
                seq_len,
                device=loss.device,
                dtype=loss.dtype,
            )
            self.sum_perplexity = torch.zeros(  # type: ignore
                seq_len,
                device=loss.device,
                dtype=loss.dtype,
            )
            self.sum_length = torch.zeros(  # type: ignore
                seq_len,
                device=loss.device,
                dtype=torch.long,
            )
            self.sum_loss_seq_id = torch.zeros(  # type: ignore
                seq_len,
                device=loss.device,
                dtype=loss.dtype,
            )
            self.sum_perplexity_seq_id = torch.zeros(  # type: ignore
                seq_len,
                device=loss.device,
                dtype=loss.dtype,
            )
            self.sum_length_seq_id = torch.zeros(  # type: ignore
                seq_len,
                device=loss.device,
                dtype=torch.long,
            )

        self.sum_loss += torch.sum(loss, dim=(0))
        self.sum_perplexity += torch.sum(perplexity, dim=(0))
        self.sum_length += valid_labels_mask.sum(dim=0)

        if sequence_id is not None:
            seq_id_expanded = torch.nn.functional.one_hot(
                sequence_id,
            ).transpose(-1, -2)
            seq_lens = seq_id_expanded.sum(dim=-1)
            max_num_seq = seq_lens.shape[1]
            seq_tok_ids = torch.arange(seq_len, device=sequence_id.device)[
                None, None, :].expand(bsz, max_num_seq, -1)
            mask = seq_tok_ids < seq_lens[:, :, None]
            seq_len_offsets = torch.nn.functional.pad(
                seq_lens.cumsum(dim=1)[:, :-1],
                (1, 0),
                value=0,
            )
            seq_tok_ids = seq_tok_ids + seq_len_offsets[:, :, None]
            seq_tok_ids = torch.where(
                mask,
                seq_tok_ids,
                torch.zeros_like(seq_tok_ids),
            )

            loss = loss[:, None, :].expand(-1, max_num_seq, -1)
            perplexity = perplexity[:, None, :].expand(-1, max_num_seq, -1)
            valid_labels_mask = valid_labels_mask[:, None, :].expand(
                -1,
                max_num_seq,
                -1,
            )
            loss = torch.where(
                mask,
                torch.gather(input=loss, dim=2, index=seq_tok_ids),
                torch.zeros_like(loss),
            )
            perplexity = torch.where(
                mask,
                torch.gather(input=perplexity, dim=2, index=seq_tok_ids),
                torch.zeros_like(perplexity),
            )
            mask = torch.where(
                mask,
                torch.gather(input=valid_labels_mask, dim=2, index=seq_tok_ids),
                torch.zeros_like(valid_labels_mask),
            )

            self.sum_loss_seq_id += torch.sum(loss, dim=(0, 1))
            self.sum_perplexity_seq_id += torch.sum(perplexity, dim=(0, 1))
            self.sum_length_seq_id += torch.sum(mask, dim=(0, 1))

    def compute(self) -> Dict[str, torch.Tensor]:
        """Aggregate the state over all processes to compute the metric.

        Returns:
            loss: The loss averaged across all batches as a :class:`~torch.Tensor`.
        """
        # Return average loss over entire dataset
        sum_perplexity = torch.where(
            self.sum_length != 0,
            self.sum_perplexity,
            -1,
        )
        sum_loss = torch.where(self.sum_length != 0, self.sum_loss, -1)
        sum_length = torch.where(self.sum_length != 0, self.sum_length, 1)

        sum_perplexity_seq_id = torch.where(
            self.sum_length_seq_id != 0,
            self.sum_perplexity_seq_id,
            -1,
        )
        sum_loss_seq_id = torch.where(
            self.sum_length_seq_id != 0,
            self.sum_loss_seq_id,
            -1,
        )
        sum_length_seq_id = torch.where(
            self.sum_length_seq_id != 0,
            self.sum_length_seq_id,
            1,
        )

        return {
            'mean_loss_v_len':
                sum_loss / sum_length,
            'mean_perplexity_v_len':
                sum_perplexity / sum_length,
            'sum_length':
                self.sum_length,
            'mean_loss_seq_id_v_len':
                sum_loss_seq_id / sum_length_seq_id,
            'mean_perplexity_seq_id_v_len':
                sum_perplexity_seq_id / sum_length_seq_id,
            'sum_length_seq_id':
                self.sum_length_seq_id,
        }
