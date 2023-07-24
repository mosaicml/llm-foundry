# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""A wrapper class that converts 🤗 Transformers models to composer models"""

from __future__ import annotations

import inspect
import json
import logging
from multiprocessing.sharedctypes import Value
import tempfile
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Type, Union

import numpy as np

import torch
from torchmetrics import Metric

from composer.metrics import InContextLearningMetric
from composer.models.base import ComposerModel
from composer.utils import MissingConditionalImportError, dist, get_file, import_object, is_model_fsdp, safe_torch_load

try:
    from examples.pytorch.gpt.utils import comm, gpt_decoder
    from examples.pytorch.gpt.utils.parallel_gpt import ParallelGPT
    import transformers
    from transformers import AutoTokenizer
    FASTERTRANSFORMER_INSTALLED = True
except:
    FASTERTRANSFORMER_INSTALLED = False

log = logging.getLogger(__name__)

__all__ = ['FasterTransformerInferenceModel']


class FasterTransformerInferenceModel(ComposerModel):
    """
    A wrapper class that converts FasterTransformer models to composer models. Only used for inference

    Args:
        model (): A FasterTransformer model.
        tokenizer (transformers.PreTrainedTokenizer, optional): The tokenizer used to prepare the dataset. Default ``None``.

            .. note:: If the tokenizer is provided, its config will be saved in the composer checkpoint, and it can be reloaded
                using :meth:`HuggingFaceModel.hf_from_composer_checkpoint`. If the tokenizer is not provided here, it will not be saved in the composer checkpoint.
        metrics (list[Metric], optional): list of torchmetrics to apply to the output of `eval_forward` during training. If ``eval_metrics`` is ``None``, these will also be used as ``eval_metrics``.  Default: ``None``.
        eval_metrics (list[Metric], optional): list of torchmetrics to compute on the eval_dataloader, or be accessible to :class:`Evaluator`s. Default: ``None``.
    .. warning:: This wrapper is designed to work with 🤗 datasets that define a `labels` column.
    """

    def __init__(self,
                 model: ParallelGPT,
                 tokenizer: Optional[Union[transformers.PreTrainedTokenizer,
                                           transformers.PreTrainedTokenizerFast]] = None,
                 metrics: Optional[List[Metric]] = None,
                 eval_metrics: Optional[List[Metric]] = None,
                 beam_width: int = 1,
                 top_k: int = 1,
                 top_p: float = 1.0,
                 temperature: float = 0.0,
                 len_penalty: float = 0,
                 beam_search_diversity_rate: float = 0.0,
                 min_length: int = 0,
                 presence_penalty: float = 0.0,
                 repetition_penalty: float = 0.0) -> None:
        if not FASTERTRANSFORMER_INSTALLED:
            raise ImportError('Could not import FasterTransformer dependencies.')

        super().__init__()
        self.model = model
        self.tokenizer = tokenizer

        if self.tokenizer is None:
            log.warning(
                'The tokenizer was not provided. This means the tokenizer config will not be saved in the checkpoint.')

        if tokenizer is not None and self.model.vocab_size < len(tokenizer):
                raise ValueError(
                    f'The number of tokens in the tokenizer is greater than the number of tokens in the model.'
                    f' This would cause an error during training.'
                    f' You can resize the model embeddings to {len(tokenizer)} from {self.model.vocab_size}.')
        elif tokenizer is not None and self.model.vocab_size > len(tokenizer):
            # when the embedding size is greater than the tokenizer vocab size,
            # the embeddings do not _need_ to be resized to match the tokenizer vocab size,
            # and should be done by the user if desired
            log.info(
                f'The number of tokens in the tokenizer is less than the number of tokens in the model.'
                f' You may want to resize the model embeddings to {len(tokenizer)} from {self.model.vocab_size}'
                f' by calling `model.resize_token_embeddings(len(tokenizer))` before calling the `HuggingFaceModel`'
                f' constructor. The vocab size is sometimes intentionally set to a multiple of 32 or 64 to improve'
                f' performance.')


        self.train_metrics: Optional[Dict] = None
        self.val_metrics: Optional[Dict] = None

        if eval_metrics is not None:
            self.val_metrics = {metric.__class__.__name__: metric for metric in eval_metrics}
        if metrics is not None:
            self.train_metrics = {metric.__class__.__name__: metric for metric in metrics}
            # if eval_metrics is None, use the same metrics as train_metrics
            if eval_metrics is None:
                self.val_metrics = {metric.__class__.__name__: metric for metric in metrics}

        self.labels: Optional[torch.Tensor] = None  # set in eval_forward() if exists

        # Arguments for generation
        self.beam_width = beam_width
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature
        self.beam_search_diversity_rate = beam_search_diversity_rate
        self.min_length = min_length
        self.len_penalty = len_penalty
        self.presence_penalty = presence_penalty
        self.repetition_penalty = repetition_penalty


    def forward(self, batch):
        raise ValueError("FasterTransformerInferenceModel does not support .forward() because this model cannot be used to train. \
            Please call .eval_forward() if you wish to evaluate this model, or create a new class if you wish to train a Composer FasterTransformer model.")

    def loss(self, outputs, batch):
        return outputs

    def eval_forward(self, batch, outputs: Optional[Any] = None):

        # Set FasterTransformer inference arguments
        if batch.get('input_ids', None) != None:
            batch_size = batch['input_ids'].shape[0]
            batch['input_ids'] = batch['input_ids'].to(dtype=torch.int32)
            start_lengths = torch.IntTensor(np.ones(shape=(batch_size))* batch['input_ids'].shape[1])
            #print("Start lengths:", start_lengths)
            #print("Batch input ids shape:", batch['input_ids'].shape)
            #print("Batch input ids dtype:", batch['input_ids'].dtype)

        else:
            start_lengths = torch.IntTensor([batch.shape[1] * batch_size])

        repetition_penalty_vec = None if self.repetition_penalty == 1. else self.repetition_penalty * torch.ones(
            batch_size, dtype=torch.float32)
        presence_penalty_vec = None if self.presence_penalty == 0. else self.presence_penalty * torch.ones(
            batch_size, dtype=torch.float32)

        infer_decode_args = {
            'beam_width':
                self.beam_width,
            'top_k':
                self.top_k * torch.ones(batch_size, dtype=torch.int32),
            'top_p':
                self.top_p * torch.ones(batch_size, dtype=torch.float32),
            'temperature':
                self.temperature * torch.ones(batch_size, dtype=torch.float32),
            'repetition_penalty':
                repetition_penalty_vec,
            'presence_penalty':
                presence_penalty_vec,
            'beam_search_diversity_rate':
                self.beam_search_diversity_rate *
                torch.ones(batch_size, dtype=torch.float32),
            'len_penalty':
                self.len_penalty * torch.ones(size=[batch_size], dtype=torch.float32),
            'bad_words_list':
                None,
            'min_length':
               self.min_length * torch.ones(size=[batch_size], dtype=torch.int32),
            'random_seed':
                torch.zeros([batch_size], dtype=torch.int64)
        }

        # If the batch mode is generate, we will generate a requested number of tokens using the underlying
        # model's generate function. Extra generation kwargs can be passed in via the batch. Strings will
        # be returned from eval_forward
        if batch.get('mode', None) == 'generate':
            if self.tokenizer is None:
                raise ValueError(
                    'Generation eval cannot be used without providing a tokenizer to the model constructor.')

            """
            self.labels = batch.pop('labels')
            ft_outputs = self.model(
                start_ids=batch['input_ids'], 
                start_lengths=start_lengths,
                output_len=batch['generation_length'],
                return_output_length=True,
                return_cum_log_probs=1,
                **infer_decode_args)
            output_token_ids, _, _ = ft_outputs
            # Return string
            return self.tokenizer.batch_decode(output_token_ids[:, batch['input_ids'].shape[1]:], skip_special_tokens=True)
            """

            raise ValueError('Cannot run `generate` batches. Please try with `icl_task`.')

        elif batch.get('mode', None) == 'icl_task':
            self.labels = batch.pop('labels')
            for i in range(batch_size):
                print("Inputs:", batch['input_ids'][i][0:80])
                inputs_tokenized = self.tokenizer.decode(batch['input_ids'][i][0:100])
                print("Inputs Tokenized:", inputs_tokenized)
                print("Continuation Indices:", batch['continuation_indices'][i])
            
            input_lens = [batch['continuation_indices'][i][0] for i in range(batch_size)]
            output_lens = [batch['continuation_indices'][i].shape[0] for i in range(batch_size)]

            input_lens_tensor = torch.IntTensor(input_lens)
            max_output_len = max(output_lens)
            ft_outputs = self.model(
                start_ids=batch['input_ids'],
                start_lengths=input_lens_tensor,
                output_len=max_output_len, # either 0 or 1
                return_output_length=True,
                return_cum_log_probs=1,
                **infer_decode_args)
            print("FasterTransformer Outputs:", ft_outputs[0][0][0][0:100])
            outputs_tokenized = self.tokenizer.decode(ft_outputs[0][0][0][0:100])
            print("Outputs Tokenized:", outputs_tokenized)
            print("FasterTransformer Log Probs:", ft_outputs[2])
            quit()
            # Return logits
            # TODO: Understand difference between log probs and logits
            _, _, cum_log_probs = ft_outputs
            output = cum_log_probs
            # if we are in the single class case, then remove the classes dimension
            if output.ndim == 2 and output.shape[1] == 1:
                output = output.squeeze(dim=1)
            return cum_log_probs
        else:
            if outputs:
                output = outputs
            else:
                ft_outputs = self.model(
                                start_ids=batch,
                                start_lengths=start_lengths,
                                output_len=self.model.weights.max_seq_len,
                                return_output_length=True,
                                return_cum_log_probs=1,
                                **infer_decode_args)
                output_token_ids, _, _ = ft_outputs
                output = output_token_ids
        return output


    def get_metrics(self, is_train: bool = False) -> Dict[str, Metric]:
        if is_train:
            metrics = self.train_metrics
        else:
            metrics = self.val_metrics

        return metrics if metrics else {}

    def update_metric(self, batch: Any, outputs: Any, metric: Metric) -> None:
        if isinstance(metric, InContextLearningMetric) and batch.get('mode', None) == 'icl_task':
            assert self.labels is not None
            metric.update(batch, outputs, self.labels)
        else:
            metric.update(outputs, self.labels)  # pyright: ignore [reportGeneralTypeIssues]

