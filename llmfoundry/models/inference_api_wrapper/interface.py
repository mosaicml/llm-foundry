# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, Optional

import torch
from composer.core.types import Batch
from composer.metrics import InContextLearningMetric
from composer.metrics.nlp import (InContextLearningLMAccuracy,
                                  InContextLearningLMExpectedCalibrationError,
                                  InContextLearningMCExpectedCalibrationError,
                                  InContextLearningMultipleChoiceAccuracy,
                                  InContextLearningQAAccuracy,
                                  LanguageCrossEntropy, LanguagePerplexity)
from composer.models import ComposerModel
from torchmetrics import Metric
from transformers import AutoTokenizer


class InferenceAPIEvalWrapper(ComposerModel):

    def __init__(self, model_cfg: Dict, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer
        self.labels = None
        # set up training and eval metrics
        eval_metrics = [
            LanguageCrossEntropy(),
            LanguagePerplexity(),
            InContextLearningLMAccuracy(),
            InContextLearningMultipleChoiceAccuracy(),
            InContextLearningQAAccuracy(),
            InContextLearningLMExpectedCalibrationError(),
            InContextLearningMCExpectedCalibrationError()
        ]
        self.eval_metrics = {
            metric.__class__.__name__: metric for metric in eval_metrics
        }
        super().__init__()

    def completion_to_string(self, completion: Any):
        raise NotImplementedError

    def get_metrics(self, is_train: bool = False):
        if is_train:
            metrics = None
        else:
            metrics = self.eval_metrics

        return metrics if metrics else {}

    def get_next_token_logit_tensor(self,
                                    prompt: str) -> Optional[torch.Tensor]:
        raise NotImplementedError

    def rebatch(self, batch: Batch):
        # default is a no-op, but Chat API modifies these
        return batch

    def eval_forward(self, batch: Batch, outputs: Optional[Any] = None):
        padding_tok = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id else self.tokenizer.eos_token_id
        # If the batch mode is generate, we will generate a requested number of tokens using the underlying
        # model's generate function. Extra generation kwargs can be passed in via the batch. Strings will
        # be returned from eval_forward

        if batch.get('mode', '') == 'generate':
            outputs = []
            # generate-based implementation
            for tokens, _ in zip(batch['input_ids'], batch['labels']):

                tokens = tokens.tolist()
                tokens = [t for t in tokens if t != padding_tok]
                prompt = self.tokenizer.decode(tokens)
                sample_outputs = []
                for i in range(
                        0,
                        batch.get('generation_kwargs',
                                  {}).get('num_return_sequences', 1)):
                    api_output = self.try_generate_completion(  #
                        prompt,
                        num_tokens=batch['generation_length'],
                        generation_kwargs=batch.get('generation_kwargs', {}))
                    sample_output = self.completion_to_string(api_output)[0]
                    sample_outputs.append(sample_output)
                if len(sample_outputs) == 1:
                    sample_outputs = sample_outputs[0]

                outputs.append(sample_outputs)
            return outputs
        else:
            # forward-based implementation
            output_logits_batch = []
            for tokens, cont_idxs in zip(batch['input_ids'],
                                         batch['continuation_indices']):

                seqlen = tokens.shape[0]
                tokens = tokens.tolist()
                cont_idxs = cont_idxs.tolist()
                expected_cont_tokens = tokens[cont_idxs[0]:cont_idxs[-1] + 1]
                output_logits = torch.nn.functional.one_hot(
                    torch.tensor(tokens[1:cont_idxs[0]]),
                    num_classes=len(self.tokenizer))
                for i in range(len(expected_cont_tokens)):
                    # decode one token at a time
                    prompt = self.tokenizer.decode(tokens[:cont_idxs[0]] +
                                                   expected_cont_tokens[0:i])
                    next_logit_tensor = self.get_next_token_logit_tensor(prompt)
                    if next_logit_tensor is None:
                        continue
                    output_logits = torch.cat(
                        [output_logits,
                         next_logit_tensor.reshape(1, -1)])
                padding = torch.nn.functional.one_hot(
                    torch.full((seqlen - output_logits.shape[0],), padding_tok),
                    num_classes=len(self.tokenizer))
                output_logits = torch.cat([output_logits, padding])
                output_logits_batch.append(output_logits)

            return torch.stack(output_logits_batch).to(
                batch['input_ids'].device)

    def update_metric(self, batch: Any, outputs: Any, metric: Metric) -> None:
        if isinstance(metric, InContextLearningMetric) and batch.get(
                'mode', None) == 'icl_task':
            batch = self.rebatch(batch)
            self.labels = batch.pop('labels')
            self.labels[:, :-1] = self.labels[:, 1:].clone()
            self.labels[:, -1] = -100
            assert self.labels is not None
            metric.update(batch, outputs, self.labels)
        elif isinstance(metric, InContextLearningMetric) and batch.get(
                'mode', None) == 'generate':
            self.labels = batch.pop('labels')
            assert self.labels is not None
            metric.update(batch=batch, outputs=outputs, labels=self.labels)
        else:
            raise NotImplementedError(
                'Inference API wrapper only supports InContextLearningMetrics and mode=icl_task'
            )

    def forward(self):
        raise NotImplementedError(
            "Inference API wrapper doesn't support forward")

    def loss(self):
        raise NotImplementedError("Inference API wrapper doesn't support loss")
