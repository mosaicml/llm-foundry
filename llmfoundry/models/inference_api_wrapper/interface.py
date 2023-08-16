
from typing import Any, Optional


import torch
from composer.metrics import InContextLearningMetric
# required for loading a python model into composer
from composer.metrics.nlp import (InContextLearningLMAccuracy,
                                  InContextLearningLMExpectedCalibrationError,
                                  InContextLearningMCExpectedCalibrationError,
                                  InContextLearningMultipleChoiceAccuracy,
                                  InContextLearningQAAccuracy,
                                  LanguageCrossEntropy, LanguagePerplexity)
from composer.models import ComposerModel
from torchmetrics import Metric


class InferenceAPIEvalWrapper(ComposerModel):
    def __init__(self, model_cfg, tokenizer):
        self.model_name = model_cfg['version']
        self.tokenizer = tokenizer
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
        super(InferenceAPIEvalWrapper, self).__init__()
        self.mocked_layer = torch.nn.Linear(2, 3)

    def get_metrics(self, is_train: bool = False):
        if is_train:
            metrics = []
        else:
            metrics = self.eval_metrics

        return metrics if metrics else {}

    def get_next_token_logit_tensor(self, prompt):
        raise NotImplementedError
    
    def rebatch(self, batch):
        # default is a no-op, but Chat API modifies these
        return batch

    def eval_forward(self, batch, outputs: Optional[Any] = None):
        # If the batch mode is generate, we will generate a requested number of tokens using the underlying
        # model's generate function. Extra generation kwargs can be passed in via the batch. Strings will
        # be returned from eval_forward
        output_logits_batch = []
        for tokens, cont_idxs in zip(batch['input_ids'],
                                    batch['continuation_indices']):
            
            seqlen = tokens.shape[0]
            tokens = tokens.tolist()
            cont_idxs = cont_idxs.tolist()
            expected_cont_tokens = tokens[cont_idxs[0]:cont_idxs[-1] + 1]
            output_logits = torch.nn.functional.one_hot(torch.tensor(tokens[1:cont_idxs[0]]), num_classes = self.tokenizer.pad_token_id + 1)
            for i in range(len(expected_cont_tokens)):
                # decode one token at a time
                prompt = self.tokenizer.decode(tokens[:cont_idxs[0]] +
                                            expected_cont_tokens[0:i])
                next_logit_tensor = self.get_next_token_logit_tensor(prompt)
                if next_logit_tensor is None:
                     continue
                output_logits = torch.cat(
                    [output_logits, next_logit_tensor.reshape(1, -1)])
            padding = torch.nn.functional.one_hot(torch.full((seqlen - output_logits.shape[0],), self.tokenizer.pad_token_id), num_classes = self.tokenizer.pad_token_id + 1)
            output_logits = torch.cat([
                output_logits,
                padding
            ])
            output_logits_batch.append(output_logits)

        return torch.stack(output_logits_batch).to(batch['input_ids'].device)

    def update_metric(self, batch: Any, outputs: Any, metric: Metric) -> None:
        batch = self.rebatch(batch)
        self.labels = batch.pop('labels')
        self.labels[:, :-1] = self.labels[:, 1:].clone()
        self.labels[:, -1] = -100
        if isinstance(metric, InContextLearningMetric) and batch.get(
                'mode', None) == 'icl_task':
            assert self.labels is not None            
            metric.update(batch, outputs, self.labels)
        else:
            metric.update(
                outputs,
                self.labels)  # pyright: ignore [reportGeneralTypeIssues]

    def forward(self):
        pass

    def loss(self):
        pass
