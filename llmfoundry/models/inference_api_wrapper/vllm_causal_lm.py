# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Implements a VLLM based chat and causal LM inference API wrappers."""

import logging
from typing import Any, Dict, Optional

import torch
import vllm
from composer.core.types import Batch
from torchmetrics import Metric
from transformers import AutoTokenizer

from llmfoundry.eval.metrics import InContextLearningMetric
from llmfoundry.metrics.token_acc import TokenAccuracy

log = logging.getLogger(__name__)

from llmfoundry.models.inference_api_wrapper.interface import \
    InferenceAPIEvalWrapper

__all__ = [
    'VLLMCausalLMEvalWrapper',
]


class VLLMEvalInterface(InferenceAPIEvalWrapper):

    def __init__(self, om_model_config: Dict, tokenizer: AutoTokenizer) -> None:
        super().__init__(om_model_config, tokenizer)
        kwargs = om_model_config.get('kwargs')
        self.vllm_engine = vllm.LLM(tokenizer=tokenizer.name_or_path, **kwargs)

        for name, metric in self.eval_metrics.items():
            if isinstance(metric, InContextLearningMetric) or isinstance(
                    metric, TokenAccuracy):
                # _to_sync is set to sync_on_compute in metric init
                self.eval_metrics[name]._to_sync = False


class VLLMCausalLMEvalWrapper(VLLMEvalInterface):

    def __init__(self, om_model_config: Dict, tokenizer: AutoTokenizer) -> None:
        super().__init__(om_model_config, tokenizer)
        self.om_model_config = om_model_config

    def eval_forward(self, batch: Batch, outputs: Optional[Any] = None):
        padding_tok = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id else self.tokenizer.eos_token_id

        # If the batch mode is generate, we will generate a requested number of tokens using the underlying
        # model's generate function. Extra generation kwargs can be passed in via the batch. Strings will
        # be returned from eval_forward

        if batch.get('mode', '') == 'generate':
            if 'generation_length' in batch:
                num_tokens = batch['generation_length']
            elif 'generation_kwargs' in batch:
                num_tokens = batch['generation_kwargs'].get('max_new_tokens', 2)
            num_sequences = batch.get('generation_kwargs',
                                      {}).get('num_return_sequences', 1)

            prompts = []
            for tokens, _ in zip(batch['input_ids'], batch['labels']):
                tokens = tokens.tolist()
                tokens = [t for t in tokens if t != padding_tok]
                prompts.append(tokens)

            sampling_params = vllm.SamplingParams(temperature=0,
                                                  max_tokens=num_tokens,
                                                  n=num_sequences)

            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=False):
                    results = self.vllm_engine.generate(
                        prompt_token_ids=prompts,
                        sampling_params=sampling_params,
                        use_tqdm=False)

            outputs = []
            for result in results:
                for output in result.outputs:
                    outputs.append(output.text)
            return outputs

        else:
            assert 'continuation_indices' in batch
            output_logits_batch = []
            prompts = []
            max_length = 0
            for tokens, cont_idxs in zip(batch['input_ids'],
                                         batch['continuation_indices']):
                prompts.append(tokens[0:cont_idxs[-1] + 1].tolist())
                max_length = max(max_length, cont_idxs[-1])

            sampling_params = vllm.SamplingParams(top_k=1, max_tokens=1)

            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=False):
                    results = self.vllm_engine.generate(
                        prompt_token_ids=prompts,
                        sampling_params=sampling_params,
                        use_tqdm=False)

            for tokens, cont_idxs, result in zip(batch['input_ids'],
                                                 batch['continuation_indices'],
                                                 results):
                result_seqlen, vocab_size = result.prompt_logits.shape
                assert result_seqlen == cont_idxs[-1] + 1

                result_logits = torch.narrow(result.prompt_logits, 0, 0,
                                             cont_idxs[-1])

                padding = torch.nn.functional.one_hot(
                    torch.full((max_length - cont_idxs[-1],), padding_tok),
                    num_classes=vocab_size,
                ).to(result_logits.device)
                output_logits = torch.cat([result_logits, padding])
                output_logits_batch.append(output_logits)

            return torch.stack(output_logits_batch).to(
                batch['input_ids'].device)
