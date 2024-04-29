# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Re-usable :class:`.ComposerModel` for LLM HF Models."""

from __future__ import annotations

from collections import UserDict
from typing import TYPE_CHECKING, List, Mapping, Optional, Union

import transformers
from composer.models.huggingface import HuggingFaceModel
from torchmetrics import Metric
from transformers import PreTrainedTokenizerBase
from transformers.utils.generic import ModelOutput

from llmfoundry.models.hf.hf_fsdp import prepare_hf_model_for_fsdp

if TYPE_CHECKING:
    from peft import PeftConfig, PeftModel

__all__ = ['HuggingFaceModelWithFSDP']

# HuggingFace hardcodes the ignore index to -100
_HF_IGNORE_INDEX = -100


class HuggingFaceModelWithFSDP(HuggingFaceModel):
    """Wrapper around HuggingFaceModel.

    Handles preparation for FSDP wrapping.
    """

    def __init__(self,
                 model: Union[transformers.PreTrainedModel, 'PeftModel'],
                 tokenizer: Optional[PreTrainedTokenizerBase] = None,
                 metrics: Optional[List[Metric]] = None,
                 eval_metrics: Optional[List[Metric]] = None,
                 shift_labels: bool = False,
                 init_device: Optional[str] = None,
                 peft_config: Optional['PeftConfig'] = None):
        super().__init__(
            model,
            tokenizer,
            use_logits=True,
            metrics=metrics,
            eval_metrics=eval_metrics,
            shift_labels=shift_labels,
            peft_config=peft_config,
            should_save_peft_only=True,
        )

        self.prepare_inner_model(self.model, init_device)

    def forward(self, batch: Mapping):
        if isinstance(batch, dict) or isinstance(batch, UserDict):
            # Further input validation is left to the huggingface forward call
            batch = {
                k: v for k, v in batch.items() if k in self.model_forward_args
            }
            output = self.model(**batch)  # type: ignore (thirdparty)
        else:
            raise ValueError(
                'Unexpected batch type. Expected a dictionary with keys corresponding to the inputs to the forward function of the Huggingface model'
            )
        return output

    def loss(self, outputs: ModelOutput, batch: Mapping):
        if self.config.use_return_dict:
            return outputs['loss']
        # loss is at index 0 in the output tuple, logits are at index 1
        return outputs[:2]

    @staticmethod
    def prepare_inner_model(model: Union[transformers.PreTrainedModel,
                                         'PeftModel'],
                            init_device: Optional[str] = None):
        """Prepare the inner model for FSDP wrapping.
        
        Args:
            model: The model to prepare.
            init_device: The device to initialize the model on."""
        # Note: We need to add the FSDP related attributes to the model AFTER the super init,
        # so that the (possible) embedding resizing doesn't destroy them
        prepare_hf_model_for_fsdp(model, init_device)

        # This provides support for meta initialization when using FSDP
        model.param_init_fn = lambda module: model._init_weights(module)
