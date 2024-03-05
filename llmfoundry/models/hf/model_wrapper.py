# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Re-usable :class:`.ComposerModel` for LLM HF Models."""

from __future__ import annotations

import warnings
from collections import UserDict
from typing import TYPE_CHECKING, List, Mapping, Optional

import torch
import transformers
from composer.models.huggingface import HuggingFaceModel
from torchmetrics import Metric
from transformers import PreTrainedTokenizerBase
from transformers.utils.generic import ModelOutput

from llmfoundry.models.hf.hf_fsdp import prepare_hf_model_for_fsdp
from llmfoundry.utils.warnings import VersionedDeprecationWarning

if TYPE_CHECKING:
    from peft import PeftConfig

# HuggingFace hardcodes the ignore index to -100
_HF_IGNORE_INDEX = -100


class HuggingFaceModelWithZLoss(HuggingFaceModel):
    """Wrapper around HuggingFaceModel.

    This adds z-loss, which is used in some training contexts,
    and is a convenient way to patch features that are generically
    useful for HF models.
    See use of z_loss in PaLM: https://arxiv.org/abs/2204.02311v3, Section 5.
    Also, from https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666:
        Two uses of z_loss are:
        - To keep the logits from drifting too far from zero, which can cause
            unacceptable roundoff errors in bfloat16.
        - To encourage the logits to be normalized log-probabilities.

    Handles preparation for FSDP wrapping.
    """

    def __init__(self,
                 model: transformers.PreTrainedModel,
                 tokenizer: Optional[PreTrainedTokenizerBase] = None,
                 metrics: Optional[List[Metric]] = None,
                 eval_metrics: Optional[List[Metric]] = None,
                 z_loss: float = 0.0,
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
        self.z_loss = float(z_loss)
        if self.z_loss < 0.0:
            raise ValueError(f'z_loss(={z_loss}) cannot be negative.')

        # Note: We need to add the FSDP related attributes to the model AFTER the super init,
        # so that the (possible) embedding resizing doesn't destroy them
        prepare_hf_model_for_fsdp(self.model, init_device)

        # This provides support for meta initialization when using FSDP
        self.model.param_init_fn = lambda module: self.model._init_weights(
            module)

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
            loss, logits = outputs['loss'], outputs['logits']
        else:
            # loss is at index 0 in the output tuple, logits are at index 1
            loss, logits = outputs[:2]
        if self.z_loss == 0.0:
            return loss

        warnings.warn(
            VersionedDeprecationWarning('z-loss is deprecated.',
                                        remove_version='0.7.0'))

        # Add a z_loss to the standard loss
        logits_flat = logits.view(-1, logits.size(-1))
        labels_flat = batch['labels'].view(-1)
        log_z = torch.logsumexp(logits_flat[labels_flat != _HF_IGNORE_INDEX],
                                dim=1)
        log_z2 = log_z**2
        z_loss = log_z2.mean() * self.z_loss
        if self.config.use_return_dict:
            outputs['loss'] += z_loss
            return outputs['loss']
        else:
            outputs[0] += z_loss
            return outputs[0]
