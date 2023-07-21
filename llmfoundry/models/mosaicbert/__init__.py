# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

import torch

from llmfoundry.models.mosaicbert.modeling_mosaicbert import (BertForMaskedLM,
                                    BertForSequenceClassification, BertModel,
                                    ComposerBertForMaskedLM)
from llmfoundry.models.mosaicbert.configuration_mosaicbert import BertConfig
# yapf: disable
from llmfoundry.models.layers import (BertEmbeddings, BertEncoder,
                                      BertGatedLinearUnitMLP, BertLayer,
                                      BertLMPredictionHead, BertOnlyMLMHead,
                                      BertOnlyNSPHead, BertPooler,
                                      BertPredictionHeadTransform,
                                      BertSelfOutput, BertUnpadAttention,
                                      BertUnpadSelfAttention)
# yapf: enable
from llmfoundry.models.utils.bert_padding import (IndexFirstAxis,
                                                  IndexPutFirstAxis,
                                                  index_first_axis,
                                                  index_put_first_axis,
                                                  pad_input, unpad_input,
                                                  unpad_input_only)

if torch.cuda.is_available():
    from llmfoundry.models.layers.flash_attn_triton import \
        flash_attn_func as flash_attn_func_bert  # type: ignore
    from llmfoundry.models.layers.flash_attn_triton import \
        flash_attn_qkvpacked_func as \
        flash_attn_qkvpacked_func_bert  # type: ignore

__all__ = [
    'BertConfig',
    'BertEmbeddings',
    'BertEncoder',
    'BertForMaskedLM',
    'BertForSequenceClassification',
    'BertGatedLinearUnitMLP',
    'BertLayer',
    'BertLMPredictionHead',
    'BertModel',
    'BertOnlyMLMHead',
    'BertOnlyNSPHead',
    'BertPooler',
    'BertPredictionHeadTransform',
    'BertSelfOutput',
    'BertUnpadAttention',
    'BertUnpadSelfAttention',
    'ComposerBertForMaskedLM',
    'IndexFirstAxis',
    'IndexPutFirstAxis',
    'index_first_axis',
    'index_put_first_axis',
    'pad_input',
    'unpad_input',
    'unpad_input_only',
    # These are commented out because they only exist if CUDA is available
    # 'flash_attn_func_bert',
    # 'flash_attn_qkvpacked_func_bert'
]
