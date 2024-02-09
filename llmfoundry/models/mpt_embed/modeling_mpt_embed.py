# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""A simple, flexible implementation of a GPT model.

Inspired by https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
"""

import math
import warnings
from typing import (Any, Dict, List, Mapping, MutableMapping, Optional, Tuple,
                    Union)

import torch
import torch.nn as nn
import torch.nn.functional as F
from composer.metrics import (InContextLearningCodeEvalAccuracy,
                              InContextLearningLMAccuracy,
                              InContextLearningLMExpectedCalibrationError,
                              InContextLearningMCExpectedCalibrationError,
                              InContextLearningMultipleChoiceAccuracy,
                              InContextLearningQAAccuracy,
                              LossMetric) # JP Added
from composer.metrics.nlp import LanguageCrossEntropy, LanguagePerplexity
from composer.models import HuggingFaceModel
from composer.utils import dist
from omegaconf import DictConfig
from omegaconf import OmegaConf as om
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from transformers.modeling_outputs import (BaseModelOutputWithPast,
                                           CausalLMOutputWithPast)

from llmfoundry.models.layers.attention import attn_bias_shape, build_attn_bias
from llmfoundry.models.layers.blocks import MPTBlock
from llmfoundry.models.layers.custom_embedding import SharedEmbedding
from llmfoundry.models.layers.fc import FC_CLASS_REGISTRY as FC_CLASS_REGISTRY
from llmfoundry.models.layers.ffn import \
    FFN_CLASS_REGISTRY as FFN_CLASS_REGISTRY
from llmfoundry.models.layers.ffn import MPTMLP as MPTMLP
from llmfoundry.models.layers.ffn import build_ffn as build_ffn
from llmfoundry.models.layers.norm import NORM_CLASS_REGISTRY
from llmfoundry.models.mpt.configuration_mpt import MPTConfig

# NOTE: All utils are imported directly even if unused so that
# HuggingFace can detect all the needed files to copy into its modules folder.
# Otherwise, certain modules are missing.
# isort: off
from llmfoundry.models.utils.adapt_tokenizer import (
    AutoTokenizerForMOD,  # type: ignore (see note),
    adapt_tokenizer_for_denoising,  # type: ignore (see note)
)
from llmfoundry.models.utils.hf_prefixlm_converter import (
    add_bidirectional_mask_if_missing,  # type: ignore (see note)
    convert_hf_causal_lm_to_prefix_lm,  # type: ignore (see note)
)
from llmfoundry.models.utils.meta_init_context import \
    init_empty_weights  # type: ignore (see note)
from llmfoundry.models.utils.param_init_fns import (
    generic_param_init_fn_,  # type: ignore (see note)
    MODEL_INIT_REGISTRY,
)

try:
    from llmfoundry.models.layers.flash_attn_triton import flash_attn_func as flash_attn_func
except:
    pass
# isort: on

import logging

# JP: imports from mpt/modeling_mpt
from llmfoundry.models.mpt.modeling_mpt import MPTPreTrainedModel, MPTModel, MPTForCausalLM, ComposerMPTCausalLM

log = logging.getLogger(__name__)

# JP: Added for contrastive loss
# Todo: does this exist somewhere? Move this to a util?
def dist_gather_tensor(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:

    """
    This function applies an all gather operation necessary for the contrastive 
    InfoNCELoss applied in the ComposerMPTContrastiveLM class
    """
    if t is None:
        return None

    t = t.contiguous()
    all_tensors = [torch.empty_like(t) for _ in range(dist.get_world_size())]
    torch.distributed.all_gather(all_tensors, t)

    all_tensors[dist.get_global_rank()] = t
    all_tensors = torch.cat(all_tensors, dim=0)
    return all_tensors


class ComposerMPTContrastiveLM(HuggingFaceModel):

    """ 
    JP: The following code implements a contrastive loss function using the MPT
    architecture.

    Note that the MPTForCausalLM architecture can be used as is, provided that `labels=None`
    Most of the code in this class is modeled off ComposerMPTCausalLM
    In the init, we use the MPTForCausalLM as the main model

    attention mask

    bidirectional mask

    The main addition is the function _compute_scores() in the forward pass, which implements the contrastive loss
    """

    def __init__(
        self,
        om_model_config: DictConfig,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
    ):
        resolved_om_model_config = om.to_container(om_model_config,
                                                   resolve=True)
        hf_config = MPTConfig.from_dict(resolved_om_model_config)
        model = MPTForCausalLM(hf_config)

        use_train_metrics = om_model_config.get('use_train_metrics', True)
        
        # train_metrics = [LanguageCrossEntropy(),
        #                  LanguagePerplexity()] if use_train_metrics else []

        # JP Add
        train_metrics = [LanguageCrossEntropy()] if use_train_metrics else []

        # JP: These metrics might not work for the contrastive loss
        # eval_metrics = [
        #     LanguageCrossEntropy(),
        #     LanguagePerplexity(),
        #     InContextLearningLMAccuracy(),
        #     InContextLearningMultipleChoiceAccuracy(),
        #     InContextLearningQAAccuracy(),
        #     InContextLearningCodeEvalAccuracy(),
        #     InContextLearningLMExpectedCalibrationError(),
        #     InContextLearningMCExpectedCalibrationError(),
        # ]
        eval_metrics = []

        super().__init__(
            model=model,
            tokenizer=tokenizer,
            use_logits=False, # JP: set to False
            metrics=train_metrics,
            eval_metrics=eval_metrics, # might be worth setting to []
            shift_labels=False, # JP: set to False
            allow_embedding_resizing=True, # JP: Not sure what this does. was set to True
        )

        # Temperature for InfoNCELoss
        self.temperature = resolved_om_model_config.get('temperature', 1)

        # Set the vector representation to either be the average of all the individual token vectors,
        # or the EOS token at the end of the sequence
        self.vector_representation = resolved_om_model_config.get('contrastive_config',
                                                                  {}).get('vector_representation','avg')
        self.normalize_output = resolved_om_model_config.get('contrastive_config',
                                                                  {}).get('normalize_output',True)
        
        step_size = self.config.to_dict().get("pos_step_size", 2)
        self.pattern = torch.arange(1, step_size, device=model.device)

        self.n_active_params = sum(p.numel() for p in self.parameters())
        loss_fn_config = om_model_config.get('loss_fn', 'fused_crossentropy')
        if loss_fn_config == 'fused_crossentropy':
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
                    'if installing from pypi, or (2) set your config model.loss_fn=torch_crossentropy.'
                )
        elif loss_fn_config == 'torch_crossentropy':
            self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        else:
            raise ValueError(
                f'Specified loss_fn={self.loss_fn} not recognized. `loss_fn` must be one of [`fused_crossentropy`, `torch_crossentropy`].'
            )

    def format_queries_batch(self, batch, last_hidden_state):
        """ Format `queries` by selecting every other pair from the batch 
        JP: Note that there could be a better way to do this
        """
        queries = {}
        step_size = self.config.to_dict().get("pos_step_size", 2)
        for key in batch.keys():
            queries[key] = batch[key][0::step_size, :] # no need to reshape.reshape(batch[key].size(0), -1)
            
        return queries, last_hidden_state[0::step_size, :, :]
    
    def format_passages_batch(self, batch, last_hidden_state):
        """ Format `passages` by selecting every other pair from the batch """
        passages = {}
        step_size = self.config.to_dict().get("pos_step_size", 2)

        # Index on a variable step size
        for key in batch.keys():
            num_blocks = batch[key].size(0) // step_size
            index = torch.arange(1, num_blocks * step_size + 1, device=last_hidden_state.device).view(num_blocks, step_size)
            index = index[:, :step_size - 1].reshape(-1)
            passages[key] = batch[key][index]
        
        return passages, last_hidden_state[index, :, :]

    def forward(self, batch: MutableMapping) -> CausalLMOutputWithPast:
        if self.model.transformer.prefix_lm:
            add_bidirectional_mask_if_missing(batch)
        # Note: prefix_mask is only used if model.prefix_lm is True

        # JP - add
        # Reshape pairs so that 
        dim1,dim2,dim3=batch['input_ids'].shape
        # print(batch['input_ids'].shape)
        # print(batch['input_ids'].reshape((dim1*dim2,dim3)))
        reshaped_input_ids = batch['input_ids'].reshape((dim1*dim2,dim3))
        batch['input_ids'] = reshaped_input_ids
        
        # still need TO DO for prefix_mask and input_embeds
        if 'attention_mask' in batch:
            reshaped_attention_mask = batch['attention_mask'].reshape((dim1*dim2,dim3))
            batch['attention_mask'] = reshaped_attention_mask
            # print('attention mask shape: ', batch['attention_mask'].shape)
        if 'prefix_mask' in batch:
            reshaped_prefix_mask = batch['prefix_mask'].reshape((dim1*dim2,dim3))
            batch['prefix_mask'] = reshaped_prefix_mask
        if 'sequence_id' in batch:
            reshaped_sequence_id = batch['sequence_id'].reshape((dim1*dim2,dim3))
            batch['sequence_id'] = reshaped_sequence_id
        if 'input_embeds' in batch:
            reshaped_input_embeds = batch['input_embeds'].reshape((dim1*dim2,dim3))
            batch['input_embeds'] = reshaped_input_embeds
        
        if 'labels' in batch:
            reshaped_labels = batch['labels'].reshape((dim1*dim2,dim3))
            batch['labels'] = reshaped_labels

        return self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch.get('attention_mask', None),
            prefix_mask=batch.get('prefix_mask', None),
            sequence_id=batch.get('sequence_id', None),
            inputs_embeds=batch.get('inputs_embeds', None),
        )
    
    def _compute_scores(self, batch, outputs) -> Tuple:

        """
        Run Pairs through the encoder separately in two passes, designated as q (query) and p (passage)
        [batch_size, sequence_length]
        
        the pooled_outputs is [batch_size, hidden_size]
        
        Note: at some future point we could use the flag 'token_type_ids' which was used in the original
        BERT formula to keep track of sentences A and sentences B in the next sentence prediction objective
        function. For now we split even and odd rows
        """
        (queries_batch, queries_last_hidden_state) = self.format_queries_batch(batch, outputs.hidden_states)
        (passages_batch, passages_last_hidden_state) = self.format_passages_batch(batch, outputs.hidden_states)
        
        
        # print(self.tokenizer.decode(queries_batch['input_ids'][0]))
        # print(self.tokenizer.decode(passages_batch['input_ids'][0]))

        # the output of self.model should be a @dataclass container with values
        # loss, logits, attentions, and hidden_states, which contains Hidden-states of the model at the 
        # output of each layer plus the optional initial embedding outputs.
        #
        # in order to access the final output of the hidden states, we can do hidden_states[-1]
        
        # JP: Note we went from p_encoder_outputs.masked_fill to q_encoder_outputs.hidden_states.masked_fill
        # Might need to do hidden_states[-1] to only select activations from the last layer
        #
        # Note that we do _not_ want to use the logits; we want the full output dimension to be the hidden_dim
        # for batch of [16,128], q_encoder_outputs.hidden_states has shape [16,128,768] but q_encoder_outputs.hidden_states[-1].shape:  torch.Size([128, 768])
        # q_last_hidden also has shape [16,128,768]. q_pooled_outputs should be [16,768]
        
        # print('\nq_encoder_outputs.hidden_states[-1].shape: ', q_encoder_outputs.hidden_states[-1].shape)
        if self.vector_representation == 'eos':

            # This assumes that that each query and passage ends with an EOS token
            
            # create an array with indicees corresponding to rows along the batch dimension
            row_indices = torch.arange(queries_batch.get('attention_mask', None).shape[0])

            # need to take the last token in the attention mask that is not padded
            # flip the mask and then find argmax
            queries_flipped_mask = ~queries_batch.get('attention_mask', None).bool()
            last_true_indices = queries_flipped_mask.int().argmax(dim=1) - 1 # need to subtract 1 because it is findinng the first instance of the padded token
            q_pooled_outputs = queries_last_hidden_state[row_indices,last_true_indices,:] # b,s,h --> b,h

            # This is another way of achieving the same result
            # q_pooled_outputs = torch.stack([ queries_last_hidden_state[i, j, :] for i, j in enumerate(queries_batch.get('attention_mask', None).sum(dim=1) - 1)])

            passages_flipped_mask = ~passages_batch.get('attention_mask', None).bool()
            last_true_indices = passages_flipped_mask.int().argmax(dim=1) - 1
            p_pooled_outputs = passages_last_hidden_state[row_indices,last_true_indices,:] # b,s,h --> b,h

        else: 

            # Default to taking the average of all the vectors for each token in the sequence
            q_last_hidden = queries_last_hidden_state.masked_fill(~queries_batch.get('attention_mask', None)[..., None].bool(), 0.0)
            q_pooled_outputs = q_last_hidden.sum(dim=1) / queries_batch.get('attention_mask', None).sum(dim=1)[..., None]
            
            p_last_hidden = passages_last_hidden_state.masked_fill(~passages_batch.get('attention_mask', None)[..., None].bool(), 0.0)
            p_pooled_outputs = p_last_hidden.sum(dim=1) / passages_batch.get('attention_mask', None).sum(dim=1)[..., None]


            
        #print('>>p_pooled_outputs shape:',p_pooled_outputs.shape)
        
        if self.normalize_output:
            q_pooled_outputs = F.normalize(q_pooled_outputs, dim=-1) # Todo: should be configurable when L2 normalizing
            p_pooled_outputs = F.normalize(p_pooled_outputs, dim=-1)

        q_pooled_outputs = q_pooled_outputs.contiguous() # Why do we need to make this contiguous?
        p_pooled_outputs = p_pooled_outputs.contiguous() # Why do we need to make this contiguous?

        is_distributed = self.config.to_dict().get("is_distributed", True)
        
        # if q_pooled_outputs.shape[0] < p_pooled_outputs.shape[0]:
        #     q_pooled_outputs = q_pooled_outputs.repeat(p_pooled_outputs.shape[0], 1)
        
        if is_distributed:
            # All Gather is included
            all_q_pooled_outputs = dist_gather_tensor(q_pooled_outputs)
            all_p_pooled_outputs = dist_gather_tensor(p_pooled_outputs)
        else:
            # No All Gather
            all_q_pooled_outputs = q_pooled_outputs
            all_p_pooled_outputs = p_pooled_outputs
        
        all_scores, all_labels = self.full_contrastive_scores_and_labels(queries=all_q_pooled_outputs, 
                                                                         passages=all_p_pooled_outputs)
        
        all_labels = torch.arange(all_scores.size(0), device=q_pooled_outputs.device, dtype=torch.long)
        all_labels = all_labels * (p_pooled_outputs.size(0) // q_pooled_outputs.size(0))
        
        scale = 1 / self.temperature
        
        all_scores = all_scores * scale
        
        # start = dist.get_global_rank() * q_pooled_outputs.shape[0]
        
        # local_query_indices = torch.arange(start, start + q_pooled_outputs.shape[0], dtype=torch.long).to(q_pooled_outputs.device)
        
        # scores = all_scores.index_select(dim=0, index=local_query_indices)
        # all_scores[dist.get_global_rank()] = scores
        # labels = all_labels.index_select(dim=0, index=local_query_indices)s
        # scores = all_scores
        # labels = all_labels
        #print('>>labels',labels.shape) # should be torch.Size([64])

        #print('>> mean scores',all_scores.mean())
        #print('>> mean diagonal scores', all_scores.diagonal().mean())
        return all_scores, all_labels
    
    def full_contrastive_scores_and_labels(self, queries: torch.Tensor, passages: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        # the labels 
        labels = torch.arange(0, passages.shape[0], dtype=torch.long, device=passages.device)
        
        # this calculates the inner product between query and passage pairs
        qp = torch.mm(queries, passages.t())

        #print('>> qp shape:', qp.shape)

        return qp, labels

    def loss(self, outputs: CausalLMOutputWithPast,
             batch: Mapping) -> torch.Tensor:
        
        scores, labels = self._compute_scores(batch, outputs)
                
        loss = self.loss_fn(scores, labels)

        self.labels = labels # JP Added, necessary for train metrics LanguageCrossEntropy
        # Note that LanguageCrossEntropy() calculates loss with respect to logits
        # e.g. losses = self.loss_fn(logits, target)
        # This is different from how we are calculating the loss between the output vectors of query vs passage

        
        # Based on https://github.com/microsoft/unilm/blob/b60c741f746877293bb85eed6806736fc8fa0ffd/simlm/src/models/biencoder_model.py#L60C62-L60C62
        # We are scaling the loss by the world size because we think it will be divided by the world size in the backward pass
        # This is a hacky way of getting around implementing our own backward pass
        # loss *= dist.get_world_size()
        
        return loss # do we also need to pass the labels and the scores?
    
        # {
        #     'loss': loss,
        #     'logits': scores, # This doesn't seem right, but needs to be here for torchmetrics
        #     'labels': labels
        # }

    def flops_per_batch(self, batch: Mapping) -> int:
        # Note: this computation does not take into account padding, and assumes
        # that the dataset has been constructed without padding. Additionally, we
        # assume the backward pass is approximately 2x the forward pass

        bs, msl = batch['input_ids'].shape[0:2]
        params_flops_per_token = 2 * self.n_active_params
        params_flops_per_seq = params_flops_per_token * msl
        attn_flops_per_seq = (self.model.config.n_layers * 2 * 2 *
                              (self.model.config.d_model * (msl**2)))

        return (params_flops_per_seq + attn_flops_per_seq) * 3 * bs
