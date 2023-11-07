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
                              InContextLearningQAAccuracy)
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

log = logging.getLogger(__name__)

# JP: Added for contrastive loss
# Todo: does this exist somewhere? Move this to a util?
def dist_gather_tensor(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if t is None:
        return None

    t = t.contiguous()
    all_tensors = [torch.empty_like(t) for _ in range(dist.get_world_size())]
    torch.distributed.all_gather(all_tensors, t)

    all_tensors[dist.get_global_rank()] = t
    all_tensors = torch.cat(all_tensors, dim=0)
    return all_tensors


class MPTPreTrainedModel(PreTrainedModel):
    config_class = MPTConfig
    base_model_prefix = 'model'
    _no_split_modules = ['MPTBlock']


class MPTModel(MPTPreTrainedModel):

    def __init__(self, config: MPTConfig):
        config._validate_config()
        super().__init__(config)

        self.attn_impl = config.attn_config['attn_impl']
        self.prefix_lm = config.attn_config['prefix_lm']
        self.attn_uses_sequence_id = config.attn_config['attn_uses_sequence_id']
        self.alibi = config.attn_config['alibi']
        self.alibi_bias_max = config.attn_config['alibi_bias_max']

        self.learned_pos_emb = config.learned_pos_emb

        if config.init_device == 'mixed':
            if dist.get_local_rank() == 0:
                config.init_device = 'cpu'
            else:
                config.init_device = 'meta'

        # Configure layernorm, low_precision_layernorm, rmsnorm
        # or low_precision_rmsnorm
        if config.norm_type.lower() not in NORM_CLASS_REGISTRY.keys():
            norm_options = ' | '.join(NORM_CLASS_REGISTRY.keys())
            raise NotImplementedError(
                f'Requested norm type ({config.norm_type}) is not implemented within this repo (Options: {norm_options}).'
            )
        norm_class = NORM_CLASS_REGISTRY[config.norm_type.lower()]

        # CogView (https://arxiv.org/abs/2105.13290) and GLM-130B (https://arxiv.org/abs/2210.02414)
        # both report this helping with stabilizing training
        self.embedding_fraction = config.embedding_fraction

        self.wte = SharedEmbedding(config.vocab_size,
                                   config.d_model,
                                   device=config.init_device)
        if self.learned_pos_emb:
            self.wpe = torch.nn.Embedding(config.max_seq_len,
                                          config.d_model,
                                          device=config.init_device)
        self.emb_drop = nn.Dropout(config.emb_pdrop)
        self.blocks = nn.ModuleList([
            MPTBlock(
                device=config.init_device,
                **config.to_dict(),
            ) for _ in range(config.n_layers)
        ])
        self.norm_f = norm_class(config.d_model, device=config.init_device)

        if config.init_device != 'meta':
            log.info(
                f'We recommend using config.init_device="meta" with Composer + FSDP for faster initialization.'
            )
            self.apply(self.param_init_fn)

        self.is_causal = not self.prefix_lm

        # define attn mask
        self._attn_bias_initialized = False
        self.attn_bias = None
        self.attn_bias_shape = attn_bias_shape(
            self.attn_impl,
            config.n_heads,
            config.max_seq_len,
            self.alibi,
            prefix_lm=self.prefix_lm,
            causal=self.is_causal,
            use_sequence_id=self.attn_uses_sequence_id,
        )

        if config.no_bias:
            for module in self.modules():
                if hasattr(module, 'bias') and isinstance(
                        module.bias, nn.Parameter):
                    log.info(f'Removing bias ({module.bias}) from {module}.')
                    module.register_parameter('bias', None)

                # For transformer engine
                if hasattr(module, 'use_bias'):
                    log.info(f'Setting use_bias=False for {module}.')
                    module.use_bias = False

        log.debug(self)
        log.debug(f'Using {self.config.init_config["name"]} initialization.')

    def get_input_embeddings(self) -> nn.Embedding:
        return self.wte

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.wte = value

    @torch.no_grad()
    def _attn_bias(
        self,
        device: torch.device,
        dtype: torch.dtype,
        attention_mask: Optional[torch.ByteTensor] = None,
        prefix_mask: Optional[torch.ByteTensor] = None,
        sequence_id: Optional[torch.LongTensor] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.ByteTensor]]:
        if not self._attn_bias_initialized:
            if self.attn_bias_shape:
                self.attn_bias = torch.zeros(self.attn_bias_shape,
                                             device=device,
                                             dtype=dtype)
                self.attn_bias = build_attn_bias(
                    self.attn_impl,
                    self.attn_bias,
                    self.config.n_heads,
                    self.config.max_seq_len,
                    causal=self.is_causal,
                    alibi=self.alibi,
                    alibi_bias_max=self.alibi_bias_max,
                )
            self._attn_bias_initialized = True

        # flash does not support prefix_lm and will incorporate any
        # attention_mask inside the attention module
        if self.attn_impl == 'flash':
            return self.attn_bias, attention_mask

        if self.attn_bias is not None:
            # .to(*args, **kwargs) is a no-op if tensor is already on
            # specified device or of specified dtype
            self.attn_bias = self.attn_bias.to(dtype=dtype, device=device)

        attn_bias = self.attn_bias

        # If using torch or triton, we incorporate the prefix_mask (if appropriate)
        if self.prefix_lm:
            assert isinstance(attn_bias, torch.Tensor)  # pyright
            assert isinstance(prefix_mask, torch.Tensor)  # pyright
            attn_bias = self._apply_prefix_mask(attn_bias, prefix_mask)

        # If using torch or triton, we incorporate sequence_id (if appropriate)
        if self.attn_uses_sequence_id and sequence_id is not None:
            assert isinstance(attn_bias, torch.Tensor)  # pyright
            attn_bias = self._apply_sequence_id(attn_bias, sequence_id)

        # If using torch or triton, we incorporate attention_mask. This will output
        # None in place of attention_mask since it will not be further needed in the
        # attention modules.
        if attention_mask is not None:
            s_k = attention_mask.shape[-1]
            if attn_bias is None:
                attn_bias = torch.zeros((1, 1, 1, s_k),
                                        device=device,
                                        dtype=dtype)
            else:
                # clamp to 0 necessary for torch 2.0 compile()
                _s_k = max(0, attn_bias.size(-1) - s_k)
                attn_bias = attn_bias[:, :, :, _s_k:]
            if prefix_mask is not None and (attention_mask.shape !=
                                            prefix_mask.shape):
                raise ValueError(
                    f'attention_mask shape={attention_mask.shape} ' +
                    f'and prefix_mask shape={prefix_mask.shape} are not equal.')
            min_val = torch.finfo(attn_bias.dtype).min
            attn_bias = attn_bias.masked_fill(
                ~attention_mask.view(-1, 1, 1, s_k), min_val)

        return attn_bias, None

    def _apply_prefix_mask(self, attn_bias: torch.Tensor,
                           prefix_mask: torch.Tensor) -> torch.Tensor:
        s_k, s_q = attn_bias.shape[-2:]
        if (s_k != self.config.max_seq_len) or (s_q != self.config.max_seq_len):
            raise ValueError(
                'attn_bias does not match the expected shape. ' +
                f'The last two dimensions should both be {self.config.max_length} '
                + f'but are {s_k} and {s_q}.')
        seq_len = prefix_mask.shape[-1]
        if seq_len > self.config.max_seq_len:
            raise ValueError(
                f'prefix_mask sequence length cannot exceed max_seq_len={self.config.max_seq_len}'
            )

        # select seq_len subset of attn mask
        attn_bias = attn_bias[..., :seq_len, :seq_len]

        # Mix the causal max and the bidirectional mask to get the full
        # allowable attention (i.e. full = not accounting for padding yet)
        causal = torch.tril(
            torch.ones((seq_len, seq_len),
                       dtype=torch.bool,
                       device=prefix_mask.device)).view(1, 1, seq_len, seq_len)
        prefix = prefix_mask.view(-1, 1, 1, seq_len)
        cannot_attend = ~torch.logical_or(causal, prefix.bool())

        min_val = torch.finfo(attn_bias.dtype).min
        attn_bias = attn_bias.masked_fill(cannot_attend, min_val)

        return attn_bias

    def _apply_sequence_id(self, attn_bias: torch.Tensor,
                           sequence_id: torch.LongTensor) -> torch.Tensor:
        seq_len = sequence_id.shape[-1]
        if seq_len > self.config.max_seq_len:
            raise ValueError(
                f'sequence_id sequence length cannot exceed max_seq_len={self.config.max_seq_len}'
            )

        # select seq_len subset of attn mask
        attn_bias = attn_bias[..., :seq_len, :seq_len]

        # Restrict attention to tokens that share the same value
        # in sequence_id
        cannot_attend = torch.logical_not(
            torch.eq(
                sequence_id.view(-1, seq_len, 1),
                sequence_id.view(-1, 1, seq_len),
            )).unsqueeze(1)
        min_val = torch.finfo(attn_bias.dtype).min
        attn_bias = attn_bias.masked_fill(cannot_attend, min_val)

        return attn_bias

    def forward(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[List[Tuple[torch.FloatTensor]]] = None,
        attention_mask: Optional[torch.ByteTensor] = None,
        prefix_mask: Optional[torch.ByteTensor] = None,
        sequence_id: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        use_cache: Optional[bool] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> BaseModelOutputWithPast:
        return_dict = (return_dict
                       if return_dict is not None else self.config.return_dict)
        use_cache = (use_cache
                     if use_cache is not None else self.config.use_cache)

        if attention_mask is not None:
            attention_mask = attention_mask.bool()  # type: ignore

        if prefix_mask is not None:
            prefix_mask = prefix_mask.bool()  # type: ignore

        # These args are passed in by keyword in huggingface's generate function
        # https://github.com/huggingface/transformers/blob/68287689f2f0d8b7063c400230b3766987abf18d/src/transformers/generation/utils.py#L2201-L2206
        # but have not yet been fully implemented in MPTModel
        if not return_dict:
            raise NotImplementedError(
                'return_dict False is not implemented yet for MPT')
        if output_attentions:
            if self.attn_impl != 'torch':
                raise NotImplementedError(
                    'output_attentions is not implemented for MPT when using attn_impl `flash` or `triton`.'
                )

        if (self.training and attention_mask is not None and
                attention_mask[:, 0].sum() != attention_mask.shape[0]):
            raise NotImplementedError(
                'MPT does not support training with left padding.')

        if self.prefix_lm and prefix_mask is None:
            raise ValueError(
                'prefix_mask is a required argument when MPT is configured with prefix_lm=True.'
            )

        # Raise a not implemented error if input_embeds is not None (this is an arg in huggingface transformers and we need to support it for PEFT)
        if inputs_embeds is not None:
            raise NotImplementedError(
                'inputs_embeds is not implemented for MPT.')

        if self.training:
            if self.attn_uses_sequence_id and sequence_id is None:
                raise ValueError(
                    'sequence_id is a required argument when MPT is configured with attn_uses_sequence_id=True '
                    + 'and the model is in train mode.')
            elif (self.attn_uses_sequence_id is False) and (sequence_id
                                                            is not None):
                warnings.warn(
                    'MPT received non-None input for `sequence_id` but is configured with attn_uses_sequence_id=False. '
                    +
                    'This input will be ignored. If you want the model to use `sequence_id`, set attn_uses_sequence_id to True.'
                )

        S = input_ids.size(1)

        assert (
            S <= self.config.max_seq_len
        ), f'Cannot forward input with seq_len={S}, this model only supports seq_len<={self.config.max_seq_len}'

        tok_emb = self.wte(input_ids)
        if self.learned_pos_emb:
            past_position = 0
            if past_key_values is not None:
                if len(past_key_values) != self.config.n_layers:
                    raise ValueError(
                        f'past_key_values must provide a past_key_value for each attention '
                        +
                        f'layer in the network ({len(past_key_values)=}; {self.config.n_layers=}).'
                    )
                # For attn_impl: triton and flash the past key tensor spec is (batch, seq, dim).
                # For attn_impl: torch the past key tensor spec is (batch, heads, head_dim, seq).
                # Here we shift position embedding using the `seq` dim of the past key
                past_position = past_key_values[0][0].size(1)
                if self.attn_impl == 'torch':
                    past_position = past_key_values[0][0].size(3)

            if S + past_position > self.config.max_seq_len:
                raise ValueError(
                    f'Cannot forward input with past sequence length {past_position} and current sequence length '
                    +
                    f'{S + 1}, this model only supports total sequence length <= {self.config.max_seq_len}.'
                )
            pos = torch.arange(
                past_position,
                S + past_position,
                dtype=torch.long,
                device=input_ids.device,
            ).unsqueeze(0)
            if attention_mask is not None:
                # adjust the position indices to account for padding tokens
                pos = torch.clamp(
                    pos - torch.cumsum((~attention_mask).to(torch.int32),
                                       dim=1)[:, past_position:],
                    min=0,
                )

            pos_emb = self.wpe(pos)
            x = tok_emb + pos_emb
        else:
            # ALiBi and NoPE use this path (RoPE will also use this path if / when enabled)
            x = tok_emb

        if self.embedding_fraction == 1:
            x = self.emb_drop(x)
        else:
            # this implementation is proposed on page 7 of the GLM-130B paper https://arxiv.org/abs/2210.02414
            x_shrunk = (x * self.embedding_fraction) + (
                x.detach() * (1 - self.embedding_fraction))
            assert isinstance(self.emb_drop, nn.Module)  # pyright
            x = self.emb_drop(x_shrunk)

        attn_bias, attention_mask = self._attn_bias(
            device=x.device,
            dtype=torch.float32,
            attention_mask=attention_mask,
            prefix_mask=prefix_mask,
            sequence_id=sequence_id,
        )

        # initialize the past key values cache if it should be used
        presents = () if use_cache else None
        if use_cache and past_key_values is None:
            past_key_values = [() for _ in range(self.config.n_layers)
                              ]  # type: ignore

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        for b_idx, block in enumerate(self.blocks):
            if output_hidden_states:
                assert all_hidden_states is not None  # pyright
                all_hidden_states = all_hidden_states + (x,)
            past_key_value = (past_key_values[b_idx]
                              if past_key_values is not None else None)
            x, attn_weights, present = block(
                x,
                past_key_value=past_key_value,
                attn_bias=attn_bias,
                attention_mask=attention_mask,
                is_causal=self.is_causal,
                output_attentions=bool(output_attentions),
            )
            if presents is not None:
                presents += (present,)

            if output_attentions:
                assert all_self_attns is not None  # pyright
                all_self_attns = all_self_attns + (attn_weights,)

        x = self.norm_f(x)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            assert all_hidden_states is not None  # pyright
            all_hidden_states = all_hidden_states + (x,)

        return BaseModelOutputWithPast(
            last_hidden_state=x,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    # Param Initialization, needed for device='meta' fast initialization
    def param_init_fn(self, module: nn.Module) -> None:
        init_fn_name = self.config.init_config['name']
        MODEL_INIT_REGISTRY[init_fn_name](
            module=module,
            n_layers=self.config.n_layers,
            d_model=self.config.d_model,
            **self.config.init_config,
        )

    # FSDP Wrap function
    def fsdp_wrap_fn(self, module: nn.Module) -> bool:
        return isinstance(module, MPTBlock)

    # Activation Checkpointing
    def activation_checkpointing_fn(self, module: nn.Module) -> bool:
        return isinstance(module, MPTBlock)


# JP: If we are lucky, we don't actually have to implement this class
# at all. And we can just use the logits of the final layer
class MPTForContrastiveLM(MPTPreTrainedModel):
    """
    The following class applies a contrastive loss
    to the standard MPT architecture
    """

    # JP
    def __init__(self, config: MPTConfig):
        super().__init__(config)
        # if not config.tie_word_embeddings: #JP unclear whether this is necessary
        #     raise ValueError(
        #         'MPTForContrastiveLM only supports tied word embeddings') # JP: remove


        # JP confirm that this is not a decoder
        if config.is_decoder:
            warnings.warn(
                'If you want to use `BertForMaskedLM` make sure `config.is_decoder=False` for '
                + 'bi-directional self-attention.')

        log.info(f'Instantiating an MPTForCausalLM model from {__file__}')

        self.transformer: MPTModel = MPTModel(config)

        for child in self.transformer.children():
            if isinstance(child, torch.nn.ModuleList):
                continue
            if isinstance(child, torch.nn.Module):
                child._fsdp_wrap = True

        # JP: for BERT, we usually init which ontains a linear classification layer
        # self.cls = BertOnlyMLMHead(config,
        #                           self.bert.embeddings.word_embeddings.weight)

        # enables scaling output logits; similar to a softmax "temperature"
        # PaLM paper uses scale 1/sqrt(config.d_model)
        self.logit_scale = None
        if config.logit_scale is not None:
            logit_scale = config.logit_scale
            if isinstance(logit_scale, str):
                if logit_scale == 'inv_sqrt_d_model':
                    logit_scale = 1 / math.sqrt(config.d_model)
                else:
                    raise ValueError(
                        f"{logit_scale=} is not recognized as an option; use numeric value or 'inv_sqrt_d_model'."
                    )
            self.logit_scale = logit_scale

    def get_input_embeddings(self) -> nn.Embedding:
        return self.transformer.wte

    def set_input_embeddings(
            self, value: Union[SharedEmbedding, nn.Embedding]) -> None:
        self.transformer.wte = value

    # JP: Note that weight tying means the input embedding layer weights are the same
    # as the output embedding weights, following https://paperswithcode.com/method/weight-tying
    # There is no reason to do this for our contrastive objective...
    #
    def get_output_embeddings(self) -> nn.Embedding:
        return self.transformer.wte

    def set_output_embeddings(
            self, new_embeddings: Union[SharedEmbedding, nn.Embedding]) -> None:
        self.transformer.wte = new_embeddings

    def set_decoder(self, decoder: MPTModel) -> None:
        self.transformer = decoder

    def get_decoder(self) -> MPTModel:
        return self.transformer
    
    # JP: The output should be formatted as BaseModelOutputWithPooling with last_hidden_state,
    # pooler_output, hidden_states and attentions. CausalLMOutputWithPast below also includes
    # past_key_values, which can help speed up sequential decoding
    def forward(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[List[Tuple[torch.FloatTensor]]] = None,
        attention_mask: Optional[torch.ByteTensor] = None,
        prefix_mask: Optional[torch.ByteTensor] = None,
        sequence_id: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        use_cache: Optional[bool] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> CausalLMOutputWithPast:
        return_dict = (return_dict
                       if return_dict is not None else self.config.return_dict)
        use_cache = (use_cache
                     if use_cache is not None else self.config.use_cache)

        # if input_embeds is not none, raise a not implemented error
        if inputs_embeds is not None:
            raise NotImplementedError(
                'inputs_embeds has to be None (for hf/peft support).')
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.transformer(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            prefix_mask=prefix_mask,
            sequence_id=sequence_id,
            return_dict=return_dict,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
        )

        # move outputs to same device as weights for token embedding
        # needed to support HF `device_map`
        # JP: we'll keep this weight tying and assume it works in the bidirectional case
        logits = self.transformer.wte(
            input = outputs.last_hidden_state.to(self.transformer.wte.weight.device),
            unembed = True,
        )

        if self.logit_scale is not None:
            if self.logit_scale == 0:
                warnings.warn(
                    f'Multiplying logits by {self.logit_scale=}. This will produce uniform (uninformative) outputs.'
                )
            logits *= self.logit_scale

        loss = None

        # JP: keep loss None for now, and define the loss in the Composer model
        # if labels is not None:
        #     _labels = torch.roll(labels, shifts=-1)
        #     _labels[:, -1] = -100
        #     loss = F.cross_entropy(
        #         logits.view(-1, logits.size(-1)),
        #         _labels.to(logits.device).view(-1),
        #     )


        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values, # likely not used, but will keep just in case
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class MPTForCausalLM(MPTPreTrainedModel):

    def __init__(self, config: MPTConfig):
        super().__init__(config)
        if not config.tie_word_embeddings:
            raise ValueError(
                'MPTForCausalLM only supports tied word embeddings')

        log.info(f'Instantiating an MPTForCausalLM model from {__file__}')

        self.transformer: MPTModel = MPTModel(config)

        for child in self.transformer.children():
            if isinstance(child, torch.nn.ModuleList):
                continue
            if isinstance(child, torch.nn.Module):
                child._fsdp_wrap = True

        # enables scaling output logits; similar to a softmax "temperature"
        # PaLM paper uses scale 1/sqrt(config.d_model)
        self.logit_scale = None
        if config.logit_scale is not None:
            logit_scale = config.logit_scale
            if isinstance(logit_scale, str):
                if logit_scale == 'inv_sqrt_d_model':
                    logit_scale = 1 / math.sqrt(config.d_model)
                else:
                    raise ValueError(
                        f"{logit_scale=} is not recognized as an option; use numeric value or 'inv_sqrt_d_model'."
                    )
            self.logit_scale = logit_scale

    def get_input_embeddings(self) -> nn.Embedding:
        return self.transformer.wte

    def set_input_embeddings(
            self, value: Union[SharedEmbedding, nn.Embedding]) -> None:
        self.transformer.wte = value

    def get_output_embeddings(self) -> nn.Embedding:
        return self.transformer.wte

    def set_output_embeddings(
            self, new_embeddings: Union[SharedEmbedding, nn.Embedding]) -> None:
        self.transformer.wte = new_embeddings

    def set_decoder(self, decoder: MPTModel) -> None:
        self.transformer = decoder

    def get_decoder(self) -> MPTModel:
        return self.transformer

    def forward(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[List[Tuple[torch.FloatTensor]]] = None,
        attention_mask: Optional[torch.ByteTensor] = None,
        prefix_mask: Optional[torch.ByteTensor] = None,
        sequence_id: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        use_cache: Optional[bool] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> CausalLMOutputWithPast:
        return_dict = (return_dict
                       if return_dict is not None else self.config.return_dict)
        use_cache = (use_cache
                     if use_cache is not None else self.config.use_cache)

        # if input_embeds is not none, raise a not implemented error
        if inputs_embeds is not None:
            raise NotImplementedError(
                'inputs_embeds has to be None (for hf/peft support).')
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.transformer(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            prefix_mask=prefix_mask,
            sequence_id=sequence_id,
            return_dict=return_dict,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
        )

        # move outputs to same device as weights for token embedding
        # needed to support HF `device_map`
        logits = self.transformer.wte(
            outputs.last_hidden_state.to(self.transformer.wte.weight.device),
            True,
        )

        if self.logit_scale is not None:
            if self.logit_scale == 0:
                warnings.warn(
                    f'Multiplying logits by {self.logit_scale=}. This will produce uniform (uninformative) outputs.'
                )
            logits *= self.logit_scale

        loss = None
        if labels is not None:
            _labels = torch.roll(labels, shifts=-1)
            _labels[:, -1] = -100
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                _labels.to(logits.device).view(-1),
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    # Param Initialization, needed for device='meta' fast initialization
    def param_init_fn(self, module: nn.Module) -> None:
        init_fn_name = self.config.init_config['name']
        MODEL_INIT_REGISTRY[init_fn_name](
            module=module,
            n_layers=self.config.n_layers,
            d_model=self.config.d_model,
            **self.config.init_config,
        )

    # FSDP Wrap function
    def fsdp_wrap_fn(self, module: nn.Module) -> bool:
        return isinstance(module, MPTBlock)

    # Activation Checkpointing
    def activation_checkpointing_fn(self, module: nn.Module) -> bool:
        return isinstance(module, MPTBlock)

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[List[Tuple[torch.Tensor,
                                             torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        if inputs_embeds is not None:
            raise NotImplementedError(
                'inputs_embeds is not implemented for MPT yet')

        attention_mask = kwargs['attention_mask'].bool()
        if attention_mask[:, -1].sum() != attention_mask.shape[0]:
            raise NotImplementedError(
                'MPT does not support generation with right padding.')

        if self.transformer.attn_uses_sequence_id and self.training:
            sequence_id = torch.zeros_like(input_ids[:1])
        else:
            sequence_id = None

        if past_key_values is not None:
            input_ids = input_ids[:, -1].unsqueeze(-1)

        if self.transformer.prefix_lm:
            # Leverage a convenience of sequential generation!
            prefix_mask = torch.ones_like(attention_mask)
            # This requires that we're using the cache
            if kwargs.get('use_cache') == False:
                raise NotImplementedError(
                    'MPT with prefix_lm=True does not support use_cache=False.')
        else:
            prefix_mask = None

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'prefix_mask': prefix_mask,
            'sequence_id': sequence_id,
            'past_key_values': past_key_values,
            'use_cache': kwargs.get('use_cache', True),
        }

    @staticmethod
    def _reorder_cache(
            past_key_values: List[Tuple[torch.Tensor, torch.Tensor]],
            beam_idx: torch.LongTensor) -> List[Tuple[torch.Tensor, ...]]:
        """Used by HuggingFace generate when using beam search with kv-caching.

        See https://github.com/huggingface/transformers/blob/3ec7a47664ebe40c40f4b722f6bb1cd30c3821ec/src/transformers/models/gpt2/modeling_gpt2.py#L1122-L1133
        for an example in transformers.
        """
        reordered_past = []
        for layer_past in past_key_values:
            reordered_past += [
                tuple(
                    past_state.index_select(0, beam_idx)
                    for past_state in layer_past)
            ]
        return reordered_past


class ComposerMPTCausalLM(HuggingFaceModel):

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
        train_metrics = [LanguageCrossEntropy(),
                         LanguagePerplexity()] if use_train_metrics else []
        eval_metrics = [
            LanguageCrossEntropy(),
            LanguagePerplexity(),
            InContextLearningLMAccuracy(),
            InContextLearningMultipleChoiceAccuracy(),
            InContextLearningQAAccuracy(),
            InContextLearningCodeEvalAccuracy(),
            InContextLearningLMExpectedCalibrationError(),
            InContextLearningMCExpectedCalibrationError(),
        ]

        super().__init__(
            model=model,
            tokenizer=tokenizer,
            use_logits=True,
            metrics=train_metrics,
            eval_metrics=eval_metrics,
            shift_labels=True,
            allow_embedding_resizing=True,
        )

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

    def get_targets(self, batch: Mapping) -> torch.Tensor:
        targets = torch.roll(batch['labels'], shifts=-1)
        targets[:, -1] = -100
        return targets

    def forward(self, batch: MutableMapping) -> CausalLMOutputWithPast:
        if self.model.transformer.prefix_lm:
            add_bidirectional_mask_if_missing(batch)
        # Note: prefix_mask is only used if model.prefix_lm is True
        return self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch.get('attention_mask', None),
            prefix_mask=batch.get('bidirectional_mask', None),
            sequence_id=batch.get('sequence_id', None),
            inputs_embeds=batch.get('inputs_embeds', None),
        )

    def loss(self, outputs: CausalLMOutputWithPast,
             batch: Mapping) -> torch.Tensor:
        targets = self.get_targets(batch)
        return self.loss_fn(outputs.logits.view(-1, outputs.logits.size(-1)),
                            targets.view(-1))

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


class ComposerMPTContrastiveLM(HuggingFaceModel):

    """ 
    JP: The following code implements a contrastive loss function using the MPT
    architecture.

    Note that the MPTForCausalLM architecture can be used as is, provided that `labels=None`
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
        train_metrics = [LanguageCrossEntropy(),
                         LanguagePerplexity()] if use_train_metrics else []
        # JP: These metrics should work
        eval_metrics = [
            LanguageCrossEntropy(),
            LanguagePerplexity(),
            InContextLearningLMAccuracy(),
            InContextLearningMultipleChoiceAccuracy(),
            InContextLearningQAAccuracy(),
            InContextLearningCodeEvalAccuracy(),
            InContextLearningLMExpectedCalibrationError(),
            InContextLearningMCExpectedCalibrationError(),
        ]

        super().__init__(
            model=model,
            tokenizer=tokenizer,
            use_logits=False, # JP: set to False
            metrics=train_metrics,
            eval_metrics=eval_metrics,
            shift_labels=False, # JP: set to False
            allow_embedding_resizing=True, # JP: Not sure what this does
        )

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

    def format_queries_batch(self, batch):
        """ Format `queries` by selecting every other pair from the batch """
        queries = {}
        for key in batch.keys():
            queries[key] = batch[key][:,0::2,:].reshape(batch[key].size(0), -1)
        
        return queries
    
    def format_passages_batch(self, batch):
        """ Format `passages` by selecting every other pair from the batch """
        passages = {}
        for key in batch.keys():
            passages[key] = batch[key][:,1::2,:].reshape(batch[key].size(0), -1)
        
        return passages
    
    # JP: Update
    def get_targets(self, batch: Mapping) -> torch.Tensor:
        #targets = torch.roll(batch['labels'], shifts=-1)
        #targets[:, -1] = -100
        #return targets
        passages = self.format_passages_batch(batch)
        targets = torch.arange(0, passages.shape[0], dtype=torch.long, device=passages.device)
        return targets


    def forward(self, batch: MutableMapping) -> CausalLMOutputWithPast:
        if self.model.transformer.prefix_lm:
            add_bidirectional_mask_if_missing(batch)
        # Note: prefix_mask is only used if model.prefix_lm is True
        return self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch.get('attention_mask', None),
            prefix_mask=batch.get('bidirectional_mask', None),
            sequence_id=batch.get('sequence_id', None),
            inputs_embeds=batch.get('inputs_embeds', None),
        )
    
    def _compute_scores(self, batch) -> Tuple:

        # Run Pairs through the encoder separately in two passes, designated as q (query) and p (passage)
        # [batch_size, sequence_length]
        #
        # the pooled_outputs is [batch_size, hidden_size]
        #
        # Note: at some future point we could use the flag 'token_type_ids' which was used in the original
        # BERT formula to keep track of sentences A and sentences B in the next sentence prediction objective
        # function. For now we split even and odd rows
        queries_batch = self.format_queries_batch(batch)
        passages_batch = self.format_passages_batch(batch)
        
        
        # print(self.tokenizer.decode(queries_batch['input_ids'][0]))
        # print(self.tokenizer.decode(passages_batch['input_ids'][0]))

        q_encoder_outputs = self.model(
                                        input_ids=queries_batch['input_ids'],
                                        token_type_ids=queries_batch.get('token_type_ids', None),
                                        attention_mask=queries_batch.get('attention_mask', None),
                                        position_ids=queries_batch.get('position_ids', None),
                                        masked_tokens_mask=queries_batch.get('masked_tokens_mask', None),
                                    )

        p_encoder_outputs = self.model(
                                        input_ids=passages_batch['input_ids'],
                                        token_type_ids=passages_batch.get('token_type_ids', None),
                                        attention_mask=passages_batch.get('attention_mask', None),
                                        position_ids=passages_batch.get('position_ids', None),
                                        masked_tokens_mask=passages_batch.get('masked_tokens_mask', None),
                                    )

        # JP: Note we went from p_encoder_outputs.masked_fill to q_encoder_outputs.hidden_states.masked_fill
        # Might need to do hidden_states[-1] to only select activations from the last layer
        q_last_hidden = q_encoder_outputs.hidden_states[-1].masked_fill(~queries_batch.get('attention_mask', None)[..., None].bool(), 0.0)
        q_pooled_outputs = q_last_hidden.sum(dim=1) / queries_batch.get('attention_mask', None).sum(dim=1)[..., None]
        
        p_last_hidden = p_encoder_outputs.hidden_states[-1].masked_fill(~passages_batch.get('attention_mask', None)[..., None].bool(), 0.0)
        p_pooled_outputs = p_last_hidden.sum(dim=1) / passages_batch.get('attention_mask', None).sum(dim=1)[..., None]
        
        #print('>>p_pooled_outputs shape:',p_pooled_outputs.shape)
        
        q_pooled_outputs = F.normalize(q_pooled_outputs, dim=-1) # Todo: should be configurable when L2 normalizing
        p_pooled_outputs = F.normalize(p_pooled_outputs, dim=-1)

        q_pooled_outputs = q_pooled_outputs.contiguous() # Why do we need to make this contiguous?
        p_pooled_outputs = p_pooled_outputs.contiguous() # Why do we need to make this contiguous?

        all_q_pooled_outputs = dist_gather_tensor(q_pooled_outputs)
        all_p_pooled_outputs = dist_gather_tensor(p_pooled_outputs)
        # all_q_pooled_outputs = q_pooled_outputs
        # all_p_pooled_outputs = p_pooled_outputs
        
        all_scores, all_labels = self.full_contrastive_scores_and_labels(queries=all_q_pooled_outputs, 
                                                                         passages=all_p_pooled_outputs)
        
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

        print('>> mean scores',all_scores.mean())
        print('>> mean diagonal scores', all_scores.diagonal().mean())
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
        
        scores, labels = self._compute_scores(batch)
                
        loss = self.loss_fn(scores, labels)
        
        # Based on https://github.com/microsoft/unilm/blob/b60c741f746877293bb85eed6806736fc8fa0ffd/simlm/src/models/biencoder_model.py#L60C62-L60C62
        # We are scaling the loss by the world size because we think it will be divided by the world size in the backward pass
        # This is a hacky way of getting around implementing our own backward pass
        # loss *= dist.get_world_size()
        
        return loss
    
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
