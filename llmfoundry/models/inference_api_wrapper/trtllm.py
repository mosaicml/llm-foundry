# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Implements a TRT-LLM evaluation model wrapped around a
:class:`.ComposerModel`."""

import json
from pathlib import Path
from typing import Any, Optional

import tensorrt_llm
import torch
from omegaconf import DictConfig
from tensorrt_llm.runtime import ModelConfig, SamplingConfig
from transformers import PreTrainedTokenizer

from llmfoundry.models.inference_api_wrapper.interface import \
    InferenceAPIEvalWrapper

__all__ = ['TRTLLMEvalWrapper']


# From tensorrt_llm/examples/{model_name}/build.py
def get_engine_name(model: str, dtype: str, tp_size: int, rank: int):
    return '{}_{}_tp{}_rank{}.engine'.format(model, dtype, tp_size, rank)


class TRTLLMEvalWrapper(InferenceAPIEvalWrapper):

    def __init__(
        self,
        model_cfg: DictConfig,
        tokenizer: PreTrainedTokenizer,
    ):

        super().__init__(model_cfg, tokenizer)

        tensorrt_llm.logger.set_level(model_cfg['log_level'])

        # Load TRT config from file
        engine_dir = Path(model_cfg['engine_dir'])
        config_path = engine_dir / 'config.json'
        with open(config_path, 'r') as f:
            config = json.load(f)

        # Set vars from config
        use_gpt_attention_plugin = config['plugin_config'][
            'gpt_attention_plugin']
        inflight_batching_gpt_attention_plugin = config['plugin_config'][
            'inflight_batching_gpt_attention_plugin']
        remove_input_padding = config['plugin_config']['remove_input_padding']
        if remove_input_padding:
            raise ValueError(
                'TRT-LLM Evaluation Wrapper does not support remove_input_padding.'
            )
        dtype = config['builder_config']['precision']
        world_size = config['builder_config']['tensor_parallel']
        assert world_size == tensorrt_llm.mpi_world_size(), \
            f'Engine world size ({world_size}) != Runtime world size ({tensorrt_llm.mpi_world_size()})'
        num_heads = config['builder_config']['num_heads'] // world_size
        hidden_size = config['builder_config']['hidden_size'] // world_size
        vocab_size = config['builder_config']['vocab_size']
        num_layers = config['builder_config']['num_layers']
        multi_query_mode = config['builder_config']['multi_query_mode']
        paged_kv_cache = config['builder_config'].get('paged_kv_cache', False)
        tokens_per_block = config['builder_config'].get('tokens_per_block', 64)
        use_prompt_tuning = config['builder_config'].get(
            'use_prompt_tuning', False)

        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        # Device and rank
        runtime_rank = tensorrt_llm.mpi_rank()
        runtime_mapping = tensorrt_llm.Mapping(world_size, runtime_rank)
        torch.cuda.set_device(runtime_rank % runtime_mapping.gpus_per_node)

        # Tokenization and sampling
        self.END_ID = model_cfg.get('eos_token_id', self.tokenizer.eos_token_id)
        self.PAD_ID = model_cfg.get('pad_token_id', self.tokenizer.pad_token_id)
        if self.PAD_ID == None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        print('EOS TOKEN:', self.END_ID)
        print('Pad token:', self.PAD_ID)

        self.sampling_config = SamplingConfig(end_id=self.END_ID,
                                              pad_id=self.PAD_ID,
                                              num_beams=1)

        # Load TRT engine
        engine_name = get_engine_name(model_cfg['version'], dtype, world_size,
                                      runtime_rank)
        serialize_path = engine_dir / engine_name
        with open(serialize_path, 'rb') as f:
            engine_buffer = f.read()

        # Initialize generation session for model
        trt_model_config = ModelConfig(
            num_heads=num_heads,
            hidden_size=self.hidden_size,
            vocab_size=self.vocab_size,
            num_layers=num_layers,
            gpt_attention_plugin=use_gpt_attention_plugin,
            inflight_batching_gpt_attention_plugin=
            inflight_batching_gpt_attention_plugin,
            multi_query_mode=multi_query_mode,
            remove_input_padding=remove_input_padding,
            paged_kv_cache=paged_kv_cache,
            tokens_per_block=tokens_per_block,
            use_prompt_tuning=use_prompt_tuning)
        self.decoder = tensorrt_llm.runtime.GenerationSession(
            trt_model_config, engine_buffer, runtime_mapping)

    def eval_forward(self, batch, outputs: Optional[Any] = None):
        # If the batch mode is generate, we will generate a requested number of tokens using the underlying
        # model's generate function. Strings will be returned from eval_forward
        output_logits_batch = []
        batch = self.rebatch(batch)
        for tokens, cont_idxs in zip(batch['input_ids'],
                                     batch['continuation_indices']):

            seqlen = tokens.shape[0]
            tokens = tokens.tolist()
            cont_idxs = cont_idxs.tolist()
            expected_cont_tokens = tokens[cont_idxs[0]:cont_idxs[-1] + 1]

            prompt = tokens[:cont_idxs[0]]
            input_ids = torch.tensor([prompt], dtype=torch.int, device='cuda')
            input_lengths = torch.tensor([input_ids.size(1)], dtype=torch.int, device='cuda')
            #print("prompt:", self.tokenizer.decode(prompt))
            #print("Input ids data:", input_ids, len(input_ids), input_ids[0].shape)
            #print("Input lengths:", input_lengths)
            #print(cont_idxs[0])
            #print("Expected continuation tokens:", len(expected_cont_tokens))
            self.decoder.setup(input_lengths.size(0),
                               torch.max(input_lengths).item(),
                               len(expected_cont_tokens))

            output_idsg, output_logits_list = self.decoder.decode(
                input_ids, input_lengths, self.sampling_config)

            #print("Decoded output:", self.tokenizer.decode(output_ids[0][0][cont_idxs[0]:].tolist()))

            output_logits = torch.nn.functional.one_hot(
                torch.tensor(tokens[1:cont_idxs[0]], device='cuda'),
                num_classes=self.vocab_size)

            for i in range(len(output_logits_list)):
                output_logits_list[i] = output_logits_list[i].squeeze()

            next_logit_tensor = torch.stack(output_logits_list)
            output_logits = torch.cat([output_logits, next_logit_tensor])
            #print(output_logits.shape)
            #print(output_ids[0][0][cont_idxs[0]:].tolist())
            padding = torch.nn.functional.one_hot(torch.full(
                (seqlen - output_logits.shape[0],),
                self.PAD_ID,
                device=output_logits.device),
                                                  num_classes=self.vocab_size)
            output_logits = torch.cat([output_logits, padding])
            #print("Output logits shape:", output_logits.shape)
            output_logits_batch.append(output_logits)

        return torch.stack(output_logits_batch).to(batch['input_ids'].device)
