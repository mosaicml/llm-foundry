# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Implements a TRT-LLM evaluation model wrapped around a
:class:`.ComposerModel`."""

import json
from pathlib import Path
from typing import Any, Optional

import torch
from omegaconf import DictConfig
from transformers import PreTrainedTokenizer

from llmfoundry.models.inference_api_wrapper.interface import InferenceAPIEvalWrapper

__all__ = ['TRTLLMEvalWrapper']

try:
    import tensorrt_llm
    from tensorrt_llm.runtime import ModelConfig, SamplingConfig
    from tensorrt_llm.quantization import QuantMode
    TRT_LLM_INSTALLED = True
except ImportError:
    TRT_LLM_INSTALLED = False


def check_if_trt_llm_installed():
    if not TRT_LLM_INSTALLED:
        raise ImportError(
            'TRT-LLM is not installed. It must be installed to use the TRTLLMEValWrapper.'
        )


# From tensorrt_llm/examples/{model_name}/build.py
def get_engine_name(model: str, dtype: str, tp_size: int, pp_size: int, rank: int):
    if pp_size == 1:
        return '{}_{}_tp{}_rank{}.engine'.format(model, dtype, tp_size, rank)
    return '{}_{}_tp{}_pp{}_rank{}.engine'.format(model, dtype, tp_size,
                                                  pp_size, rank)


class TRTLLMEvalWrapper(InferenceAPIEvalWrapper):

    def __init__(
        self,
        model_cfg: DictConfig,
        tokenizer: PreTrainedTokenizer,
    ):
        check_if_trt_llm_installed()

        super().__init__(model_cfg, tokenizer)

        tensorrt_llm.logger.set_level(model_cfg['log_level'])

        # Load TRT config from file
        engine_dir = Path(model_cfg['engine_dir'])
        config_path = engine_dir / 'config.json'
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        dtype = config['builder_config']['precision']
        tp_size = config['builder_config']['tensor_parallel']
        pp_size = config['builder_config'].get('pipeline_parallel', 1)
        world_size = tp_size * pp_size

        assert world_size == tensorrt_llm.mpi_world_size(), \
            f'Engine world size ({world_size}) != Runtime world size ({tensorrt_llm.mpi_world_size()})'
        
        num_heads = config['builder_config']['num_heads'] // tp_size
        hidden_size = config['builder_config']['hidden_size'] // tp_size
        vocab_size = config['builder_config']['vocab_size']
        num_layers = config['builder_config']['num_layers']
        use_gpt_attention_plugin = bool(
            config['plugin_config']['gpt_attention_plugin'])
        remove_input_padding = config['plugin_config']['remove_input_padding']
        #if remove_input_padding:
        #    raise ValueError(
        #        'TRT-LLM Evaluation Wrapper does not support remove_input_padding.'
        #    )
       
        num_kv_heads = config['builder_config'].get('num_kv_heads', num_heads)
        paged_kv_cache = config['plugin_config']['paged_kv_cache']
        tokens_per_block = config['plugin_config']['tokens_per_block']
        use_custom_all_reduce = config['plugin_config'].get('use_custom_all_reduce',
                                                            False)

        quant_mode = QuantMode(config['builder_config']['quant_mode'])
        if config['builder_config'].get('multi_query_mode', False):
            tensorrt_llm.logger.warning(
                "`multi_query_mode` config is deprecated. Please rebuild the engine."
            )
            num_kv_heads = 1
        num_kv_heads = (num_kv_heads + tp_size - 1) // tp_size

        model_config = tensorrt_llm.runtime.ModelConfig(
            vocab_size=vocab_size,
            num_layers=num_layers,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            hidden_size=hidden_size,
            paged_kv_cache=paged_kv_cache,
            tokens_per_block=tokens_per_block,
            gpt_attention_plugin=use_gpt_attention_plugin,
            remove_input_padding=remove_input_padding,
            use_custom_all_reduce=use_custom_all_reduce,
            dtype=dtype,
            quant_mode=quant_mode,
            gather_all_token_logits=True)


        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        # Device and rank
        runtime_rank = tensorrt_llm.mpi_rank()
        runtime_mapping = tensorrt_llm.Mapping(world_size,
                                           runtime_rank,
                                           tp_size=tp_size,
                                           pp_size=pp_size)
        self.device_num = runtime_rank % runtime_mapping.gpus_per_node
        self.device = torch.device('cuda:' + str(self.device_num))
        torch.cuda.set_device(self.device_num)

        # Tokenization and sampling
        self.END_ID = model_cfg.get('eos_token_id', self.tokenizer.eos_token_id)
        self.PAD_ID = model_cfg.get('pad_token_id', self.tokenizer.pad_token_id)
        if self.PAD_ID == None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.PAD_ID = self.tokenizer.eos_token_id
          
        print('EOS TOKEN:', self.END_ID)
        print('Pad token:', self.PAD_ID)

        self.sampling_config = SamplingConfig(end_id=self.END_ID,
                                              pad_id=self.PAD_ID,
                                              num_beams=1,
                                              return_dict=True)

        # Load TRT engine
        engine_name = get_engine_name(model_cfg['version'], dtype, tp_size, pp_size,
                                      runtime_rank)
        serialize_path = engine_dir / engine_name
        with open(serialize_path, 'rb') as f:
            engine_buffer = f.read()

        self.decoder = tensorrt_llm.runtime.GenerationSession(
            model_config, engine_buffer, runtime_mapping, debug_mode=False)

        print("!!! Initialized generation session for rank:", runtime_rank)        
        torch.cuda.synchronize()
        
        # Move metrics to proper device (doesn't help, have to do this in update_metric())
        # for key, value in self.eval_metrics.items():
        #    self.eval_metrics[key] = value.to(device=self.device)
        #    print("Eval metric now at:", self.eval_metrics[key].device) 

    def rebatch(self, batch):
        """
        Move tensors in batch to the correct GPU.
        """
        if isinstance(batch, dict):
            for key, value in batch.items():
                batch[key] = self.rebatch(value)
            return batch
        elif isinstance(batch, torch.Tensor):
            return batch.to(device=self.device)
        elif isinstance(batch, list):
            return [self.rebatch(b) for b in batch]
        
        return batch


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
            
            
            input_ids = torch.tensor([prompt], dtype=torch.int, device=self.device)
            input_lengths = torch.tensor([input_ids.size(1)],
                                         dtype=torch.int,
                                         device=self.device)
            # print("prompt:", self.tokenizer.decode(prompt))
            # print("Input device:", input_ids.get_device())
            #print("Input ids data:", input_ids, len(input_ids), input_ids[0].shape)
            #print("Input lengths:", input_lengths)
            #print(cont_idxs[0])
            #print("Expected continuation tokens:", len(expected_cont_tokens))
            with torch.no_grad():
                self.decoder.setup(input_lengths.size(0),
                               torch.max(input_lengths).item(),
                               len(expected_cont_tokens))

                output_dict = self.decoder.decode(
                    input_ids, input_lengths, self.sampling_config, return_dict=True)
            
            torch.cuda.synchronize()

            context_logits = output_dict['context_logits']
            context_logits = context_logits.squeeze()
            output_logits_list = output_dict['generation_logits']
            # print("Output ids:", output_dict['output_ids'][0][0][cont_idxs[0]:].tolist())
            for i in range(len(output_logits_list)):
                output_logits_list[i] = output_logits_list[i].squeeze()
                # print("Output ids:", self.tokenizer.decode(output_dict['output_ids'][0][0].tolist()))
            # print("Context logits:", context_logits.shape)
            # print("Output logits list:", output_logits_list)
            if len(output_logits_list) > 0:
                # print("Output logits 0 shape:", output_logits_list[0].shape)
                output_logits_tensor = torch.stack(output_logits_list)
                # print("Output logits stacked:", output_logits_tensor.shape)
                combined_logits = torch.cat([context_logits, output_logits_tensor])
            else:
                combined_logits = context_logits

            # print("Combined logits shape:", combined_logits.shape)
            
            padding = torch.nn.functional.one_hot(
                torch.full(
                    (seqlen - combined_logits.shape[0],),
                    self.PAD_ID,
                    device=combined_logits.device
                ),
                num_classes=self.vocab_size)
            padded_combined_logits = torch.cat([combined_logits, padding])

            # print("Padded combined logits shape:", padded_combined_logits.shape)

            output_logits_batch.append(padded_combined_logits)
            
        return torch.stack(output_logits_batch).to(self.device) #(batch['input_ids'].device)

        #print("Decoded output:", self.tokenizer.decode(output_ids[0][0][cont_idxs[0]:].tolist()))
        """
            # Old logits logic, back before TRT-LLM natively returned logits
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
        """
