# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Implements a TRT-LLM evaluation model wrapped around a
:class:`.ComposerModel`."""

import os
import sys
import json
from pathlib import Path
from typing import Any, Optional, List, Tuple

import warnings
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
    return 'rank{}.engine'.format(rank)


class TRTLLMEvalWrapper(InferenceAPIEvalWrapper):

    def __init__(
        self,
        model_cfg: DictConfig,
        tokenizer: PreTrainedTokenizer,
    ):
        check_if_trt_llm_installed()

        if tensorrt_llm.mpi_rank() != 0:
            f = open(os.devnull, 'w')
            sys.stdout = f
            sys.stderr = f 
        super().__init__(model_cfg, tokenizer)

        tensorrt_llm.logger.set_level(model_cfg['log_level'])

        # Load TRT config from file
        engine_dir = Path(model_cfg['engine_dir'])
        config_path = engine_dir / 'config.json'
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        pretrained_config = config['pretrained_config']
        quantization_config = pretrained_config['quantization']
        build_config = config['build_config']
        plugin_config = build_config['plugin_config']
 
        dtype = pretrained_config['dtype']
        tp_size = pretrained_config['mapping']['tp_size']
        pp_size = pretrained_config['mapping'].get('pp_size', 1)
        world_size = tp_size * pp_size

        assert world_size == tensorrt_llm.mpi_world_size(), \
            f'Engine world size ({world_size}) != Runtime world size ({tensorrt_llm.mpi_world_size()})'
        
        num_heads = pretrained_config['num_attention_heads'] // tp_size
        hidden_size = pretrained_config['hidden_size'] // tp_size
        
        max_batch_size = build_config['max_batch_size']
        vocab_size = pretrained_config['vocab_size']
        num_layers = pretrained_config['num_hidden_layers']
        
        use_gpt_attention_plugin = bool(plugin_config['gpt_attention_plugin'])
        remove_input_padding = plugin_config['remove_input_padding']
        if remove_input_padding:
            raise ValueError(
                'TRT-LLM Evaluation Wrapper does not support remove_input_padding.'
            )
       
        num_kv_heads = build_config.get('num_key_value_heads', num_heads)
        paged_kv_cache = plugin_config['paged_kv_cache']
        tokens_per_block = plugin_config['tokens_per_block']
        use_custom_all_reduce = plugin_config.get('use_custom_all_reduce',
                                                            False)
        quant_mode = QuantMode.from_quant_algo(
            quant_algo=quantization_config['quant_algo'],
            kv_cache_quant_algo=quantization_config['kv_cache_quant_algo'])
        
        if pretrained_config.get('multi_query_mode', False):
            tensorrt_llm.logger.warning(
                "`multi_query_mode` config is deprecated. Please rebuild the engine."
            )
            num_kv_heads = 1
        num_kv_heads = (num_kv_heads + tp_size - 1) // tp_size

        model_config = tensorrt_llm.runtime.ModelConfig(
            max_batch_size=max_batch_size,
            max_beam_width=1,
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
            gather_context_logits=build_config.get('gather_context_logits', False),
            gather_generation_logits=build_config.get('gather_generation_logits', False),
        ) 


        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.max_output_len = build_config['max_output_len']

        # Device and rank
        runtime_rank = tensorrt_llm.mpi_rank()
        runtime_mapping = tensorrt_llm.Mapping(world_size,
                                           runtime_rank,
                                           tp_size=tp_size,
                                           pp_size=pp_size)
        self.device_num = runtime_rank % runtime_mapping.gpus_per_node
        self.device = torch.device('cuda:' + str(self.device_num))
        torch.cuda.set_device(self.device_num)

        print("My rank:", runtime_rank)
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
        engine_name = 'rank{}.engine'.format(runtime_rank)
        serialize_path = engine_dir / engine_name
        with open(serialize_path, 'rb') as f:
            engine_buffer = f.read()

        self.decoder = tensorrt_llm.runtime.GenerationSession(
            model_config, engine_buffer, runtime_mapping, debug_mode=False)

        print("!!! Initialized generation session for rank:", runtime_rank)        
        torch.cuda.synchronize()


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
    

        # Remove potential additional dim, cast to int32
        """
        batch_input_ids = [
            x.flatten().type(torch.int32) for x in batch_input_ids
        ]
        input_lengths = [x.size(0) for x in batch_input_ids]
        max_length = max(input_lengths)
        # Right padding for trt-llm
        paddings = [
            torch.ones(max_length - l, dtype=torch.int32, device=self.device) * pad_id
            for l in input_lengths
        ]
        batch_input_ids = [
            torch.cat([x, pad]) for x, pad in zip(batch_input_ids, paddings)
        ]
        batch_input_ids = torch.stack(batch_input_ids).to(device=self.device)
        input_lengths = torch.tensor(input_lengths, dtype=torch.int32, device=self.device)
        return batch_input_ids, input_lengths
        """

    def eval_forward(self, batch, outputs: Optional[Any] = None):
        # Run TRTLLM forward pass
        output_logits_batch = []
        batch = self.rebatch(batch)

        # Question-answering tasks
        if 'continuation_indices' not in batch:
            # Batched version
            batch_size = len(batch['input_ids'])
            prompt_lens = []
            max_prompt_len = 0
            for tokens in batch['input_ids']:
                prompt = tokens.tolist()
                pad_start = (tokens == self.PAD_ID).nonzero(as_tuple=True)[0]
                end_prompt_idx = len(prompt)
                if pad_start.shape[0] >= 1:
                    end_prompt_idx = pad_start[0]
                prompt_lens.append(end_prompt_idx)
                if end_prompt_idx > max_prompt_len:
                    max_prompt_len = end_prompt_idx

            input_ids = torch.narrow(batch['input_ids'], 1, 0, max_prompt_len).to(dtype=torch.int, device=self.device) 
            input_lengths = torch.tensor(prompt_lens, dtype=torch.int, device=self.device)
           
            torch.set_printoptions(threshold=10_000)
            print("Prompt0:", input_ids[0])
            #print("Input shape:", input_ids.shape)
            #print("Input lengths:", input_lengths)
            max_generation_length = 256
            with torch.no_grad():
                self.decoder.setup(batch_size,
                                    input_lengths.max().item(),
                                    batch.get('generation_length', max_generation_length))
                output_dict = self.decoder.decode(input_ids, input_lengths, self.sampling_config, return_dict=True)
                #self.decoder.setup(1,
                #                    input_lengths[:1].max().item(),
                #                    batch.get('generation_length', max_generation_length))
                #output_dict2 = self.decoder.decode(input_ids[:1,:], input_lengths[:1], self.sampling_config, return_dict=True)

            #answer1 = output_dict['output_ids'][0].squeeze()[prompt_lens[0]:prompt_lens[0]+max_generation_length]
            #answer2 = output_dict2['output_ids'][0].squeeze()[prompt_lens[0]:prompt_lens[0]+max_generation_length]
            #all_equal = torch.equal(answer1, answer2)
            """
            if not all_equal:
                print("Prompt:", input_ids[0])
                print("Answer 1:", self.tokenizer.decode(answer1))
                print("Answer 2:", self.tokenizer.decode(answer2))
                print("Shape 1:", answer1.shape)
                print("Shape 2", answer2.shape)
                difference = answer1 - answer2
                nonzero_indices = difference.nonzero(as_tuple=True)
                nonzero = difference[difference.nonzero(as_tuple=True)]            
                print("EQUAL?", all_equal)
                print("Difference:", difference)
                print("nonzero indices:", nonzero_indices)
                print("Nonzero Elements", nonzero)
                quit()
            """
            output_ids = [output_dict['output_ids'][i][0].tolist()[prompt_lens[i]:prompt_lens[i]+batch.get('generation_length', max_generation_length)] for i in range(batch_size)]

            #print("Output:", output_ids)
        
            decoded_strs = [self.tokenizer.decode(out) for out in output_ids]
            print("decoded strs:", decoded_strs)
            return decoded_strs
            
            # Non-batched version
            """
            output_strs = []
            for tokens in batch['input_ids']:
                #print("RAW Tokens:", tokens)
                seqlen = tokens.shape[0]
                prompt = tokens.tolist()
                eos_occurence = (tokens == 2).nonzero(as_tuple=True)[0]
                end_prompt_idx = len(prompt)
                if eos_occurence.shape[0] >= 1:
                    end_prompt_idx = eos_occurence[0]
                prompt = prompt[:end_prompt_idx]
                input_ids = torch.tensor([prompt], dtype=torch.int, device=self.device)
                input_lengths = torch.tensor([input_ids.size(1)], dtype=torch.int, device=self.device)
                print("prompt:", self.tokenizer.decode(prompt))
                #print("promp tokens:", prompt)
                #print("Input lengths:", input_lengths)
                #print("Generation Length:", batch['generation_length'])
                #print("Batch keys:", batch.keys())                
                torch.cuda.synchronize()
                with torch.no_grad():
                    self.decoder.setup(batch_size=input_lengths.size(0),
                                        max_context_length=torch.max(input_lengths).item(),
                                        max_new_tokens=batch.get('generation_length', 200))
                    output_dict = self.decoder.decode(
                                    input_ids,
                                    input_lengths,
                                    self.sampling_config, 
                                    #stopping_criteria=batch['stopping_criteria'],
                                    return_dict=True)
                
                #print("Shape:", output_dict['output_ids'].shape)
                decoded_str = self.tokenizer.decode(output_dict['output_ids'][0][0].tolist()[len(prompt):])
                output_strs.append(decoded_str)
                print("Decoded OUTPUT:", decoded_str)
                #print("-------------")
                #print("Output ids:", output_dict['output_ids'][0][0].tolist())
            return output_strs
            """

        #################
        # Batched version of language modeling/multiple choice tasks
        batch_size = len(batch['input_ids'])
        seqlen = batch['input_ids'].shape[1]
        #print("Seq len:", seqlen)
        prompt_lens = []
        continuation_lens = []
        for tokens, cont_idxs in zip(batch['input_ids'],
                                     batch['continuation_indices']):            
            tokens = tokens.tolist()
            cont_idxs = cont_idxs.tolist()
            expected_cont_tokens = tokens[cont_idxs[0]:cont_idxs[-1] + 1]
            prompt = tokens[:cont_idxs[0]]
            prompt_lens.append(cont_idxs[0])
            continuation_lens.append(len(expected_cont_tokens))
    
        input_lengths = torch.tensor(prompt_lens, dtype=torch.int, device=self.device)
        input_ids = torch.full((batch_size, max(prompt_lens)), fill_value=self.PAD_ID, device=self.device, dtype=torch.int)
        for i in range(batch_size):
            input_ids[i][:prompt_lens[i]] = batch['input_ids'][i][:prompt_lens[i]]
        
        #print("New batch shape", input_ids.shape)
        #print("Continuation lengths:", continuation_lens)
        #print("Prompt:", input_ids)
        #print("Input shape:", input_ids.shape)
        #print("Input lengths:", input_lengths)
        with torch.no_grad():
            self.decoder.setup(batch_size,
                                input_lengths.max().item(),
                                 max(continuation_lens))
            output_dict = self.decoder.decode(input_ids, input_lengths, self.sampling_config, return_dict=True)
        torch.cuda.synchronize()

        output_logits_list = output_dict['generation_logits']
        #print("Output logits list", output_logits_list)
        # print("Output ids:", output_dict['output_ids'][0][0][cont_idxs[0]:].tolist())
        
        # output_logits_list length is == max(continuation_lens)
        # Output logits_list[i] is of shape (batch_size, vocab_size)
        if len(output_logits_list) > 0:
            #print("Shape:", output_logits_list[0].shape)
            output_logits_tensor = torch.stack(output_logits_list, dim=1)
        else:
            output_logits_tensor = None 
        
        #if output_logits_tensor is not None:
            #print("Output logits tensor shape:", output_logits_tensor.shape)

        # Put together logits
        # We loop through batch_size dimension rather than deal with NestedTensor
        output_logits_batch = []
        for i in range(batch_size):
            # First create context "logits" (one-hot vector with 1 at token position)
            tokens = input_ids[i].tolist()
            context_psuedologits = torch.nn.functional.one_hot(
                torch.tensor(tokens[1:prompt_lens[i]], device=self.device),
                num_classes=self.vocab_size)
            # Then add generation logits (up to continuation_length)
            if output_logits_tensor is not None:
                output_logits_trimmed = output_logits_tensor[i][:continuation_lens[i]]
                # print("Output logits trimmed shape:", output_logits_trimmed.shape)
                combined_logits = torch.cat([context_psuedologits, output_logits_trimmed])
            else:
                combined_logits = context_psuedologits
            # Then pad with Padding token "logits" to end of sequence length
            padding = torch.nn.functional.one_hot(
                torch.full(
                    (seqlen - combined_logits.shape[0],),
                    self.PAD_ID,
                    device=self.device
                ),
                num_classes=self.vocab_size)
            padded_combined_logits = torch.cat([combined_logits, padding])   
            output_logits_batch.append(padded_combined_logits)

        return torch.stack(output_logits_batch).to(self.device)
        ###############################################
        # NON BATCHED VERSION
        # Language modeling and multiple choice tasks
        """
        for tokens, cont_idxs in zip(batch['input_ids'],
                                     batch['continuation_indices']):
            # print("******************************")
            seqlen = tokens.shape[0]
            tokens = tokens.tolist()
            # print("Tokens:", tokens)
            # print("Continuation indices:", cont_idxs)
            cont_idxs = cont_idxs.tolist()
            expected_cont_tokens = tokens[cont_idxs[0]:cont_idxs[-1] + 1]
            # print("Expected continuation tokens:", expected_cont_tokens)
            prompt = tokens[:cont_idxs[0]]

            
            input_ids = torch.tensor([prompt], dtype=torch.int, device=self.device)
            input_lengths = torch.tensor([input_ids.size(1)],
                                         dtype=torch.int,
                                         device=self.device)
            # print("*** PROMPT:", self.tokenizer.decode(prompt))
            # print("Input device:", input_ids.get_device())
            # print("Input ids data:", input_ids, len(input_ids), input_ids[0].shape)
            # print("Input lengths:", input_lengths)
            #print("Expected continuation tokens:", len(expected_cont_tokens))
            with torch.no_grad():
                self.decoder.setup(input_lengths.size(0),
                               torch.max(input_lengths).item(),
                               len(expected_cont_tokens))

                output_dict = self.decoder.decode(
                    input_ids, input_lengths, self.sampling_config, return_dict=True)
            
            torch.cuda.synchronize()

            context_psuedologits = torch.nn.functional.one_hot(
                torch.tensor(tokens[1:cont_idxs[0]], device=self.device),
                num_classes=self.vocab_size)
            output_logits_list = output_dict['generation_logits']
            # print("Output ids:", output_dict['output_ids'][0][0][cont_idxs[0]:].tolist())
            for i in range(len(output_logits_list)):
                output_logits_list[i] = output_logits_list[i].squeeze()
                # print("*** Output string:", self.tokenizer.decode(output_dict['output_ids'][0][0][cont_idxs[0]:].tolist()))
            #print("Context logits:", context_psuedologits.shape)
            if len(output_logits_list) > 0:
                output_logits_tensor = torch.stack(output_logits_list)
                # print("Output logits stacked:", output_logits_tensor.shape)
                combined_logits = torch.cat([context_psuedologits, output_logits_tensor])
            else:
                combined_logits = context_psuedologits
            #print("Seqlen", seqlen)
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