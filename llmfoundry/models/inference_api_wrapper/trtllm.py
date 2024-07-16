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
        # Only print on rank 0
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


    def eval_forward(self, batch, outputs: Optional[Any] = None):
        # Run TRT-LLM Forward Pass without any input padding
        output_logits_batch = []
        batch = self.rebatch(batch)
        batch_size = len(batch['input_ids'])
        prompt_lens = []
        unpadded_input_ids_list = []

        if 'continuation_indices' not in batch:
            # Question-answering tasks
            max_prompt_len = 0
            for tokens in batch['input_ids']:
                pad_start = (tokens == self.PAD_ID).nonzero(as_tuple=True)[0]
                eos_start = (tokens == self.END_ID).nonzero(as_tuple=True)[0]
                end_prompt_idx = len(tokens.tolist())
                if pad_start.shape[0] >= 1:
                    end_prompt_idx = pad_start[0]
                if eos_start.shape[0] >= 1 and eos_start[0] < end_prompt_idx:
                    end_prompt_idx = eos_start[0]
                prompt_lens.append(end_prompt_idx)
                if end_prompt_idx > max_prompt_len:
                    max_prompt_len = end_prompt_idx
            
            for i in range(batch_size):
                #print("Prompt:\n", self.tokenizer.decode(batch['input_ids'][i][:prompt_lens[i]].tolist()))
                unpadded_input_ids_list += batch['input_ids'][i][:prompt_lens[i]].tolist()

            unpadded_input_ids = torch.tensor(unpadded_input_ids_list, dtype=torch.int, device=self.device)
            input_lengths = torch.tensor(prompt_lens, dtype=torch.int, device=self.device)

            MAX_GEN_LEN = 256
            max_generation_length = batch.get('generation_length', MAX_GEN_LEN)
            torch.cuda.synchronize()
            with torch.no_grad():
                self.decoder.setup(batch_size,
                                    input_lengths.max().item(),
                                    max_generation_length)
                output_dict = self.decoder.decode(unpadded_input_ids, input_lengths, self.sampling_config, return_dict=True)
            torch.cuda.synchronize()

            output_ids = [output_dict['output_ids'][i][0].tolist()[prompt_lens[i]:prompt_lens[i]+max_generation_length] for i in range(batch_size)]
            decoded_strs = [self.tokenizer.decode(out) for out in output_ids]
            # print("Output string:", decoded_strs)
            return decoded_strs
        elif 'gold_indices' in batch:
            # Multiple choice tasks
            seqlen = batch['input_ids'].shape[1]
            """"
            Generate one-step at a time
            """
            prompt_lens = [cont_idxs.tolist()[-1] + 1 for cont_idxs in batch['continuation_indices']]            
            logits_list = []
            with torch.no_grad():
                for tokens, cont_idxs in zip(batch['input_ids'], batch['continuation_indices']):
                    cont_idxs = cont_idxs.tolist()
                    #print("Continuation Indices:", cont_idxs)
                    #print("Continuation tokens:", self.tokenizer.decode(tokens.tolist()[cont_idxs[0]:cont_idxs[-1] + 1]))
                    cont_length = cont_idxs[-1] + 1 - cont_idxs[0]
                    logits = torch.nn.functional.one_hot(
                                tokens[1:cont_idxs[0]],
                                num_classes=self.vocab_size,
                            ).to(device=self.device)
                    for i in range(cont_length):
                        # decode one token at a time
                        self.decoder.setup(1, cont_idxs[0]+i, 1)
                        output_dict = self.decoder.decode(tokens[:cont_idxs[0]+i].to(dtype=torch.int32, device=self.device),
                                            torch.tensor([cont_idxs[0]+i], dtype=torch.int, device=self.device), 
                                            self.sampling_config,
                                            return_dict=True)
                        next_logit_tensor = torch.squeeze(output_dict['generation_logits'][0])
                        #print("Decoded output:\n", self.tokenizer.decode(output_dict['output_ids'][0].squeeze()))
                        # append next logit to logits tensor
                        logits = torch.cat([logits, next_logit_tensor.reshape(1, -1)])

                    padding = torch.nn.functional.one_hot(
                                torch.full((max(prompt_lens) - logits.shape[0],), self.PAD_ID),
                                num_classes=self.vocab_size,
                            ).to(device=next_logit_tensor.device)
                    logits = torch.cat([logits, padding])
                    logits_list.append(logits)  
                
            return torch.stack(logits_list).to(device=self.device, dtype=torch.float)
            """
            Normal (context logits) version
            """
            torch.cuda.synchronize()
            continuation_starts = [cont_idxs.tolist()[0] for cont_idxs in batch['continuation_indices']]            
            prompt_lens = [cont_idxs.tolist()[-1] + 1 for cont_idxs in batch['continuation_indices']]            

            logits_list = []
            with torch.no_grad():
                # Batched version: 
                # Doesn't work because TRT-LLM has bug with batched context logits.
                """
                for i in range(batch_size):
                    #print("Prompt:", self.tokenizer.decode(batch['input_ids'][i][:prompt_lens[i]].tolist()))
                    unpadded_input_ids_list += batch['input_ids'][i][:prompt_lens[i]].tolist()

                unpadded_input_ids = torch.tensor(unpadded_input_ids_list, dtype=torch.int, device=self.device)
                input_lengths = torch.tensor(prompt_lens, dtype=torch.int, device=self.device)
                self.decoder.setup(batch_size,
                                    input_lengths.max().item(),
                                    1)
                output_dict = self.decoder.decode(unpadded_input_ids, input_lengths, self.sampling_config, return_dict=True)
                logits = output_dict['context_logits']
                """    
                # Unbatched version
                for tokens, cont_idxs in zip(batch['input_ids'], batch['continuation_indices']):
                    # Tensorrt-LLM Input must be int32 tensor, not int64 tensor!
                    prompt_len = cont_idxs.tolist()[-1] + 1
                    self.decoder.setup(1, prompt_len, 1)
                    output_dict = self.decoder.decode(tokens[:prompt_len].to(dtype=torch.int32, device=self.device),
                                        torch.tensor([prompt_len], dtype=torch.int, device=self.device), 
                                        self.sampling_config,
                                        return_dict=True)
                    context_logits = torch.squeeze(output_dict['context_logits'])
                    prompt_psuedologits = torch.nn.functional.one_hot(
                        tokens[1:cont_idxs[0]],
                        num_classes=self.vocab_size)
                    context_logits = context_logits[cont_idxs[0]:]
                    context_logits = torch.cat([prompt_psuedologits, context_logits])

                    pad_len = max(prompt_lens) - context_logits.shape[0]
                    if pad_len != 0:
                        padding = torch.nn.functional.one_hot(
                            torch.full((pad_len,), self.PAD_ID),
                            num_classes=self.vocab_size,
                        ).to(device=context_logits.device)
                        context_logits = torch.cat([context_logits, padding])

                    logits_list.append(context_logits)
            torch.cuda.synchronize()
            
            return torch.stack(logits_list).to(device=self.device, dtype=torch.float)
            """
            # Batched version 
            # Context Logits beyond input lengths should be one-hot vectors
            # with a one in the padding token position.        
            """
            for i in range(batch_size):
                pad_len = logits.shape[1] - prompt_lens[i]
                if pad_len == 0:
                    continue
                padding = torch.nn.functional.one_hot(
                    torch.full((pad_len,), self.PAD_ID),
                    num_classes=logits.shape[2],
                    )
                logits[i,prompt_lens[i]:,:] = padding
            return logits
        else:
            # Language Modeling Tasks
            seqlen = batch['input_ids'].shape[1]
            continuation_lens = []
            for tokens, cont_idxs in zip(batch['input_ids'],
                                        batch['continuation_indices']):            
                tokens = tokens.tolist()
                cont_idxs = cont_idxs.tolist()
                expected_cont_tokens = tokens[cont_idxs[0]:cont_idxs[-1] + 1]
                prompt_lens.append(cont_idxs[0])
                continuation_lens.append(len(expected_cont_tokens))
        
            for i in range(batch_size):
                unpadded_input_ids_list += batch['input_ids'][i][:prompt_lens[i]].tolist()
            
            unpadded_input_ids = torch.tensor(unpadded_input_ids_list, dtype=torch.int, device=self.device)
            input_lengths = torch.tensor(prompt_lens, dtype=torch.int, device=self.device)
            
            torch.cuda.synchronize()
            with torch.no_grad():
                self.decoder.setup(batch_size,
                                    input_lengths.max().item(),
                                    max(continuation_lens))
                output_dict = self.decoder.decode(unpadded_input_ids, input_lengths, self.sampling_config, return_dict=True)
            torch.cuda.synchronize()

            output_logits_list = output_dict['generation_logits']
            if len(output_logits_list) > 0:
                output_logits_tensor = torch.stack(output_logits_list, dim=1)
            else:
                output_logits_tensor = None 

            # Put together logits
            output_logits_batch = []
            for i in range(batch_size):
                prior_data = 0 if i == 0 else sum(prompt_lens[:i])
                # First create context "logits" (one-hot vector with 1 at token position)
                context_psuedologits = torch.nn.functional.one_hot(
                    torch.tensor(unpadded_input_ids_list[prior_data+1:prior_data+prompt_lens[i]], device=self.device),
                    num_classes=self.vocab_size)
                # Then add generation logits (up to continuation_length)
                if output_logits_tensor is not None:
                    output_logits_trimmed = output_logits_tensor[i][:continuation_lens[i]]
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

