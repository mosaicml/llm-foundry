# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Convert MPT model checkpoint to FT format.

It's a modified version of
https://github.com/NVIDIA/FasterTransformer/blob/main/examples/pytorch/gpt/utils/huggingface_gpt_convert.py
"""

import argparse
import configparser
import os
from typing import Any, Dict, List

import numpy as np
import torch
import transformers


def get_weight_data_type(data_type: str):
    if data_type == 'fp32':
        return np.float32
    elif data_type == 'fp16':
        return np.float16
    else:
        raise RuntimeError('Unsupported data type: {data_type} for conversion')


def write_zero_bias(weight_name: str, weight_file_path: str,
                    bias_shape: List[int]) -> None:
    """Write zeros for bias.

    MPT model might not have bias while FT expects bias.

    Args:
        weight_name (str): Name of the weight tensor.
        weight_file_path (str): Output path for storing the weight (NOT zero bias).
        bias_shape (List[int]): Shape of the bias array.
    """
    if 'weight' not in weight_file_path:
        raise RuntimeError(
            f'Cannot write zero bias for {weight_name}. Input is not a weight tensor'
        )
    print(f'zero bias for weight: {weight_name}')
    bias_file_path = weight_file_path.replace('.weight', '.bias')
    bias = np.zeros(bias_shape, dtype=np.float32)
    bias.tofile(bias_file_path)


def convert_weight_to_ft_each(save_dir: str, infer_gpu_num: int,
                              tensor_name: str, config: Dict[str, Any],
                              data: np.ndarray):
    """Convert an MPT checkpoint to a FasterTransformer compatible format.

    Args:
        save_dir (str): Path of the directory to save the weight in FT format. The directory must already exist.
        infer_gpu_num (int): The number of gpus you are planning to use for inference.
        tensor_name (str): Name of the weight tensor. Used in naming the output file.
        config (Dict[str, Any]): Configuration for the model. This is used in getting model specific parameters.
        data (np.ndarray): Tensor data in np.ndarray format.

    Returns:
        None: Writes to a file in `save_dir`. File name is based on the `tensor_name`
    """
    if tensor_name.find('input_layernorm.weight') != -1 or tensor_name.find('input_layernorm.bias') != -1 or \
        tensor_name.find('attention.dense.bias') != -1 or tensor_name.find('post_attention_layernorm.weight') != -1 or \
        tensor_name.find('post_attention_layernorm.bias') != -1 or tensor_name.find('mlp.dense_4h_to_h.bias') != -1 or \
        tensor_name.find('final_layernorm.weight') != -1 or tensor_name.find('final_layernorm.bias') != -1:

        save_path = os.path.join(save_dir, f'model.{tensor_name}.bin')
        data.tofile(save_path)
        if 'weight' in tensor_name and config['no_bias']:
            write_zero_bias(tensor_name, save_path, data.shape[-1])

    elif tensor_name.find('attention.dense.weight') != -1:
        assert data.shape == (
            config['d_model'],
            config['d_model']), f'unexpected dim for {tensor_name}'
        # nn.Linear weights are transposed
        data = data.T
        split_vals = np.split(data, infer_gpu_num, axis=0)
        for j in range(infer_gpu_num):
            save_path = os.path.join(save_dir, f'model.{tensor_name}.{j}.bin')
            split_vals[j].tofile(save_path)
        if config['no_bias']:
            fake_weight_path = os.path.join(save_dir,
                                            f'model.{tensor_name}.bin')
            write_zero_bias(tensor_name, fake_weight_path, data.shape[-1])

    elif tensor_name.find('mlp.dense_4h_to_h.weight') != -1:
        assert data.shape == (
            config['d_model'], config['expansion_ratio'] *
            config['d_model']), f'unexpected dim for {tensor_name}'
        # nn.Linear weights are transposed
        data = data.T
        split_vals = np.split(data, infer_gpu_num, axis=0)
        for j in range(infer_gpu_num):
            save_path = os.path.join(save_dir, f'model.{tensor_name}.{j}.bin')
            split_vals[j].tofile(save_path)
        if config['no_bias']:
            fake_weight_path = os.path.join(save_dir,
                                            f'model.{tensor_name}.bin')
            write_zero_bias(tensor_name, fake_weight_path, data.shape[-1])

    elif tensor_name.find('mlp.dense_h_to_4h.weight') != -1:
        assert data.shape == (
            config['expansion_ratio'] * config['d_model'],
            config['d_model']), f'unexpected dim for {tensor_name}'
        # nn.Linear weights are transposed
        data = data.T

        split_vals = np.split(data, infer_gpu_num, axis=-1)
        for j in range(infer_gpu_num):
            save_path = os.path.join(save_dir, f'model.{tensor_name}.{j}.bin')
            split_vals[j].tofile(save_path)
            if config['no_bias']:
                write_zero_bias(tensor_name, save_path, split_vals[j].shape[-1])

    elif tensor_name.find('mlp.dense_h_to_4h.bias') != -1:
        assert data.shape == (
            config['expansion_ratio'] *
            config['d_model'],), f'unexpected dim for {tensor_name}'
        split_vals = np.split(data, infer_gpu_num, axis=-1)
        for j in range(infer_gpu_num):
            save_path = os.path.join(save_dir + f'model.{tensor_name}.{j}.bin')
            split_vals[j].tofile(save_path)

    elif tensor_name.find('attention.query_key_value.bias') != -1:
        assert data.shape == (
            3 * config['d_model'],), f'unexpected dim for {tensor_name}'

        data = data.reshape(3, config['d_model'])

        split_vals = np.split(data, infer_gpu_num, axis=-1)

        for j in range(infer_gpu_num):
            save_path = os.path.join(save_dir, f'model.{tensor_name}.{j}.bin')
            split_vals[j].tofile(save_path)

    elif tensor_name.find('attention.query_key_value.weight') != -1:
        assert data.shape == (
            3 * config['d_model'],
            config['d_model']), f'unexpected dim for {tensor_name}'
        # nn.Linear weights are transposed
        data = data.T

        data = data.reshape(config['d_model'], 3, config['d_model'])
        split_vals = np.split(data, infer_gpu_num, axis=-1)

        for j in range(infer_gpu_num):
            save_path = os.path.join(save_dir, f'model.{tensor_name}.{j}.bin')
            split_vals[j].tofile(save_path)
            if config['no_bias']:
                write_zero_bias(tensor_name, save_path,
                                (3, split_vals[j].shape[-1]))

    else:
        raise RuntimeError(f'Tensor with name {tensor_name} is not handled')


def convert_mpt_to_ft(model_name_or_path: str,
                      output_dir: str,
                      infer_gpu_num: int = 1,
                      weight_data_type: str = 'fp32',
                      force: bool = False) -> None:
    """Convert an MPT checkpoint to a FasterTransformer compatible format.

    Args:
        model_name_or_path (str): The HF hub name of the model (e.g., mosaicml/mpt-7b) or the path of a directory
            containing an MPT checkpoint in a local dir.
        output_dir (str): Path of the directory to save the checkpoint in FT format. The directory must not already exist.
        infer_gpu_num (int): The number of gpus you are planning to use for inference.
        weight_data_type (str): Data type of the weights in the input checkpoint.
        force (bool): force conversion even with unsupported features in FT.
    """
    save_dir = os.path.join(output_dir, f'{infer_gpu_num}-gpu')

    if (os.path.exists(save_dir) == False):
        os.makedirs(save_dir)
    else:
        raise RuntimeError(f'Output path {save_dir} already exists!')

    # do conversion on cpu
    torch_device = 'cpu'

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name_or_path, trust_remote_code=True).to(torch_device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)

    hf_config = vars(model.config)

    config = configparser.ConfigParser()
    config['gpt'] = {}
    try:
        config['gpt']['model_name'] = 'mpt' if hf_config[
            '_name_or_path'] == '' else hf_config['_name_or_path']
        config['gpt']['head_num'] = str(hf_config['n_heads'])
        n_embd = hf_config['d_model']
        config['gpt']['size_per_head'] = str(n_embd // hf_config['n_heads'])
        config['gpt']['inter_size'] = str(n_embd * hf_config['expansion_ratio'])
        config['gpt']['max_pos_seq_len'] = str(hf_config['max_seq_len'])
        config['gpt']['num_layer'] = str(hf_config['n_layers'])
        config['gpt']['vocab_size'] = str(hf_config['vocab_size'])
        config['gpt']['start_id'] = str(
            hf_config['bos_token_id']
        ) if hf_config['bos_token_id'] != None else str(tokenizer.bos_token_id)
        config['gpt']['end_id'] = str(
            hf_config['eos_token_id']
        ) if hf_config['eos_token_id'] != None else str(tokenizer.eos_token_id)
        config['gpt']['weight_data_type'] = weight_data_type
        config['gpt']['tensor_para_size'] = str(infer_gpu_num)
        # nn.LayerNorm default eps is 1e-5
        config['gpt']['layernorm_eps'] = str(1e-5)
        if hf_config['attn_config']['alibi']:
            config['gpt']['has_positional_encoding'] = str(False)
            config['gpt']['use_attention_linear_bias'] = str(True)
        if hf_config['attn_config']['clip_qkv'] and not force:
            raise RuntimeError(
                'clip_qkv is enabled for this MPT model. This may not work as expected in FT. Use --force to force a conversion.'
            )
        if hf_config['attn_config']['qk_ln'] and not force:
            raise RuntimeError(
                'qk_ln is enabled for this MPT model. This may not work as expected in FT. Use --force to force a conversion.'
            )

        with open(os.path.join(save_dir, 'config.ini'), 'w') as configfile:
            config.write(configfile)
    except:
        print(f'Failed to save the config in config.ini.')
        raise

    np_weight_data_type = get_weight_data_type(weight_data_type)

    param_remapping = {
        'norm_1.bias': 'input_layernorm.bias',
        'norm_1.weight': 'input_layernorm.weight',
        'attn.Wqkv.bias': 'attention.query_key_value.bias',
        'attn.Wqkv.weight': 'attention.query_key_value.weight',
        'attn.out_proj.bias': 'attention.dense.bias',
        'attn.out_proj.weight': 'attention.dense.weight',
        'norm_2.bias': 'post_attention_layernorm.bias',
        'norm_2.weight': 'post_attention_layernorm.weight',
        'ffn.up_proj.bias': 'mlp.dense_h_to_4h.bias',
        'ffn.up_proj.weight': 'mlp.dense_h_to_4h.weight',
        'ffn.down_proj.bias': 'mlp.dense_4h_to_h.bias',
        'ffn.down_proj.weight': 'mlp.dense_4h_to_h.weight',
    }

    for name, param in model.named_parameters():
        print(f'Working on parameter {name} ...')
        data = param.detach().cpu().numpy().astype(np_weight_data_type)
        if name.find('weight') == -1 and name.find('bias') == -1:
            print(f'found a parameter name that is not handled: {name}')
            continue
        if name == 'transformer.wpe.weight':
            assert data.shape == (
                hf_config['max_seq_len'],
                hf_config['d_model']), f'unexpected dim for {name}'
            data.tofile(os.path.join(save_dir, 'model.wpe.bin'))
        elif name == 'transformer.wte.weight':
            assert data.shape == (
                hf_config['vocab_size'],
                hf_config['d_model']), f'unexpected dim for {name}'
            data.tofile(os.path.join(save_dir, 'model.wte.bin'))
        elif name == 'transformer.norm_f.bias':
            assert data.shape == (
                hf_config['d_model'],), f'unexpected dim for {name}'
            data.tofile(os.path.join(save_dir,
                                     'model.final_layernorm.bias.bin'))
        elif name == 'transformer.norm_f.weight':
            assert data.shape == (
                hf_config['d_model'],), f'unexpected dim for {name}'
            save_path = os.path.join(save_dir,
                                     'model.final_layernorm.weight.bin')
            data.tofile(save_path)
            if hf_config['no_bias']:
                write_zero_bias(name, save_path, data.shape[-1])
        elif name == 'transformer.lm_head.weight':
            data.tofile(os.path.join(save_dir, 'model.lm_head.weight.bin'))
        else:
            for mpt_pattern, ft_pattern in param_remapping.items():
                if name.find(mpt_pattern) != -1:
                    new_name = name.replace('transformer.blocks.',
                                            'layers.').replace(
                                                mpt_pattern, ft_pattern)
                    convert_weight_to_ft_each(save_dir, infer_gpu_num, new_name,
                                              hf_config, data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--save_dir',
                        '-o',
                        type=str,
                        help='Directory to save converted checkpoint in',
                        required=True)
    parser.add_argument(
        '--name_or_dir',
        '-i',
        type=str,
        help=
        'HF hub Model name (e.g., mosaicml/mpt-7b) or local dir path to load checkpoint from',
        required=True)
    parser.add_argument('--infer_gpu_num',
                        '-i_g',
                        type=int,
                        help='How many gpus for inference?',
                        required=True)
    parser.add_argument(
        '--force',
        action='store_true',
        help=
        'Force conversion to FT even if some features may not work as expected in FT'
    )
    parser.add_argument('--weight_data_type',
                        type=str,
                        help='Data type of weights in the input checkpoint',
                        default='fp32',
                        choices=['fp32', 'fp16'])

    args = parser.parse_args()
    print('\n=============== Argument ===============')
    for key in vars(args):
        print('{}: {}'.format(key, vars(args)[key]))
    print('========================================')

    convert_mpt_to_ft(args.name_or_dir, args.save_dir, args.infer_gpu_num,
                      args.weight_data_type, args.force)
