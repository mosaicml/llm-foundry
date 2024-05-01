# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Helper methods for the checkpoint conversion scripts.

The checkpoint conversion scripts are located in the
llmfoundry/scripts/inference/benchmarking/ folder. Users should run those
scripts directly to convert between checkpoints; this file contains only common
utility functions that are present in multiple scripts.
"""

import json
import logging
import os
import random
import string
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from transformers import (AutoTokenizer, PreTrainedTokenizer,
                          PreTrainedTokenizerFast)

log = logging.getLogger(__name__)

__all__ = [
    'get_hf_tokenizer_from_composer_state_dict',
    'load_tokenizer',
    'convert_and_save_ft_weights',
]


def _get_weight_data_type(data_type: str):
    if data_type == 'fp32':
        return np.float32
    elif data_type == 'fp16':
        return np.float16
    else:
        raise RuntimeError('Unsupported data type: {data_type} for conversion.')


# TODO: move this functionality to composer once the bug fixes are upstreamed
def get_hf_tokenizer_from_composer_state_dict(
    state_dict: Dict[str, Any],
    trust_remote_code: bool,
    tokenizer_save_dir: Optional[str] = None,
) -> Optional[PreTrainedTokenizer]:
    if 'state' not in state_dict:
        raise RuntimeError(
            'Unexpected composer state dictionary. Did you pass in a full composer checkpoint?'
        )
    if 'integrations' not in state_dict[
            'state'] or 'huggingface' not in state_dict['state']['integrations']:
        raise RuntimeError(
            'Did not find HuggingFace related state (e.g., tokenizer) in the provided composer checkpoint!'
        )
    hf_tokenizer_state = state_dict['state']['integrations']['huggingface'][
        'tokenizer']
    hf_tokenizer = None
    if hf_tokenizer_state != {}:
        if tokenizer_save_dir is None:
            unique_suffix = ''.join(
                random.choices(string.ascii_letters + string.digits, k=6))
            tokenizer_save_dir = os.path.join(
                os.getcwd(), f'tokenizer-save-dir-{unique_suffix}')
        os.makedirs(tokenizer_save_dir, exist_ok=True)

        for filename, saved_content in hf_tokenizer_state.items():
            # For backwards compatibility, check if the filename already has the file extension
            if filename.endswith(saved_content['file_extension']):
                tokenizer_file_name = filename
            else:
                tokenizer_file_name = filename + saved_content['file_extension']

            # This cannot be a temporary directory because huggingface relies on the slow tokenizer file
            # being persistent on disk
            tokenizer_file_path = Path(tokenizer_save_dir) / tokenizer_file_name
            if saved_content['file_extension'] == '.json':
                with open(tokenizer_file_path, 'w') as _tmp_file:
                    json.dump(saved_content['content'], _tmp_file)
            elif saved_content['file_extension'] == '.txt':
                with open(tokenizer_file_path, 'w') as _tmp_file:
                    for line in saved_content['content']:
                        _tmp_file.write(line)
                        _tmp_file.write('\n')
            elif saved_content['file_extension'] == '.py':
                with open(tokenizer_file_path, 'w') as _tmp_file:
                    _tmp_file.write(saved_content['content'])
            elif saved_content['file_extension'] == '.model':
                try:
                    import sentencepiece as spm
                except ImportError as e:
                    raise ImportError(
                        'To load SentencePiece model, you need to install `sentencepiece`.'
                    ) from e

                s = spm.SentencePieceProcessor()
                s.load_from_serialized_proto(saved_content['content'])
                with open(tokenizer_file_path, 'wb') as _tmp_file:
                    _tmp_file.write(s.serialized_model_proto())

        hf_tokenizer = load_tokenizer(tokenizer_save_dir,
                                      trust_remote_code=trust_remote_code)

        # remove 'name_or_path'
        hf_tokenizer.name_or_path = ''
        hf_tokenizer.init_kwargs['name_or_path'] = ''

    return hf_tokenizer


def load_tokenizer(
    tokenizer_save_dir: str, trust_remote_code: bool
) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
    try:
        return AutoTokenizer.from_pretrained(
            tokenizer_save_dir, trust_remote_code=trust_remote_code)
    except ValueError as e:
        raise ValueError(
            f'Got error while loading tokenizer with trust_remote_code={trust_remote_code}: {e}. '
            +
            'If accessing a tokenizer defined outside of the transformers module,'
            + ' please use --trust_remote_code.')


def _write_zero_bias(weight_name: str, weight_file_path: str,
                     bias_shape: Union[Tuple[int, ...],
                                       int], np_data_type: np.dtype) -> None:
    """Write zeros for bias when converting MPT to FasterTransformer weights.

    MPT model might not have bias while FT expects bias.

    Args:
        weight_name (str): Name of the weight tensor.
        weight_file_path (str): Output path for storing the weight (NOT zero bias).
        bias_shape (Union[Tuple[int, ...], int]): Shape of the bias array.
        np_data_type (np.dtype): The data type for bias.
    """
    if 'weight' not in weight_file_path:
        raise RuntimeError(
            f'Cannot write zero bias for {weight_name}. Input is not a weight tensor'
        )
    log.debug(f'zero bias for weight: {weight_name}')
    bias_file_path = weight_file_path.replace('.weight', '.bias')
    bias = np.zeros(bias_shape, dtype=np_data_type)
    bias.tofile(bias_file_path)


def _convert_weight_to_ft_each(save_dir: str, infer_gpu_num: int,
                               tensor_name: str, config: Dict[str, Any],
                               data: np.ndarray,
                               np_weight_data_type: np.dtype) -> None:
    """Convert each MPT weight to a FasterTransformer compatible format.

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
            _write_zero_bias(tensor_name, save_path, data.shape[-1],
                             np_weight_data_type
                            )  # pyright: ignore [reportGeneralTypeIssues]

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
            _write_zero_bias(tensor_name, fake_weight_path, data.shape[-1],
                             np_weight_data_type
                            )  # pyright: ignore [reportGeneralTypeIssues]

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
            _write_zero_bias(tensor_name, fake_weight_path, data.shape[-1],
                             np_weight_data_type
                            )  # pyright: ignore [reportGeneralTypeIssues]

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
                _write_zero_bias(tensor_name, save_path,
                                 split_vals[j].shape[-1], np_weight_data_type
                                )  # pyright: ignore [reportGeneralTypeIssues]

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
                _write_zero_bias(tensor_name, save_path,
                                 (3, split_vals[j].shape[-1]),
                                 np_weight_data_type
                                )  # pyright: ignore [reportGeneralTypeIssues]

    else:
        raise RuntimeError(f'Tensor with name {tensor_name} is not handled')


def convert_and_save_ft_weights(named_params: dict,
                                config: dict,
                                infer_gpu_num: int = 1,
                                weight_data_type: str = 'fp32',
                                save_dir: str = '') -> None:
    """Convert a Composer MPT checkpoint to a FasterTransformer format.

    Args:
        named_params (Dict[str, Parameter]): A dictionary containing the Composer MPT model's parameter names and data.
        config (Dict[str, Any]): Configuration for the model. This is used in getting model specific parameters.
        infer_gpu_num (int): The number of gpus you are planning to use for inference.
        weight_data_type (str): The dtype of the converted FasterTransformer model.
        save_dir (str): Path of the directory to save the weight in FT format. The directory must already exist.

    Returns:
        None: Writes to the `save_dir` folder. File names within this folder are based on the model parameter names.
    """
    np_weight_data_type = _get_weight_data_type(weight_data_type)

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

    for name, param in named_params.items():
        log.debug(f'Working on parameter {name} ...')
        data = param.detach().cpu().numpy().astype(np_weight_data_type)
        if name.find('weight') == -1 and name.find('bias') == -1:
            log.debug(f'found a parameter name that is not handled: {name}')
            continue
        if name == 'transformer.wpe.weight':
            assert data.shape == (
                config['max_seq_len'],
                config['d_model']), f'unexpected dim for {name}'
            data.tofile(os.path.join(save_dir, 'model.wpe.bin'))
        elif name == 'transformer.wte.weight':
            assert data.shape == (
                config['vocab_size'],
                config['d_model']), f'unexpected dim for {name}'
            data.tofile(os.path.join(save_dir, 'model.wte.bin'))
        elif name == 'transformer.norm_f.bias':
            assert data.shape == (
                config['d_model'],), f'unexpected dim for {name}'
            data.tofile(os.path.join(save_dir,
                                     'model.final_layernorm.bias.bin'))
        elif name == 'transformer.norm_f.weight':
            assert data.shape == (
                config['d_model'],), f'unexpected dim for {name}'
            save_path = os.path.join(save_dir,
                                     'model.final_layernorm.weight.bin')
            data.tofile(save_path)
            if config['no_bias']:
                _write_zero_bias(
                    name,
                    save_path,
                    data.shape[-1],
                    np_weight_data_type  # pyright: ignore [reportGeneralTypeIssues]
                )
        elif name == 'transformer.lm_head.weight':
            data.tofile(os.path.join(save_dir, 'model.lm_head.weight.bin'))
        else:
            for mpt_pattern, ft_pattern in param_remapping.items():
                if name.find(mpt_pattern) != -1:
                    new_name = name.replace('transformer.blocks.',
                                            'layers.').replace(
                                                mpt_pattern, ft_pattern)
                    _convert_weight_to_ft_each(
                        save_dir,
                        infer_gpu_num,
                        new_name,
                        config,
                        data,
                        np_weight_data_type  # pyright: ignore [reportGeneralTypeIssues]
                    )
