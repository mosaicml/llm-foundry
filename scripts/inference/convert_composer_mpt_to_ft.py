# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

# Note: This script is specifically for converting MPT Composer checkpoints to FasterTransformer format.

import json
import os
import tempfile
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Dict, Optional, Union, Tuple
import configparser
import numpy as np
import sentencepiece as spm

import torch
from composer.utils import (get_file, safe_torch_load)
from transformers import AutoTokenizer, PreTrainedTokenizer


def get_weight_data_type(data_type: str):
    if data_type == 'fp32':
        return np.float32
    elif data_type == 'fp16':
        return np.float16
    else:
        raise RuntimeError('Unsupported data type: {data_type} for conversion')


# TODO: maybe move this functionality to Composer
def get_hf_tokenizer_from_composer_state_dict(
        state_dict: Dict[str, Any]) -> Optional[PreTrainedTokenizer]:
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
        with tempfile.TemporaryDirectory() as _tmp_dir:
            for filename, saved_content in hf_tokenizer_state.items():
                tokenizer_file_path = Path(
                    _tmp_dir) / f'{filename}{saved_content["file_extension"]}'
                if saved_content['file_extension'] == '.json':
                    with open(tokenizer_file_path, 'w') as _tmp_file:
                        json.dump(saved_content['content'], _tmp_file)
                elif saved_content['file_extension'] == '.txt':
                    with open(tokenizer_file_path, 'w') as _tmp_file:
                        for line in saved_content['content']:
                            _tmp_file.write(line)
                            _tmp_file.write('\n')
                elif saved_content['file_extension'] == '.model':
                    s = spm.SentencePieceProcessor()
                    s.load_from_serialized_proto(saved_content['content'])
                    with open(tokenizer_file_path, 'wb') as _tmp_file:
                        _tmp_file.write(s.serialized_model_proto())
            hf_tokenizer = AutoTokenizer.from_pretrained(_tmp_dir)

            # remove 'name_or_path'
            hf_tokenizer.name_or_path = ''
            hf_tokenizer.init_kwargs['name_or_path'] = ''

    return hf_tokenizer


def write_zero_bias(weight_name: str, weight_file_path: str,
                    bias_shape: Union[Tuple[int, ...], int]) -> None:
    """Write zeros for bias.

    MPT model might not have bias while FT expects bias.

    Args:
        weight_name (str): Name of the weight tensor.
        weight_file_path (str): Output path for storing the weight (NOT zero bias).
        bias_shape (Union[Tuple[int, ...], int]): Shape of the bias array.
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
            config['d_model'], config['mlp_ratio'] *
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
            config['mlp_ratio'] * config['d_model'],
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
            config['mlp_ratio'] *
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


def convert_and_save_weights(named_params: dict,
                     composer_config: dict,
                     infer_gpu_num: int = 1,
                     weight_data_type: str = 'fp32',
                     save_dir: str = ""):
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

    for name, param in named_params.items():
        print(f'Working on parameter {name} ...')
        data = param.detach().cpu().numpy().astype(np_weight_data_type)
        if name.find('weight') == -1 and name.find('bias') == -1:
            print(f'found a parameter name that is not handled: {name}')
            continue
        if name == 'transformer.wpe.weight':
            assert data.shape == (
                composer_config['max_seq_len'],
                composer_config['d_model']), f'unexpected dim for {name}'
            data.tofile(os.path.join(save_dir, 'model.wpe.bin'))
        elif name == 'transformer.wte.weight':
            assert data.shape == (
                composer_config['vocab_size'],
                composer_config['d_model']), f'unexpected dim for {name}'
            data.tofile(os.path.join(save_dir, 'model.wte.bin'))
        elif name == 'transformer.norm_f.bias':
            assert data.shape == (
                composer_config['d_model'],), f'unexpected dim for {name}'
            data.tofile(os.path.join(save_dir,
                                     'model.final_layernorm.bias.bin'))
        elif name == 'transformer.norm_f.weight':
            assert data.shape == (
                composer_config['d_model'],), f'unexpected dim for {name}'
            save_path = os.path.join(save_dir,
                                     'model.final_layernorm.weight.bin')
            data.tofile(save_path)
            if composer_config['no_bias']:
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
                                              composer_config, data)


def save_ft_config(composer_config: dict,
                tokenizer: PreTrainedTokenizer,
                save_dir,
                infer_gpu_num: int = 1,
                weight_data_type: str = 'fp32',
                force: bool = False):

    config = configparser.ConfigParser()
    config['gpt'] = {}
    try:
        config['gpt']['model_name'] = 'mpt'
        config['gpt']['head_num'] = str(composer_config['n_heads'])
        n_embd = composer_config['d_model']
        config['gpt']['size_per_head'] = str(n_embd // composer_config['n_heads'])
        config['gpt']['inter_size'] = str(n_embd * composer_config['mlp_ratio'])
        config['gpt']['max_pos_seq_len'] = str(composer_config['max_seq_len'])
        config['gpt']['num_layer'] = str(composer_config['n_layers'])
        config['gpt']['vocab_size'] = str(composer_config['vocab_size'])
        config['gpt']['start_id'] = str(tokenizer.bos_token_id)
        config['gpt']['end_id'] = str(tokenizer.eos_token_id)
        config['gpt']['weight_data_type'] = weight_data_type
        config['gpt']['tensor_para_size'] = str(infer_gpu_num)
        # nn.LayerNorm default eps is 1e-5
        config['gpt']['layernorm_eps'] = str(1e-5)
        if composer_config['alibi']:
            config['gpt']['has_positional_encoding'] = str(False)
            config['gpt']['use_attention_linear_bias'] = str(True)
        if composer_config['attn_clip_qkv'] and not force:
            raise RuntimeError(
                'clip_qkv is enabled for this MPT model. This may not work as expected in FT. Use --force to force a conversion.'
            )
        if composer_config['attn_qk_ln'] and not force:
            raise RuntimeError(
                'qk_ln is enabled for this MPT model. This may not work as expected in FT. Use --force to force a conversion.'
            )

        with open(os.path.join(save_dir, 'config.ini'), 'w') as configfile:
            config.write(configfile)
        return config
    except:
        print(f'Failed to save the config in config.ini.')
        raise


def write_ft_checkpoint_from_composer_checkpoint(
        checkpoint_path: Union[Path, str],
        infer_gpu_num: int,
        save_dir: Union[Path, str],
        output_precision: str = 'fp32',
        local_checkpoint_save_location: Optional[Union[Path,
                                                       str]] = None) -> None:
    """Convert a Composer checkpoint to a FasterTransformer checkpoint folder.

    .. note:: This function may not work properly if you used surgery algorithms when you trained your model. In that case you may need to
        edit the parameter conversion methods to properly convert your custom model.

    Args:
        checkpoint_path (Union[Path, str]): Path to the composer checkpoint, can be a local path, or a remote path beginning with ``s3://``, or another backend
            supported by Composer.
        infer_gpu_num (int): The number of gpus you are planning to use for inference.
        save_dir (str): Path of the directory to save the checkpoint in FT format.
        output_precision (str, optional): The precision of the output weights saved to the FasterTransformer model. Can be either ``fp32`` or ``fp16``.
        local_checkpoint_save_location (Optional[Union[Path, str]], optional): If specified, where to save the checkpoint file to locally.
                                                                                If the input ``checkpoint_path`` is already a local path, this will be a symlink.
                                                                                Defaults to None, which will use a temporary file.
    """
    dtype = {
        'fp32': torch.float32,
        'fp16': torch.float16,
    }[output_precision]

    # default local path to a tempfile if path is not provided
    if local_checkpoint_save_location is None:
        tmp_dir = tempfile.TemporaryDirectory()
        local_checkpoint_save_location = Path(
            tmp_dir.name) / 'local-composer-checkpoint.pt'


    # download the checkpoint file
    print(
        f'Downloading checkpoint from {checkpoint_path} -> {local_checkpoint_save_location}'
    )
    get_file(str(checkpoint_path), str(local_checkpoint_save_location))

    # Load the Composer checkpoint. Use it to get the
    # Composer state dict and weights
    print('Loading checkpoint into CPU RAM...')
    composer_state_dict = safe_torch_load(local_checkpoint_save_location)

    # Extract Composer config from state dict
    if 'state' not in composer_state_dict:
        raise RuntimeError(
            f'"state" is not an available key in the provided composer checkpoint. Is {local_checkpoint_save_location} ill-formed?'
        )
    if 'integrations' not in composer_state_dict[
            'state'] or 'huggingface' not in composer_state_dict['state']['integrations']:
        raise RuntimeError(
            'Did not find HuggingFace related state (e.g., tokenizer) in the provided composer checkpoint!'
        )
    composer_config = composer_state_dict['state']['integrations']['huggingface'][
        'model']['config']['content']


    # Extract the HF tokenizer
    print('#' * 30)
    print('Extracting HF Tokenizer...')
    hf_tokenizer = get_hf_tokenizer_from_composer_state_dict(
        composer_state_dict)
    if hf_tokenizer is None:
        print('Warning! No HF Tokenizer found!')

    # Extract the model weights
    weights_state_dict = composer_state_dict['state']['model']
    torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(
        weights_state_dict, prefix='model.')

    # Converting weights to desired dtype
    for k, v in weights_state_dict.items():
        if isinstance(v, torch.Tensor):
            weights_state_dict[k] = v.to(dtype=dtype)

    # Convert the weights using the config and tokenizer to FasterTransformer format
    print('#' * 30)
    print('Saving FasterTransformer config...')
    save_ft_config(composer_config,
                   tokenizer=hf_tokenizer,
                   save_dir=save_dir,
                   weight_data_type=output_precision)
    print('#' * 30)
    print('Converting weights to FasterTransformer format...')
    convert_and_save_weights(named_params=weights_state_dict,
                     composer_config=composer_config,
                     infer_gpu_num=infer_gpu_num,
                     weight_data_type=output_precision,
                     save_dir=save_dir)

    print('#' * 30)
    print(f'FasterTransformer checkpoint folder successfully created at {save_dir}.')

    print('Done.')
    print('#' * 30)


def parse_args() -> Namespace:
    """Parse commandline arguments."""
    parser = ArgumentParser(
        description=
        'Convert an MPT Composer checkpoint into a standard FasterTransformer checkpoint folder.'
    )
    parser.add_argument('--composer_path',
                        '-i',
                        type=str,
                        help='Composer checkpoint path. Can be a local file path or cloud URI',
                        required=True)
    parser.add_argument('--local_checkpoint_save_location',
                        type=str,
                        help='If specified, where to save the checkpoint file to locally. \
                            If the input ``checkpoint_path`` is already a local path, this will be a symlink. \
                            Defaults to None, which will use a temporary file.',
                        default=None)
    parser.add_argument('--ft_save_dir',
                        '-o',
                        type=str,
                        help='Directory to save FasterTransformer converted checkpoint in',
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
    parser.add_argument('--output_precision',
                        type=str,
                        help='Data type of weights in the FasterTransformer output model. Input checkpoint weights will be converted to this dtype.',
                        choices=['fp32', 'fp16'],
                        default='fp32')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print('\n=============== Argument ===============')
    for key in vars(args):
        print('{}: {}'.format(key, vars(args)[key]))
    print('========================================')

    save_dir = os.path.join(args.ft_save_dir, f'{args.infer_gpu_num}-gpu')

    if os.path.exists(save_dir) == False:
        os.makedirs(save_dir)
    else:
        raise RuntimeError(f'Output path {save_dir} already exists!')

    write_ft_checkpoint_from_composer_checkpoint(
        checkpoint_path=args.composer_path,
        infer_gpu_num=args.infer_gpu_num,
        save_dir=save_dir,
        output_precision=args.output_precision,
        local_checkpoint_save_location=args.local_checkpoint_save_location)