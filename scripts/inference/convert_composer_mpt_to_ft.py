# Copyright 2022-2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

# Note: This script is specifically for converting MPT Composer checkpoints to FasterTransformer format.

import configparser
import os
import tempfile
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from composer.utils import get_file, safe_torch_load
from transformers import PreTrainedTokenizer

from llmfoundry.utils import (convert_and_save_ft_weights,
                              get_hf_tokenizer_from_composer_state_dict)


def save_ft_config(composer_config: Dict[str, Any],
                   tokenizer: PreTrainedTokenizer,
                   save_dir: str,
                   infer_gpu_num: int = 1,
                   weight_data_type: str = 'fp32',
                   force: bool = False):

    config = configparser.ConfigParser()
    config['gpt'] = {}
    try:
        config['gpt']['model_name'] = 'mpt'
        config['gpt']['head_num'] = str(composer_config['n_heads'])
        n_embd = composer_config['d_model']
        config['gpt']['size_per_head'] = str(n_embd //
                                             composer_config['n_heads'])
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
        save_dir: str,
        trust_remote_code: bool,
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
        trust_remote_code (bool): Whether or not to use code outside of the transformers module.
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
            'state'] or 'huggingface' not in composer_state_dict['state'][
                'integrations']:
        raise RuntimeError(
            'Did not find HuggingFace related state (e.g., tokenizer) in the provided composer checkpoint!'
        )
    composer_config = composer_state_dict['state']['integrations'][
        'huggingface']['model']['config']['content']

    # Extract the HF tokenizer
    print('#' * 30)
    print('Extracting HF Tokenizer...')
    hf_tokenizer = get_hf_tokenizer_from_composer_state_dict(
        composer_state_dict, trust_remote_code)
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
    convert_and_save_ft_weights(named_params=weights_state_dict,
                                config=composer_config,
                                infer_gpu_num=infer_gpu_num,
                                weight_data_type=output_precision,
                                save_dir=save_dir)

    print('#' * 30)
    print(
        f'FasterTransformer checkpoint folder successfully created at {save_dir}.'
    )

    print('Done.')
    print('#' * 30)


def parse_args() -> Namespace:
    """Parse commandline arguments."""
    parser = ArgumentParser(
        description=
        'Convert an MPT Composer checkpoint into a standard FasterTransformer checkpoint folder.'
    )
    parser.add_argument(
        '--composer_path',
        '-i',
        type=str,
        help='Composer checkpoint path. Can be a local file path or cloud URI',
        required=True)
    parser.add_argument(
        '--local_checkpoint_save_location',
        type=str,
        help='If specified, where to save the checkpoint file to locally. \
                            If the input ``checkpoint_path`` is already a local path, this will be a symlink. \
                            Defaults to None, which will use a temporary file.',
        default=None)
    parser.add_argument(
        '--ft_save_dir',
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
    parser.add_argument(
        '--output_precision',
        type=str,
        help=
        'Data type of weights in the FasterTransformer output model. Input checkpoint weights will be converted to this dtype.',
        choices=['fp32', 'fp16'],
        default='fp32')
    parser.add_argument(
        '--trust_remote_code',
        action='store_true',
        help='Whether or not to use code outside of transformers module.')

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
        local_checkpoint_save_location=args.local_checkpoint_save_location,
        trust_remote_code=args.trust_remote_code)
