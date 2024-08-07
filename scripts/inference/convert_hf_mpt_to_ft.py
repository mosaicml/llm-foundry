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

import transformers

from llmfoundry.utils import convert_and_save_ft_weights


def convert_mpt_to_ft(
    model_name_or_path: str,
    output_dir: str,
    infer_gpu_num: int = 1,
    weight_data_type: str = 'fp32',
    force: bool = False,
) -> None:
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
        model_name_or_path,
        trust_remote_code=True,
    ).to(torch_device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
    )

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
            hf_config['bos_token_id'],
        ) if hf_config['bos_token_id'] != None else str(tokenizer.bos_token_id)
        config['gpt']['end_id'] = str(
            hf_config['eos_token_id'],
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
                'clip_qkv is enabled for this MPT model. This may not work as expected in FT. Use --force to force a conversion.',
            )
        if hf_config['attn_config']['qk_ln'] and not force:
            raise RuntimeError(
                'qk_ln is enabled for this MPT model. This may not work as expected in FT. Use --force to force a conversion.',
            )

        with open(os.path.join(save_dir, 'config.ini'), 'w') as configfile:
            config.write(configfile)
    except:
        print(f'Failed to save the config in config.ini.')
        raise

    named_params_dict = dict(model.named_parameters())
    convert_and_save_ft_weights(
        named_params=named_params_dict,
        config=hf_config,
        infer_gpu_num=infer_gpu_num,
        weight_data_type=weight_data_type,
        save_dir=save_dir,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        '--save_dir',
        '-o',
        type=str,
        help='Directory to save converted checkpoint in',
        required=True,
    )
    parser.add_argument(
        '--name_or_dir',
        '-i',
        type=str,
        help=
        'HF hub Model name (e.g., mosaicml/mpt-7b) or local dir path to load checkpoint from',
        required=True,
    )
    parser.add_argument(
        '--infer_gpu_num',
        '-i_g',
        type=int,
        help='How many gpus for inference?',
        required=True,
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help=
        'Force conversion to FT even if some features may not work as expected in FT',
    )
    parser.add_argument(
        '--weight_data_type',
        type=str,
        help='Data type of weights in the input checkpoint',
        default='fp32',
        choices=['fp32', 'fp16'],
    )

    args = parser.parse_args()
    print('\n=============== Argument ===============')
    for key in vars(args):
        print('{}: {}'.format(key, vars(args)[key]))
    print('========================================')

    convert_mpt_to_ft(
        args.name_or_dir,
        args.save_dir,
        args.infer_gpu_num,
        args.weight_data_type,
        args.force,
    )
