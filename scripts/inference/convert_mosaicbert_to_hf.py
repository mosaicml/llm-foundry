# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import os
import tempfile
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Optional, Tuple, Union
import json
import torch
import transformers
from composer.models.huggingface import get_hf_config_from_composer_state_dict
from composer.utils import (get_file,
                            parse_uri, safe_torch_load)
from transformers import PretrainedConfig, PreTrainedTokenizerBase, AutoModelForMaskedLM
import requests
from llmfoundry.utils.huggingface_hub_utils import \
    edit_files_for_hf_compatibility


def download_file(url: str, file_name: Union[Path, str]):
    response = requests.get(url)
    
    if response.status_code == 200:
        with open(file_name, 'wb') as file:
            file.write(response.content)
        print(f'File downloaded as {file_name}')
    else:
        print(f'Failed to download file. Status code: {response.status_code}')


def write_huggingface_pretrained_from_composer_checkpoint(
    checkpoint_path: Union[Path, str],
    output_path: Union[Path, str],
    trust_remote_code: bool,
    output_precision: str = 'fp32',
    local_checkpoint_save_location: Optional[Union[Path, str]] = None,
    bert_config_path: Optional[str] = None,
) -> Tuple[PretrainedConfig, Optional[PreTrainedTokenizerBase]]:
    """Convert a Composer checkpoint to a pretrained HF checkpoint folder.

    Write a ``config.json`` and ``pytorch_model.bin``, like
    :meth:`transformers.PreTrainedModel.from_pretrained` expects, from a
    composer checkpoint.

    Args:
        checkpoint_path (Union[Path, str]): Path to the composer checkpoint, can be a local path, or a remote path beginning with ``s3://``, or another backend
            supported by :meth:`composer.utils.maybe_create_object_store_from_uri`.
        output_path (Union[Path, str]): Path to the folder to write the output to.
        trust_remote_code (bool): Whether or not to use code outside of the transformers module.
        output_precision (str, optional): The precision of the output weights saved to `pytorch_model.bin`. Can be one of ``fp32``, ``fp16``, or ``bf16``.
        local_checkpoint_save_location (Optional[Union[Path, str]], optional): If specified, where to save the checkpoint file to locally.
                                                                                If the input ``checkpoint_path`` is already a local path, this will be a symlink.
                                                                                Defaults to None, which will use a temporary file.
        bert_config_path (Optional[str], optional): Path to the bert config file. Defaults to None. A placeholder config from mosaicml/mosaic-bert-base will be used if not provided.
    """
    dtype = {
        'fp32': torch.float32,
        'fp16': torch.float16,
        'bf16': torch.bfloat16,
    }[output_precision]

    # default local path to a tempfile if path is not provided
    if local_checkpoint_save_location is None:
        tmp_dir = tempfile.TemporaryDirectory()
        local_checkpoint_save_location = Path(
            tmp_dir.name) / 'local-composer-checkpoint.pt'

    # create folder
    os.makedirs(output_path)

    # download the checkpoint file
    print(
        f'Downloading checkpoint from {checkpoint_path} -> {local_checkpoint_save_location}'
    )
    get_file(str(checkpoint_path), str(local_checkpoint_save_location))

    # Load the Composer checkpoint state dict
    print('Loading checkpoint into CPU RAM...')
    composer_state_dict = safe_torch_load(local_checkpoint_save_location)

    if bert_config_path is not None:
        #json
        with open(bert_config_path, 'r') as f:
            bert_config = json.load(f)   
        composer_state_dict["state"]["integrations"] = {"huggingface":{"model":{"config":{"content": bert_config}}}}

    else:
        # placeholder config from mosaicml/mosaic-bert-base
        composer_state_dict["state"]["integrations"]={"huggingface":{
            "model":{"config":{"content":{
                "_name_or_path": "mosaicml/mosaic-bert",
                "alibi_starting_size": 512,
                "architectures": [
                    "BertForMaskedLM"
                ],
                "attention_probs_dropout_prob": 0.1,
                "auto_map": {
                    "AutoConfig": "configuration_bert.BertConfig",
                    "AutoModelForMaskedLM": "bert_layers.BertForMaskedLM"
                },
                "classifier_dropout": None,
                "gradient_checkpointing": False,
                "hidden_act": "gelu",
                "hidden_dropout_prob": 0.1,
                "hidden_size": 768,
                "initializer_range": 0.02,
                "intermediate_size": 3072,
                "layer_norm_eps": 1e-12,
                "max_position_embeddings": 512,
                "model_type": "bert",
                "num_attention_heads": 12,
                "num_hidden_layers": 12,
                "pad_token_id": 0,
                "position_embedding_type": "absolute",
                "torch_dtype": "float32",
                "transformers_version": "4.26.0",
                "type_vocab_size": 2,
                "use_cache": False,
                "vocab_size": 30522

        }}}},
        "tokenizer":{}
    }
    if 'state' not in composer_state_dict:
        raise RuntimeError(
            f'"state" is not an available key in the provided composer checkpoint. Is {local_checkpoint_save_location} ill-formed?'
        )

    # Build and save HF Config
    print('#' * 30)
    print('Saving HF Model Config...')
    hf_config = get_hf_config_from_composer_state_dict(composer_state_dict)
    hf_config.torch_dtype = dtype
    hf_config.save_pretrained(output_path)
    print(hf_config)

    # Extract the HF model weights
    print('#' * 30)
    print('Saving HF Model Weights...')
    weights_state_dict = composer_state_dict
    if 'state' in weights_state_dict:
        weights_state_dict = weights_state_dict['state']['model']
    torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(
        weights_state_dict, prefix='model.')

    # Convert weights to desired dtype
    for k, v in weights_state_dict.items():
        if isinstance(v, torch.Tensor):
            weights_state_dict[k] = v.to(dtype=dtype)

    # Save weights
    torch.save(weights_state_dict, Path(output_path) / 'pytorch_model.bin')

    print('#' * 30)
    print(f'HF checkpoint folder successfully created at {output_path}.')

    return hf_config


def parse_args() -> Namespace:
    """Parse commandline arguments."""
    parser = ArgumentParser(
        description=
        'Convert a HuggingFace causal LM in a Composer checkpoint into a standard HuggingFace checkpoint folder, and optionally upload to the hub.'
    )
    parser.add_argument('--composer_path', type=str, required=True)
    parser.add_argument('--hf_output_path', type=str, required=True)
    
    parser.add_argument('--bert_config_path', type=str)
    parser.add_argument('--local_checkpoint_save_location',
                        type=str,
                        default=None)
    parser.add_argument('--output_precision',
                        type=str,
                        choices=['fp32', 'fp16', 'bf16'],
                        default='fp32')
    parser.add_argument('--hf_repo_for_upload', type=str, default=None)
    parser.add_argument('--test_uploaded_model', action='store_true')
    parser.add_argument(
        '--trust_remote_code',
        action='store_true',
        help='Whether or not to use code outside of transformers module.')

    return parser.parse_args()


def _convert_composer_to_hf(args: Namespace) -> None:
    print()
    print('#' * 30)
    print('Converting Composer checkpoint to HuggingFace checkpoint format...')

    _, _, local_folder_path = parse_uri(args.hf_output_path)

    config = write_huggingface_pretrained_from_composer_checkpoint(
        checkpoint_path=args.composer_path,
        output_path=local_folder_path,
        trust_remote_code=args.trust_remote_code,
        output_precision=args.output_precision,
        local_checkpoint_save_location=args.local_checkpoint_save_location,
        bert_config_path=args.bert_config_path)


    dtype = {
        'fp32': torch.float32,
        'fp16': torch.float16,
        'bf16': torch.bfloat16,
    }[args.output_precision]

    print(f'Loading model from {local_folder_path}')

    download_file("https://huggingface.co/mosaicml/mosaic-bert-base/raw/main/bert_layers.py", f"{local_folder_path}/bert_layers.py")
    download_file("https://huggingface.co/mosaicml/mosaic-bert-base/raw/main/bert_padding.py", f"{local_folder_path}/bert_padding.py")
    download_file("https://huggingface.co/mosaicml/mosaic-bert-base/raw/main/configuration_bert.py", f"{local_folder_path}/configuration_bert.py")
    download_file("https://huggingface.co/mosaicml/mosaic-bert-base/raw/main/flash_attn_triton.py", f"{local_folder_path}/flash_attn_triton.py")


    config = transformers.BertConfig.from_pretrained(local_folder_path)
    loaded_hf_model = AutoModelForMaskedLM.from_pretrained(local_folder_path,config=config,trust_remote_code=True)


    loaded_hf_model.save_pretrained(local_folder_path)

    # Only need to edit files for MPT because it has custom code
    if config.model_type == 'mpt':
        print('Editing files for HF compatibility...')
        edit_files_for_hf_compatibility(local_folder_path)

    if args.hf_repo_for_upload is not None:
        from huggingface_hub import HfApi
        api = HfApi()

        print(
            f'Uploading {args.hf_output_path} to HuggingFace Hub at {args.hf_repo_for_upload}'
        )
        api.create_repo(repo_id=args.hf_repo_for_upload,
                        use_auth_token=True,
                        repo_type='model',
                        private=True,
                        exist_ok=True)
        print('Repo created.')

        # ignore the full checkpoint file if we now have sharded checkpoint files
        ignore_patterns = []
        if any(
                f.startswith('pytorch_model-00001')
                for f in os.listdir(args.hf_output_path)):
            ignore_patterns.append('pytorch_model.bin')

        api.upload_folder(folder_path=args.hf_output_path,
                          repo_id=args.hf_repo_for_upload,
                          use_auth_token=True,
                          repo_type='model',
                          ignore_patterns=ignore_patterns)
        print('Folder uploaded.')

        if args.test_uploaded_model:
            print('Testing uploaded model...')
            hub_config = transformers.BertConfig.from_pretrained(args.hf_repo_for_upload)
            hub_model = AutoModelForMaskedLM.from_pretrained(args.hf_repo_for_upload,config=hub_config,trust_remote_code=True)


            assert sum(p.numel() for p in hub_model.parameters()) == sum(
                p.numel() for p in loaded_hf_model.parameters())
            assert all(
                str(type(module1)).split('.')[-2:] == str(type(module2)).split(
                    '.')[-2:] for module1, module2 in zip(
                        hub_model.modules(), loaded_hf_model.modules()))

            assert next(
                hub_model.parameters()
            ).dtype == dtype, f'Expected model dtype to be {dtype}, but got {next(hub_model.parameters()).dtype}'
            
    print(
        'Composer checkpoint successfully converted to HuggingFace checkpoint format.'
    )


def convert_composer_to_hf(args: Namespace) -> None:

    try:
        _convert_composer_to_hf(args)
    except Exception as e:
        raise e

if __name__ == '__main__':
    convert_composer_to_hf(parse_args())
