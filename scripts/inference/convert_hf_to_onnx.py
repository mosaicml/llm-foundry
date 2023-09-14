# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Basic HuggingFace -> ONNX export script.

This scripts show a basic HuggingFace -> ONNX export workflow. This works for a MPT model
that has been saved using `MPT.save_pretrained`. For more details and examples
of exporting and working with HuggingFace models with ONNX, see https://huggingface.co/docs/transformers/serialization#export-to-onnx.

Example usage:

    1) Local export

    python convert_hf_to_onnx.py --pretrained_model_name_or_path local/path/to/huggingface/folder --output_folder local/folder

    2) Remote export

    python convert_hf_to_onnx.py --pretrained_model_name_or_path local/path/to/huggingface/folder --output_folder s3://bucket/remote/folder

    3) Verify the exported model

    python convert_hf_to_onnx.py --pretrained_model_name_or_path local/path/to/huggingface/folder --output_folder local/folder --verify_export

    4) Change the batch size or max sequence length

    python convert_hf_to_onnx.py --pretrained_model_name_or_path local/path/to/huggingface/folder --output_folder local/folder --export_batch_size 1 --max_seq_len 32000
"""

import argparse
import os
from argparse import ArgumentTypeError
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from composer.utils import (maybe_create_object_store_from_uri, parse_uri,
                            reproducibility)
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def str2bool(v: Union[str, bool]):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')


def str_or_bool(v: Union[str, bool]):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        return v


def gen_random_batch(batch_size: int, vocab_size: int, max_seq_len: int):
    # generate input batch of random data
    batch = {
        'input_ids':
            torch.randint(
                low=0,
                high=vocab_size,
                size=(batch_size, max_seq_len),
                dtype=torch.int64,
            ),
        'attention_mask':
            torch.ones(size=(batch_size, max_seq_len), dtype=torch.bool)
    }
    return batch


def export_to_onnx(
    pretrained_model_name_or_path: str,
    output_folder: str,
    export_batch_size: int,
    max_seq_len: Optional[int],
    verify_export: bool,
    from_pretrained_kwargs: Dict[str, Any],
):
    reproducibility.seed_all(42)
    save_object_store = maybe_create_object_store_from_uri(output_folder)
    _, _, parsed_save_path = parse_uri(output_folder)

    print('Loading HF config/model/tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path,
                                              **from_pretrained_kwargs)
    config = AutoConfig.from_pretrained(pretrained_model_name_or_path,
                                        **from_pretrained_kwargs)

    # specifically for MPT, switch to the torch version of attention for ONNX export
    if hasattr(config, 'attn_config'):
        config.attn_config['attn_impl'] = 'torch'

    model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path,
                                                 config=config,
                                                 **from_pretrained_kwargs)
    model.eval()

    if max_seq_len is None and not hasattr(model.config, 'max_seq_len'):
        raise ValueError(
            'max_seq_len must be specified in either the model config or as an argument to this function.'
        )
    elif max_seq_len is None:
        max_seq_len = model.config.max_seq_len

    assert isinstance(max_seq_len, int)  # pyright

    print('Creating random batch...')
    sample_input = gen_random_batch(
        export_batch_size,
        len(tokenizer),
        max_seq_len,
    )

    with torch.no_grad():
        model(**sample_input)

    output_file = Path(parsed_save_path) / 'model.onnx'
    os.makedirs(parsed_save_path, exist_ok=True)
    print('Exporting the model with ONNX...')
    torch.onnx.export(
        model,
        (sample_input,),
        str(output_file),
        input_names=['input_ids', 'attention_mask'],
        output_names=['output'],
        opset_version=16,
    )

    if verify_export:
        with torch.no_grad():
            orig_out = model(**sample_input)

        import onnx
        import onnx.checker
        import onnxruntime as ort

        _ = onnx.load(str(output_file))

        onnx.checker.check_model(str(output_file))

        ort_session = ort.InferenceSession(str(output_file))

        for key, value in sample_input.items():
            sample_input[key] = value.cpu().numpy()

        loaded_model_out = ort_session.run(None, sample_input)

        torch.testing.assert_close(
            orig_out.logits.detach().numpy(),
            loaded_model_out[0],
            rtol=1e-2,
            atol=1e-2,
            msg=f'output mismatch between the orig and onnx exported model',
        )
        print('exported model ouptut matches with unexported model!!')

    if save_object_store is not None:
        print('Uploading files to object storage...')
        for filename in os.listdir(parsed_save_path):
            full_path = str(Path(parsed_save_path) / filename)
            save_object_store.upload_object(full_path, full_path)


def parse_args():
    parser = argparse.ArgumentParser(description='Convert HF model to ONNX',)
    parser.add_argument(
        '--pretrained_model_name_or_path',
        type=str,
        required=True,
    )
    parser.add_argument(
        '--output_folder',
        type=str,
        required=True,
    )
    parser.add_argument(
        '--export_batch_size',
        type=int,
        default=8,
    )
    parser.add_argument(
        '--max_seq_len',
        type=int,
        default=None,
    )
    parser.add_argument(
        '--verify_export',
        action='store_true',
    )
    parser.add_argument('--trust_remote_code',
                        type=str2bool,
                        nargs='?',
                        const=True,
                        default=True)
    parser.add_argument('--use_auth_token',
                        type=str_or_bool,
                        nargs='?',
                        const=True,
                        default=None)
    parser.add_argument('--revision', type=str, default=None)
    return parser.parse_args()


def main(args: argparse.Namespace):
    from_pretrained_kwargs = {
        'use_auth_token': args.use_auth_token,
        'trust_remote_code': args.trust_remote_code,
        'revision': args.revision,
    }

    export_to_onnx(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        output_folder=args.output_folder,
        export_batch_size=args.export_batch_size,
        max_seq_len=args.max_seq_len,
        verify_export=args.verify_export,
        from_pretrained_kwargs=from_pretrained_kwargs)


if __name__ == '__main__':
    main(parse_args())
