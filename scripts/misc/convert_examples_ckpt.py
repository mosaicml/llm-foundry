# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from argparse import ArgumentParser

from llmfoundry.command_utils import convert_examples_ckpt_from_args

if __name__ == '__main__':
    parser = ArgumentParser(
        description=
        'Convert ckpt created with the examples repo into one usable by llmfoundry.',
    )
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--local_ckpt_path', type=str, default=None)

    args = parser.parse_args()

    convert_examples_ckpt_from_args(
        checkpoint_path=args.checkpoint_path,
        output_path=args.output_path,
        local_ckpt_path=args.local_ckpt_path,
    )
