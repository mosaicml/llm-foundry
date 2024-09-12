# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from argparse import ArgumentParser

from llmfoundry.command_utils import split_eval_set_from_args


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Split training dataset into train and eval sets",
    )
    parser.add_argument(
        "--data_path_folder", required=True, type=str, help="Path to the training dataset folder"
    )
    parser.add_argument(
        "--data_path_split", required=True, type=str, help="Path to the training dataset split"
    )
    parser.add_argument(
        "--output_path",
        required=True,
        type=str,
        help="Path to save the split dataset",
    )
    parser.add_argument(
        "--eval_split_ratio",
        required=False,
        type=float,
        default=0.1,
        help="Ratio of the dataset to use for evaluation. The remainder will be used for training",
    )
    parser.add_argument(
        "--max_eval_samples",
        required=False,
        type=int,
        default=None,
        help="Maximum number of samples to include in the eval set",
    )
    parser.add_argument(
        "--seed",
        required=False,
        type=int,
        default=42,
        help="Random seed for splitting the dataset",
    )
    args = parser.parse_args()
    split_eval_set_from_args(
        data_path_folder=args.data_path_folder,
        data_path_split=args.data_path_split,
        output_path=args.output_path,
        eval_split_ratio=args.eval_split_ratio,
        max_eval_samples=args.max_eval_samples,
        seed=args.seed,
    )
