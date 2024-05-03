# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Script to profile example packing."""
import os
from typing import Dict

from llmfoundry.data.packing import profile_packing

if __name__ == '__main__':
    from argparse import ArgumentParser, Namespace

    from omegaconf import OmegaConf as om

    from llmfoundry.utils import build_tokenizer

    def parse_args() -> Namespace:
        """Parse commandline arguments."""
        parser = ArgumentParser(
            description=
            'Profile packing_ratio choices for a particular workload.',
        )
        parser.add_argument(
            '--yaml-path',
            type=str,
            required=True,
            help='Path to the YAML that defines the workload to profile.',
        )
        parser.add_argument(
            '--num-devices',
            type=int,
            required=True,
            help='How many devices your run will use.',
        )
        parser.add_argument(
            '--min',
            type=float,
            required=True,
            help='Smallest packing_ratio to test. Must be >=1.',
        )
        parser.add_argument(
            '--max',
            type=float,
            required=True,
            help='Largest packing_ratio to test. Must be larger than `min`.',
        )
        parser.add_argument(
            '--num-packing-ratios',
            type=int,
            default=20,
            help=
            'Number of packing_ratio values (spaced between `min` and `max) to try.',
        )

        args = parser.parse_args()

        if not os.path.isfile(args.yaml_path):
            raise FileNotFoundError(
                '`yaml_path` does not correspond to any existing file.',
            )
        if args.num_devices < 1:
            raise ValueError('`num_devices` must be a positive integer.')
        if args.min < 1.0:
            raise ValueError('`min` must be >=1.0.')
        if args.max < args.min:
            raise ValueError('`max` cannot be less than `min`.')
        if args.num_packing_ratios < 1:
            raise ValueError('`num_packing_ratios` must be a positive integer.')
        return args

    args = parse_args()

    with open(args.yaml_path) as f:
        cfg = om.load(f)
    if 'parameters' in cfg:
        cfg = om.to_container(cfg.parameters)
        cfg = om.create(cfg)
    device_batch_size = cfg.global_train_batch_size // args.num_devices

    # Fetch a bunch of raw examples once, which we'll re-use
    if 'train_loader' not in cfg:
        raise ValueError('config must define train_loader')
    dataloader_cfg = cfg.train_loader

    # build tokenizer
    if 'tokenizer' not in cfg:
        raise ValueError('config must define tokenizer')

    resolved_tokenizer_cfg = om.to_container(cfg.tokenizer, resolve=True)
    if not isinstance(resolved_tokenizer_cfg, Dict):
        raise ValueError(
            'tokenizer config needs to be resolved by omegaconf into a Dict.',
        )
    tokenizer_cfg = resolved_tokenizer_cfg

    tokenizer_name = tokenizer_cfg['name']
    tokenizer_kwargs = tokenizer_cfg.get('kwargs', {})
    tokenizer = build_tokenizer(tokenizer_name, tokenizer_kwargs)

    results = profile_packing(
        dataloader_cfg,
        tokenizer,
        args.min,
        args.max,
        args.num_packing_ratios,
        device_batch_size,
    )

    header = '\n\n\n packing_ratio | % PADDING | % WASTE'
    fstr = '        {:5.1f}  |  {:5.2f}%   | {:6.2f}%'

    print(header)
    print('-' * len(header))
    for packing_ratio, padding, waste in results:
        print(fstr.format(packing_ratio, padding, waste))
