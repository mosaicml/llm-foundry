# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0
import sys

from omegaconf import DictConfig
from omegaconf import OmegaConf as om

from llmfoundry.train.train import train


def main(cfg: DictConfig):
    train(cfg)


if __name__ == '__main__':
    yaml_path, args_list = sys.argv[1], sys.argv[2:]

    # Disable resolving environment variables through omegaconf.
    om.clear_resolver('oc.env')

    # Load yaml and cli arguments.
    with open(yaml_path) as f:
        yaml_cfg = om.load(f)
    cli_cfg = om.from_cli(args_list)
    cfg = om.merge(yaml_cfg, cli_cfg)
    assert isinstance(cfg, DictConfig)
    main(cfg)
