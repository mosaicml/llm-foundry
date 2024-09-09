# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0
import sys

from llmfoundry.command_utils import train_from_yaml
from llmfoundry.utils.builders import build_tp_strategy, build_save_planner
from icecream import install
install()

if __name__ == '__main__':
    # build_save_planner('dummy')
    build_tp_strategy('ffn')
    yaml_path, args_list = sys.argv[1], sys.argv[2:]
    train_from_yaml(yaml_path, args_list)
