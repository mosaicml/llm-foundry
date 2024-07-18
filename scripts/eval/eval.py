# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0
import sys

from llmfoundry.command_utils import eval_from_yaml

if __name__ == '__main__':
    yaml_path, args_list = sys.argv[1], sys.argv[2:]
    eval_from_yaml(yaml_path, args_list)
