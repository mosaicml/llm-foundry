# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0
import sys
import logging

from llmfoundry.command_utils import train_from_yaml

if __name__ == '__main__':
    logging.basicConfig(filename='./tmp/train_logfile.log', filemode='w', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    yaml_path, args_list = sys.argv[1], sys.argv[2:]
    train_from_yaml(yaml_path, args_list)
