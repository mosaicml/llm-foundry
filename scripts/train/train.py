# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0
from llmfoundry.train.train import train


if __name__ == '__main__':
    yaml_path, args_list = sys.argv[1], sys.argv[2:]
    train(yaml_path, args_list)
