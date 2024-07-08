# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0
from llmfoundry.train.train import train

def main(yaml_path: str, args_list: list[str]):
    train(yaml_path, args_list)

if __name__ == '__main__':
    yaml_path, args_list = sys.argv[1], sys.argv[2:]
    main(yaml_path, args_list)
