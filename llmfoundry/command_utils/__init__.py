# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0
<<<<<<< HEAD
from llmfoundry.command_utils.eval import (
    eval_from_yaml,
    evaluate,
)

__all__ = [
    'evaluate',
    'eval_from_yaml',
=======
from llmfoundry.command_utils.train import (
    TRAIN_CONFIG_KEYS,
    TrainConfig,
    train,
    train_from_yaml,
    validate_config,
)

__all__ = [
    'train',
    'train_from_yaml',
    'TrainConfig',
    'TRAIN_CONFIG_KEYS',
    'validate_config',
>>>>>>> origin/main
]
