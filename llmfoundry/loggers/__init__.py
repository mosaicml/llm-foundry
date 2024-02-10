# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from composer import (InMemoryLogger, MLFlowLogger, TensorboardLogger,
                      WandbLogger)

from llmfoundry import registry

registry.loggers.register('wandb', WandbLogger)
registry.loggers.register('tensorboard', TensorboardLogger)
registry.loggers.register('mlflow', MLFlowLogger)
registry.loggers.register('inmemory', InMemoryLogger)
registry.loggers.register('in_memory_logger', InMemoryLogger)
