# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0
from typing import Type

from composer.core import Algorithm, Callback
from composer.loggers import LoggerDestination
from composer.optim import ComposerScheduler
from torch.optim import Optimizer

from llmfoundry.interfaces import CallbackWithConfig
from llmfoundry.utils.registry_utils import create

_loggers_description = """The loggers registry is used to register classes that implement the LoggerDestination interface.
These classes are used to log data from the training loop, and will be passed to the loggers arg of the Trainer. The loggers
will be constructed by directly passing along the specified kwargs to the constructor."""
loggers = create('llmfoundry',
                 'loggers',
                 generic_type=Type[LoggerDestination],
                 entry_points=True,
                 description=_loggers_description)

_callbacks_description = """The callbacks registry is used to register classes that implement the Callback interface.
These classes are used to interact with the Composer event system, and will be passed to the callbacks arg of the Trainer.
The callbacks will be constructed by directly passing along the specified kwargs to the constructor."""
callbacks = create('llmfoundry',
                   'callbacks',
                   generic_type=Type[Callback],
                   entry_points=True,
                   description=_callbacks_description)

_callbacks_with_config_description = """The callbacks_with_config registry is used to register classes that implement the CallbackWithConfig interface.
These are the same as the callbacks registry, except that they additionally take the full training config as an argument to their constructor."""
callbacks_with_config = create('llm_foundry.callbacks_with_config',
                               generic_type=Type[CallbackWithConfig],
                               entry_points=True,
                               description=_callbacks_with_config_description)

_optimizers_description = """The optimizers registry is used to register classes that implement the Optimizer interface.
The optimizer will be passed to the optimizers arg of the Trainer. The optimizer will be constructed by directly passing along the
specified kwargs to the constructor, along with the model parameters."""
optimizers = create('llmfoundry',
                    'optimizers',
                    generic_type=Type[Optimizer],
                    entry_points=True,
                    description=_optimizers_description)

_algorithms_description = """The algorithms registry is used to register classes that implement the Algorithm interface.
The algorithm will be passed to the algorithms arg of the Trainer. The algorithm will be constructed by directly passing along the
specified kwargs to the constructor."""
algorithms = create('llmfoundry',
                    'algorithms',
                    generic_type=Type[Algorithm],
                    entry_points=True,
                    description=_algorithms_description)

_schedulers_description = """The schedulers registry is used to register classes that implement the ComposerScheduler interface.
The scheduler will be passed to the schedulers arg of the Trainer. The scheduler will be constructed by directly passing along the
specified kwargs to the constructor."""
schedulers = create('llmfoundry',
                    'schedulers',
                    generic_type=Type[ComposerScheduler],
                    entry_points=True,
                    description=_schedulers_description)

__all__ = [
    'loggers',
    'callbacks',
    'callbacks_with_config',
    'optimizers',
    'algorithms',
    'schedulers',
    'create',
    'builder',
    'TypedRegistry',
]
