# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
from composer.core import Callback
from composer.loggers.logger_destination import LoggerDestination
from composer.trainer import Trainer
from composer.utils import dist, get_device, reproducibility
from omegaconf import DictConfig
from omegaconf import OmegaConf as om
from rich.traceback import install

from llmfoundry.utils import (
    find_mosaicml_logger,
    log_eval_analytics,
    maybe_create_mosaicml_logger,
)

install()
from llmfoundry.utils.builders import (
    add_metrics_to_eval_loaders,
    build_callback,
    build_composer_model,
    build_evaluators,
    build_logger,
    build_tokenizer,
)
from llmfoundry.utils.config_utils import (
    EVAL_CONFIG_KEYS,
    EvalConfig,
    log_config,
    make_dataclass_and_log_config,
    process_init_device,
)
from llmfoundry.utils.registry_utils import import_file

log = logging.getLogger(__name__)

if __name__ == '__main__':
    yaml_path, args_list = sys.argv[1], sys.argv[2:]
    eval_from_yaml(yaml_path, args_list)
