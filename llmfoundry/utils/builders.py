# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import functools
import logging
import os
import re
import warnings
from collections import OrderedDict
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch
from composer import algorithms
from composer.callbacks import (EarlyStopper, Generate, LRMonitor,
                                MemoryMonitor, OptimizerMonitor,
                                RuntimeEstimator, SpeedMonitor)
from composer.core import Algorithm, Callback, Evaluator
from composer.datasets.in_context_learning_evaluation import \
    get_icl_task_dataloader
from composer.loggers import (InMemoryLogger, LoggerDestination, MLFlowLogger,
                              TensorboardLogger, WandBLogger)
from composer.optim import DecoupledAdamW
from composer.optim.scheduler import (ComposerScheduler,
                                      ConstantWithWarmupScheduler,
                                      CosineAnnealingWithWarmupScheduler,
                                      LinearWithWarmupScheduler)
from composer.utils import dist
from omegaconf import DictConfig, ListConfig
from omegaconf import OmegaConf as om
from torch.optim.optimizer import Optimizer
from torchmetrics import Metric
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from llmfoundry.callbacks import (AsyncEval, EvalGauntlet, FDiffMetrics,
                                  GlobalLRScaling, HuggingFaceCheckpointer,
                                  LayerFreezing, MonolithicCheckpointSaver,
                                  ScheduledGarbageCollector)
from llmfoundry.data.dataloader import build_dataloader
from llmfoundry.optim import (DecoupledAdaLRLion, DecoupledClipLion,
                              DecoupledLionW, DecoupledLionW_8bit)
from llmfoundry.optim.scheduler import InverseSquareRootWithWarmupScheduler
from llmfoundry.tokenizers.tiktoken import TiktokenTokenizerWrapper

log = logging.getLogger(__name__)


def build_evaluators(
    eval_loader_config: Optional[Union[DictConfig, ListConfig]],
    icl_tasks_config: Optional[Union[str, ListConfig]],
    eval_gauntlet_config: Optional[Union[str, DictConfig]],
    *,
    tokenizer: PreTrainedTokenizerBase,
    device_eval_batch_size: int,
    icl_seq_len: int,
    icl_subset_num_batches: Optional[int],
) -> Tuple[List[Evaluator], List[str], Optional[EvalGauntlet]]:

    evaluators = []
    if eval_loader_config is not None:
        evaluators = build_eval_loaders(
            eval_loader_config,
            tokenizer,
            device_eval_batch_size,
        )

    logger_keys = []
    eval_gauntlet_callback = None
    if icl_tasks_config is not None:
        icl_evaluators, logger_keys, eval_gauntlet_callback = build_icl_data_and_gauntlet(
            icl_tasks_config,
            eval_gauntlet_config,
            tokenizer,
            device_eval_batch_size,
            icl_seq_len,
            icl_subset_num_batches,
        )
        evaluators.extend(icl_evaluators)

    return evaluators, logger_keys, eval_gauntlet_callback


def build_eval_loaders(
    eval_loader_config: Union[DictConfig, ListConfig],
    tokenizer: PreTrainedTokenizerBase,
    device_eval_batch_size: int,
) -> List[Evaluator]:
    evaluators: List[Evaluator] = []
    if isinstance(eval_loader_config, ListConfig):
        eval_configs: ListConfig = eval_loader_config
        is_multi_eval = True
    else:
        eval_configs = ListConfig([eval_loader_config])
        is_multi_eval = False

    for eval_config in eval_configs:
        eval_dataloader = build_dataloader(eval_config, tokenizer,
                                           device_eval_batch_size)
        eval_loader: Evaluator = Evaluator(
            label=f'eval/{eval_config.label}' if is_multi_eval else 'eval',
            dataloader=eval_dataloader,
            # Load the eval data to fail fast. metrics will get added
            # later in add_metrics_to_eval_loaders, after the model is loaded
            metric_names=[],
        )
        evaluators.append(eval_loader)
    return evaluators


def add_metrics_to_eval_loaders(
    evaluators: List[Evaluator],
    metrics: Dict[str, Metric],
) -> List[Evaluator]:
    metric_names = list(metrics.keys())
    eval_loaders, other_evaluators = [], []
    for evaluator in evaluators:
        if evaluator.metric_names == []:
            evaluator.metric_names = metric_names
            eval_loaders.append(evaluator)
        else:
            other_evaluators.append(evaluator)

    # Put the base eval_loaders first
    return eval_loaders + other_evaluators


def build_icl_data_and_gauntlet(
    icl_tasks_config: Union[str, ListConfig],
    eval_gauntlet_config: Optional[Union[str, DictConfig]],
    tokenizer: PreTrainedTokenizerBase,
    device_eval_batch_size: int,
    icl_seq_len: int,
    icl_subset_num_batches: Optional[int] = None
) -> Tuple[List[Evaluator], List[str], Optional[EvalGauntlet]]:
    icl_evaluators, logger_keys = build_icl_evaluators(
        icl_tasks_config,
        tokenizer,
        icl_seq_len,
        device_eval_batch_size,
        icl_subset_num_batches=icl_subset_num_batches)
    eval_gauntlet_cb = None
    if eval_gauntlet_config is not None:
        if isinstance(eval_gauntlet_config, str):
            with open(eval_gauntlet_config, 'r') as icl_f:
                eval_gauntlet_cfg = om.load(icl_f)
            eval_gauntlet = eval_gauntlet_cfg.eval_gauntlet
        elif isinstance(eval_gauntlet_config, DictConfig):  # pyright: ignore
            eval_gauntlet = eval_gauntlet_config
        else:
            raise ValueError(
                f'Got invalid type for eval_gauntlet_config: {type(eval_gauntlet_config)}'
            )
        eval_gauntlet.logger_keys = logger_keys
        eval_gauntlet.benchmark_sizes = {
            e.label: e.dataloader.num_samples for e in icl_evaluators
        }
        eval_gauntlet_cb = EvalGauntlet(**eval_gauntlet)
    return icl_evaluators, logger_keys, eval_gauntlet_cb


def build_callback(
    name: str,
    kwargs: Union[DictConfig, Dict[str, Any]],
    config: Any = None,
) -> Callback:
    if name == 'lr_monitor':
        return LRMonitor()
    elif name == 'memory_monitor':
        return MemoryMonitor()
    elif name == 'speed_monitor':
        return SpeedMonitor(window_size=kwargs.get('window_size', 1),
                            gpu_flops_available=kwargs.get(
                                'gpu_flops_available', None))
    elif name == 'fdiff':
        return FDiffMetrics(**kwargs)
    elif name == 'runtime_estimator':
        return RuntimeEstimator()
    elif name == 'optimizer_monitor':
        return OptimizerMonitor(log_optimizer_metrics=kwargs.get(
            'log_optimizer_metrics', True),)
    elif name == 'generate_callback':
        prompts = kwargs.pop('prompts')
        interval = kwargs.pop('interval', None)
        # Generate callback used to be batch_log_interval, so this is for backwards compatibility
        if interval is None:
            batch_log_interval: str = kwargs.pop('batch_log_interval', '')
            if batch_log_interval:
                interval = f'{batch_log_interval}ba'
                warnings.warn(
                    ('generate_callback.batch_log_interval is deprecated and will be removed in a future release.'
                     f'Please use interval: {interval}'),
                    DeprecationWarning,
                )
            else:
                raise KeyError(
                    '"interval" must be specified with generate callback')
        return Generate(prompts=list(prompts), interval=interval, **kwargs)
    elif name == 'global_lr_scaling':
        return GlobalLRScaling(**kwargs)
    elif name == 'layer_freezing':
        return LayerFreezing(**kwargs)
    elif name == 'mono_ckpt_saver':
        return MonolithicCheckpointSaver(**kwargs)
    elif name == 'scheduled_gc':
        return ScheduledGarbageCollector(**kwargs)
    elif name == 'early_stopper':
        return EarlyStopper(**kwargs)
    elif name == 'hf_checkpointer':
        if isinstance(kwargs, DictConfig):
            kwargs = om.to_object(kwargs)  # pyright: ignore
        return HuggingFaceCheckpointer(**kwargs)
    elif name == 'async_eval':
        if config is None:
            raise ValueError(
                'Parameters config is required for async eval callback')

        return AsyncEval(**kwargs, training_params=config)
    else:
        raise ValueError(f'Not sure how to build callback: {name}')


def build_logger(name: str, kwargs: Dict[str, Any]) -> LoggerDestination:
    if name == 'wandb':
        return WandBLogger(**kwargs)
    elif name == 'tensorboard':
        return TensorboardLogger(**kwargs)
    elif name == 'in_memory_logger':
        return InMemoryLogger(**kwargs)
    elif name == 'mlflow':
        return MLFlowLogger(**kwargs)
    elif name == 'inmemory':
        return InMemoryLogger(**kwargs)
    else:
        raise ValueError(f'Not sure how to build logger: {name}')


def build_algorithm(name: str, kwargs: Dict[str, Any]) -> Algorithm:
    if name == 'gradient_clipping':
        return algorithms.GradientClipping(**kwargs)
    elif name == 'alibi':
        return algorithms.Alibi(**kwargs)
    elif name == 'gated_linear_units':
        return algorithms.GatedLinearUnits(**kwargs)
    elif name == 'low_precision_layernorm':
        return algorithms.LowPrecisionLayerNorm(**kwargs)
    else:
        raise ValueError(f'Not sure how to build algorithm: {name}')


def _extract_param_groups(
    model: torch.nn.Module,
    optimizer_config: Dict[str, Any],
) -> Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]]:
    """Extracts parameter groups defined in the optimizer config.

    The optimizer_config defines the optimizer args. It can additionally have key
    `disable_grad` which is a string or list of strings. If a string matches a
    parameter name, then that parameter will have `requires_grad=False`. This is
    useful for freezing parameters. It can additionally have a key
    `param_groups` which is a list of dicts. In this dict, key `param_str_match`
    defines a string; if a parameter name contains this string, then it will be
    in this parameter group. This is useful for grouping parameters together.
    The dict can also contain any other key that is a valid optimizer arg.
    Note: to handle name overlap conflicts, params are assigned to parameter
    groups and added to `param_groups` in the order that `param_str_match` appear
    in `param_groups`.

    Usage
    To disable gradient for all parameters that contain the string "norm" or "bias":
    ```
    optimizer_config: {
        "name": "decoupled_lionw",
        "lr": 1e-3,
        "weight_decay": 1e-2,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "disable_grad": ["norm", "bias"]
    }
    ```

    To create and modify the optimizer parameters for all parameters that contain
    the string "norm" and "bias" separately:
    ```
    optimizer_config: {
        "name": "decoupled_lionw",
        "lr": 1e-3,
        "weight_decay": 1e-2,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "param_groups": [
            {
                "param_str_match": "norm",
                "lr": 1e-4,
                "weight_decay": 0.0,
            },
            {
                "param_str_match": "bias",
                "lr": 5e-4,
                "weight_decay": 0.0,
            },
        ],
    }
    ```

    Args:
        model (torch.nn.Module): model to extract parameters from
        optimizer_config (Dict[str, Any]): optimizer config

    Returns:
        Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]]: an iterable of
            torch.Tensor's or dict's. Specifies what Tensors should be optimized
            and their param groupings.
    """
    if 'disable_grad' in optimizer_config.keys():
        str_matches = optimizer_config.pop('disable_grad')
        if isinstance(str_matches, str):
            str_matches = [str_matches]
        for str_match in str_matches:
            for n, p in model.named_parameters():
                if re.search(str_match, n):
                    p.requires_grad = False
                    log.debug(f'Setting `{n}.requires_grad = False`.')

    param_groups_config = optimizer_config.pop('param_groups', None)
    if param_groups_config is not None:
        params = []
        param_dict = OrderedDict((n, p) for n, p in model.named_parameters())

        log.debug(f'Default optimizer settings: {optimizer_config}.')
        for param_group_config in param_groups_config:
            str_match = param_group_config.pop('param_str_match')
            filter_fn = functools.partial(re.search, str_match)
            param_names = [n for n in param_dict.keys() if filter_fn(n)]
            group_params = {'params': [param_dict.pop(n) for n in param_names]}
            group_params.update(param_group_config)

            log.debug(
                f'Creating optimizer param_group with parameters: {param_names} ' +\
                f'(extracted using {str_match=}). The param_group optimizer ' +\
                f'setting overrides are: {param_group_config}.')

            params.append(group_params)

        params.insert(0, {'params': param_dict.values()})
        return params

    return model.parameters()


def build_optimizer(model: torch.nn.Module, name: str,
                    optimizer_config: Dict[str, Any]) -> Optimizer:

    params = _extract_param_groups(model, optimizer_config)

    if name == 'decoupled_adamw':
        return DecoupledAdamW(params, **optimizer_config)
    elif name == 'decoupled_lionw':
        return DecoupledLionW(params, **optimizer_config)
    elif name == 'clip_lion':
        return DecoupledClipLion(params, **optimizer_config)
    elif name == 'adalr_lion':
        return DecoupledAdaLRLion(params, **optimizer_config)
    elif name == 'decoupled_lionw_8b':
        return DecoupledLionW_8bit(params, **optimizer_config)
    else:
        raise ValueError(f'Not sure how to build optimizer: {name}')


def build_scheduler(name: str,
                    scheduler_config: Dict[str, Any]) -> ComposerScheduler:
    if name == 'constant_with_warmup':
        return ConstantWithWarmupScheduler(**scheduler_config)
    elif name == 'cosine_with_warmup':
        return CosineAnnealingWithWarmupScheduler(**scheduler_config)
    elif name == 'inv_sqrt_with_warmup':
        return InverseSquareRootWithWarmupScheduler(**scheduler_config)
    elif name == 'linear_decay_with_warmup':
        return LinearWithWarmupScheduler(**scheduler_config)
    else:
        raise ValueError(f'Not sure how to build scheduler: {name}')


def build_tokenizer(
        tokenizer_name: str,
        tokenizer_kwargs: Dict[str, Any]) -> PreTrainedTokenizerBase:
    os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    signal_file_path = f'.node_{dist.get_node_rank()}_local_rank0_completed_tokenizer_setup'

    if dist.is_available() and dist.is_initialized(
    ) and dist.get_world_size() > 1:
        # Make sure the tokenizer files are downloaded and cached first by local rank 0
        with dist.local_rank_zero_download_and_wait(signal_file_path):
            pass

    if tokenizer_name.startswith('tiktoken'):
        tokenizer = TiktokenTokenizerWrapper(**tokenizer_kwargs)
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name,
                                                  **tokenizer_kwargs)

        # HuggingFace does not respect the model_max_length kwarg, and overrides it with
        # min(kwargs['model_max_length'], original_config['model_max_length']), so we
        # explicitly set it here
        tokenizer.model_max_length = tokenizer_kwargs.get(
            'model_max_length',
            int(1e30),
        )

    if dist.is_available() and dist.is_initialized(
    ) and dist.get_world_size() > 1:
        if dist.get_local_rank() == 0:
            with open(signal_file_path, 'wb') as f:
                f.write(b'local_rank0_completed_tokenizer_setup')

        dist.barrier()

        if dist.get_local_rank() == 0:
            os.remove(signal_file_path)

    return tokenizer


def build_icl_evaluators(
    icl_tasks: Union[str, ListConfig],
    tokenizer: PreTrainedTokenizerBase,
    default_max_seq_len: int,
    default_batch_size: int,
    destination_dir: Optional[str] = None,
    icl_subset_num_batches: Optional[int] = None,
) -> Tuple[List[Evaluator], List[str]]:
    if destination_dir is None:
        destination_dir = os.getcwd()

    evaluators = []
    logger_keys = []

    icl_tasks_list = None
    if isinstance(icl_tasks, str):
        log.info(f'Extracting ICL task config from path: {icl_tasks}')
        with open(icl_tasks, 'r') as icl_f:
            icl_task_cfg = om.load(icl_f)
        icl_tasks_list = icl_task_cfg.icl_tasks
    else:
        icl_tasks_list = icl_tasks

    def _validate_cfg(icl_cfg: DictConfig):
        assert 'label' in icl_cfg
        assert 'dataset_uri' in icl_cfg and icl_cfg.dataset_uri is not None
        assert 'icl_task_type' in icl_cfg
        assert 'num_fewshot' in icl_cfg

        if 'metric_names' not in icl_cfg:
            if icl_cfg.icl_task_type == 'language_modeling':
                icl_cfg.metric_names = ['InContextLearningLMAccuracy']
            elif icl_cfg.icl_task_type == 'multiple_choice':
                icl_cfg.metric_names = [
                    'InContextLearningMultipleChoiceAccuracy'
                ]
            elif icl_cfg.icl_task_type == 'schema':
                icl_cfg.metric_names = [
                    'InContextLearningMultipleChoiceAccuracy'
                ]
            elif icl_cfg.icl_task_type == 'question_answering':
                icl_cfg.metric_names = ['InContextLearningQAAccuracy']
            elif icl_cfg.icl_task_type == 'code_evaluation':
                icl_cfg.metric_names = ['InContextLearningCodeEvalAccuracy']
            else:
                raise ValueError(
                    f'No metric_names defined, unable to build default metrics for icl_task_type={icl_cfg.icl_task_type}.'
                )

        if 'prompt_string' not in icl_cfg:
            icl_cfg.prompt_string = ''
        if 'example_delimiter' not in icl_cfg:
            icl_cfg.example_delimiter = '\n'
        if 'continuation_delimiter' not in icl_cfg:
            icl_cfg.continuation_delimiter = ' '
        if 'max_seq_len' not in icl_cfg:
            icl_cfg.max_seq_len = default_max_seq_len
        if 'batch_size' not in icl_cfg:
            icl_cfg.batch_size = default_batch_size
        if 'pass_at_k' not in icl_cfg:
            icl_cfg.pass_at_k = 1
        if 'num_beams' not in icl_cfg:
            icl_cfg.num_beams = 20

    for icl_cfg in icl_tasks_list:
        assert isinstance(icl_cfg, DictConfig)
        _validate_cfg(icl_cfg)
        for num_fewshot in list(icl_cfg.num_fewshot):
            if tokenizer.pad_token_id is None:
                # Current workaround to support GPT2 tokenizer with `pad_token_id = None`
                pad_tok_id = tokenizer.eos_token_id
            else:
                pad_tok_id = tokenizer.pad_token_id
            label = f'{icl_cfg.label}/{num_fewshot}-shot'
            metric_names = list(icl_cfg.metric_names)
            # TODO: fix Composer bug when copying local paths and destination exists
            destination_path = f'{destination_dir}/{icl_cfg.label}-{num_fewshot}.jsonl'
            if dist.get_local_rank() == 0 and os.path.exists(destination_path):
                os.remove(destination_path)
            dist.barrier()
            early_stopping_criteria = icl_cfg.get('early_stopping_criteria',
                                                  None)
            early_stopping_criteria = list(
                early_stopping_criteria
            ) if early_stopping_criteria is not None else None
            dataloaders = get_icl_task_dataloader(
                icl_cfg.icl_task_type,
                icl_cfg.dataset_uri,
                tokenizer,
                batch_size=icl_cfg.batch_size,
                max_seq_len=icl_cfg.max_seq_len,
                pad_tok_id=pad_tok_id,
                num_fewshot=num_fewshot,
                prompt_string=icl_cfg.prompt_string,
                example_delimiter=icl_cfg.example_delimiter,
                continuation_delimiter=icl_cfg.continuation_delimiter,
                question_prelimiter=icl_cfg.get('question_prelimiter', ''),
                destination_path=destination_path,
                pass_at_k=icl_cfg.pass_at_k,
                generations_per_sample=icl_cfg.num_beams,
                has_categories=icl_cfg.get('has_categories', False),
                cot_delimiter=icl_cfg.get('cot_delimiter', ''),
                early_stopping_criteria=early_stopping_criteria,
                do_normalization=icl_cfg.get('do_normalization', True))
            if hasattr(
                    icl_cfg,
                    'has_categories') and icl_cfg.has_categories and isinstance(
                        dataloaders, dict):
                for category in dataloaders.keys():
                    logger_keys.extend([
                        f'metrics/{label}/{category}/{m}' for m in metric_names
                    ])
                    evaluators.append(
                        Evaluator(label=f'{label}/{category}',
                                  dataloader=dataloaders[category],
                                  metric_names=metric_names),)
            else:
                logger_keys.extend(
                    [f'metrics/{label}/{m}' for m in metric_names])
                evaluators.append(
                    Evaluator(label=label,
                              dataloader=dataloaders,
                              metric_names=metric_names,
                              subset_num_batches=icl_subset_num_batches))

    return evaluators, logger_keys
