# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import contextlib
import functools
import logging
import os
import re
import warnings
from collections import OrderedDict
from typing import (Any, ContextManager, Dict, Iterable, List, Optional, Tuple,
                    Union)

import torch
from composer.core import Algorithm, Callback, Evaluator
from composer.loggers import LoggerDestination
from composer.models import ComposerModel
from composer.optim.scheduler import ComposerScheduler
from composer.utils import dist
from omegaconf import OmegaConf as om
from torch.optim.optimizer import Optimizer
from torchmetrics import Metric
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from llmfoundry import registry
from llmfoundry.callbacks import EvalGauntlet
from llmfoundry.data.dataloader import build_dataloader
from llmfoundry.eval.datasets.in_context_learning_evaluation import \
    get_icl_task_dataloader
from llmfoundry.tokenizers.tiktoken import TiktokenTokenizerWrapper
from llmfoundry.utils.config_utils import to_str_dict
from llmfoundry.utils.registry_utils import construct_from_registry
from llmfoundry.utils.warnings import VersionedDeprecationWarning

log = logging.getLogger(__name__)

__all__ = [
    'build_algorithm',
    'build_callback',
    'build_evaluators',
    'build_icl_data_and_gauntlet',
    'build_icl_evaluators',
    'build_logger',
    'build_optimizer',
    'build_scheduler',
    'build_tokenizer',
    'build_composer_model',
    'build_metric',
]


def build_evaluators(
    eval_loader_config: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]],
    icl_tasks_config: Optional[Union[str, List[Dict[str, Any]]]],
    eval_gauntlet_config: Optional[Union[str, Dict[str, Any]]],
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
    eval_loader_config: Union[Dict[str, Any], List[Dict[str, Any]]],
    tokenizer: PreTrainedTokenizerBase,
    device_eval_batch_size: int,
) -> List[Evaluator]:
    evaluators: List[Evaluator] = []
    if isinstance(eval_loader_config, list):
        eval_configs = eval_loader_config
        is_multi_eval = True
    else:
        eval_configs = [eval_loader_config]
        is_multi_eval = False

    for eval_config in eval_configs:
        label = None
        if 'label' in eval_config:
            label = eval_config.pop('label')
        eval_dataloader = build_dataloader(eval_config, tokenizer,
                                           device_eval_batch_size)
        eval_loader: Evaluator = Evaluator(
            label=f'eval/{label}' if is_multi_eval else 'eval',
            dataloader=eval_dataloader,
            # Load the eval data to fail fast. metrics will get added
            # later in add_metrics_to_eval_loaders, after the model is loaded
            metric_names=[],
            device_eval_microbatch_size=device_eval_batch_size,
        )
        evaluators.append(eval_loader)
    return evaluators


def add_metrics_to_eval_loaders(
    evaluators: List[Evaluator],
    metric_names: List[str],
) -> List[Evaluator]:
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
    icl_tasks_config: Union[str, List[Dict[str, Any]]],
    eval_gauntlet_config: Optional[Union[str, Dict[str, Any]]],
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
                assert isinstance(eval_gauntlet_cfg, dict)
            eval_gauntlet = to_str_dict(eval_gauntlet_cfg['eval_gauntlet'])
        elif isinstance(eval_gauntlet_config, dict):  # pyright: ignore
            eval_gauntlet = eval_gauntlet_config
        else:
            raise ValueError(
                f'Got invalid type for eval_gauntlet_config: {type(eval_gauntlet_config)}'
            )
        eval_gauntlet['logger_keys'] = logger_keys
        eval_gauntlet['benchmark_sizes'] = {
            e.label: e.dataloader.num_samples for e in icl_evaluators
        }
        eval_gauntlet_cb = EvalGauntlet(**eval_gauntlet)
    return icl_evaluators, logger_keys, eval_gauntlet_cb


def build_composer_model(
    name: str,
    cfg: Dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    init_context: Optional[ContextManager] = None,
    master_weights_dtype: Optional[str] = None,
) -> ComposerModel:
    """Builds a ComposerModel from the registry.

    Args:
        name (str): Name of the model to build.
        cfg (DictConfig): Configuration for the model.
        tokenizer (PreTrainedTokenizerBase): Tokenizer to use.
        init_context (Optional[ContextManager], optional): Context manager to use for initialization. Defaults to None.
        master_weights_dtype (Optional[str], optional): Master weights dtype. Defaults to None.

    Returns:
        ComposerModel: _description_
    """
    if init_context is None:
        init_context = contextlib.nullcontext()

    with init_context:
        model = construct_from_registry(
            name=name,
            registry=registry.models,
            pre_validation_function=ComposerModel,
            post_validation_function=None,
            kwargs={
                **cfg, 'tokenizer': tokenizer
            },
        )

    str_dtype_to_torch_dtype = {
        'f16': torch.float16,
        'float16': torch.float16,
        'bf16': torch.bfloat16,
        'bfloat16': torch.bfloat16,
    }

    if master_weights_dtype is not None:
        if master_weights_dtype not in str_dtype_to_torch_dtype:
            raise ValueError(
                f'Invalid master_weights_dtype: {master_weights_dtype}. ' +
                f'Valid options are: {list(str_dtype_to_torch_dtype.keys())}.')
        dtype = str_dtype_to_torch_dtype[master_weights_dtype]
        model = model.to(dtype=dtype)

    return model


def build_callback(
    name: str,
    kwargs: Optional[Dict[str, Any]] = None,
    config: Any = None,
) -> Callback:
    """Builds a callback from the registry."""
    registry_to_use = registry.callbacks
    if name in registry.callbacks_with_config:
        if kwargs is None:
            kwargs = {}
        if 'config' in kwargs:
            raise ValueError(
                f'`config` is a reserved keyword for callbacks with config. Please remove it from the kwargs.'
            )
        kwargs['config'] = config
        registry_to_use = registry.callbacks_with_config

    return construct_from_registry(name=name,
                                   registry=registry_to_use,
                                   partial_function=True,
                                   pre_validation_function=Callback,
                                   post_validation_function=None,
                                   kwargs=kwargs)


def build_logger(name: str,
                 kwargs: Optional[Dict[str, Any]] = None) -> LoggerDestination:
    """Builds a logger from the registry."""
    return construct_from_registry(name=name,
                                   registry=registry.loggers,
                                   partial_function=True,
                                   pre_validation_function=LoggerDestination,
                                   post_validation_function=None,
                                   kwargs=kwargs)


def build_algorithm(name: str,
                    kwargs: Optional[Dict[str, Any]] = None) -> Algorithm:
    """Builds an algorithm from the registry."""
    return construct_from_registry(name=name,
                                   registry=registry.algorithms,
                                   partial_function=True,
                                   pre_validation_function=Algorithm,
                                   post_validation_function=None,
                                   kwargs=kwargs)


def build_metric(name: str, kwargs: Optional[Dict[str, Any]] = None) -> Metric:
    """Builds a metric from the registry."""
    return construct_from_registry(name=name,
                                   registry=registry.metrics,
                                   partial_function=True,
                                   pre_validation_function=Metric,
                                   post_validation_function=None,
                                   kwargs=kwargs)


def _extract_param_groups(
    model: torch.nn.Module,
    optimizer_config: Optional[Dict[str, Any]] = None,
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
    if optimizer_config is None:
        return model.parameters()

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
    kwargs = {**optimizer_config}

    if 'params' in kwargs:
        raise ValueError(
            'The `params` will be automatically extracted from the model and ' +
            'optimizer config. Please remove it from the optimizer config kwargs.'
        )

    kwargs['params'] = params
    return construct_from_registry(name=name,
                                   registry=registry.optimizers,
                                   partial_function=True,
                                   pre_validation_function=Optimizer,
                                   post_validation_function=None,
                                   kwargs=kwargs)


def build_scheduler(
        name: str,
        scheduler_config: Optional[Dict[str, Any]] = None) -> ComposerScheduler:
    return construct_from_registry(
        name=name,
        registry=registry.schedulers,
        partial_function=True,
        pre_validation_function=ComposerScheduler,
        post_validation_function=None,
        kwargs=scheduler_config,
    )


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

    if not hasattr(
            tokenizer, 'eos_token'
    ) or tokenizer.eos_token is None:  # type: ignore (sometime's it's not none but that's ok too)
        raise ValueError(
            f'The tokenizer {tokenizer_name} must have an eos_token.')

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
    icl_tasks: Union[str, List[Dict[str, Any]]],
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
        icl_tasks_list = to_str_dict(icl_task_cfg.icl_tasks)
    else:
        icl_tasks_list = icl_tasks

    def _validate_cfg(icl_cfg: Dict[str, Any]):
        assert 'label' in icl_cfg
        assert 'dataset_uri' in icl_cfg and icl_cfg['dataset_uri'] is not None
        assert 'icl_task_type' in icl_cfg
        assert 'num_fewshot' in icl_cfg

        if 'metric_names' not in icl_cfg:
            if icl_cfg['icl_task_type'] == 'language_modeling':
                icl_cfg['metric_names'] = ['InContextLearningLMAccuracy']
            elif icl_cfg['icl_task_type'] == 'multiple_choice':
                icl_cfg['metric_names'] = [
                    'InContextLearningMultipleChoiceAccuracy'
                ]
            elif icl_cfg['icl_task_type'] == 'schema':
                icl_cfg['metric_names'] = [
                    'InContextLearningMultipleChoiceAccuracy'
                ]
            elif icl_cfg[
                    'icl_task_type'] == 'generation_task_with_answers' or icl_cfg[
                        'icl_task_type'] == 'question_answering':
                if icl_cfg['icl_task_type'] == 'question_answering':
                    warnings.warn(
                        VersionedDeprecationWarning(
                            "ICL task type 'question_answering' is now deprecated. Use identifier 'generation_task_with_answers'",
                            'v0.9.0'))
                icl_cfg['metric_names'] = [
                    'InContextLearningGenerationExactMatchAccuracy'
                ]
            elif icl_cfg['icl_task_type'] == 'code_evaluation':
                icl_cfg['metric_names'] = ['InContextLearningCodeEvalAccuracy']
            else:
                raise ValueError(
                    f'No metric_names defined, unable to build default metrics for icl_task_type={icl_cfg["icl_task_type"]}.'
                )

        if 'prompt_string' not in icl_cfg:
            icl_cfg['prompt_string'] = ''
        if 'example_delimiter' not in icl_cfg:
            icl_cfg['example_delimiter'] = '\n'
        if 'continuation_delimiter' not in icl_cfg:
            icl_cfg['continuation_delimiter'] = ' '
        if 'max_seq_len' not in icl_cfg:
            icl_cfg['max_seq_len'] = default_max_seq_len
        if 'batch_size' not in icl_cfg:
            icl_cfg['batch_size'] = default_batch_size
        if 'pass_at_k' not in icl_cfg:
            icl_cfg['pass_at_k'] = 1
        if 'fewshot_random_seed' not in icl_cfg:
            icl_cfg['fewshot_random_seed'] = 1234
        if 'generations_per_sample' not in icl_cfg:
            icl_cfg['generations_per_sample'] = 1

        if 'num_beams' in icl_cfg:
            raise ValueError(
                'num_beams is no longer supported as a top level icl_task parameter.'  + \
                'Please use generation_kwargs.num_beams instead.')

    for icl_cfg in icl_tasks_list:
        assert isinstance(
            icl_cfg, dict), f'Expected dict, got {type(icl_cfg)}, {icl_cfg=}'
        _validate_cfg(icl_cfg)
        for num_fewshot in list(icl_cfg['num_fewshot']):
            if tokenizer.pad_token_id is None:
                # Current workaround to support GPT2 tokenizer with `pad_token_id = None`
                pad_tok_id = tokenizer.eos_token_id
            else:
                pad_tok_id = tokenizer.pad_token_id
            label = f'{icl_cfg["label"]}/{num_fewshot}-shot'
            metric_names = list(icl_cfg['metric_names'])
            # TODO: fix Composer bug when copying local paths and destination exists
            destination_path = f'{destination_dir}/{icl_cfg["label"]}-{num_fewshot}.jsonl'
            if dist.get_local_rank() == 0 and os.path.exists(destination_path):
                os.remove(destination_path)
            dist.barrier()

            hf_parsing_map = icl_cfg.get('hf_parsing_map', {})
            hf_loading_vars = icl_cfg.get('hf_loading_vars', {})

            early_stopping_criteria = icl_cfg.get('early_stopping_criteria',
                                                  None)
            assert early_stopping_criteria is None or isinstance(
                early_stopping_criteria, list)
            dataloaders = get_icl_task_dataloader(
                icl_cfg['icl_task_type'],
                icl_cfg['dataset_uri'],
                tokenizer,
                batch_size=icl_cfg['batch_size'],
                max_seq_len=icl_cfg['max_seq_len'],
                pad_tok_id=pad_tok_id,
                num_fewshot=num_fewshot,
                prompt_string=icl_cfg['prompt_string'],
                example_delimiter=icl_cfg['example_delimiter'],
                hf_loading_vars=hf_loading_vars,
                hf_parsing_map=hf_parsing_map,
                continuation_delimiter=icl_cfg['continuation_delimiter'],
                question_prelimiter=icl_cfg.get('question_prelimiter', ''),
                destination_path=destination_path,
                fewshot_random_seed=icl_cfg['fewshot_random_seed'],
                pass_at_k=icl_cfg['pass_at_k'],
                generations_per_sample=icl_cfg['generations_per_sample'],
                has_categories=icl_cfg.get('has_categories', False),
                cot_delimiter=icl_cfg.get('cot_delimiter', ''),
                generation_kwargs=icl_cfg.get('generation_kwargs', {}),
                early_stopping_criteria=early_stopping_criteria,
                do_normalization=icl_cfg.get('do_normalization', True))
            if 'has_categories' in icl_cfg and icl_cfg[
                    'has_categories'] and isinstance(dataloaders, dict):
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
