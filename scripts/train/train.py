# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0
import copy
import os
import sys
import warnings
from typing import Any, Dict, List, Optional, Union

import torch
from composer import Trainer
from composer.core import Evaluator
from composer.utils import dist, get_device, reproducibility
from omegaconf import DictConfig, ListConfig
from omegaconf import OmegaConf as om
from transformers import PreTrainedTokenizerBase

from llmfoundry import (COMPOSER_MODEL_REGISTRY, ComposerHFCausalLM,
                        MPTForCausalLM, build_finetuning_dataloader,
                        build_text_denoising_dataloader)
from llmfoundry.data.text_data import build_text_dataloader
from llmfoundry.utils.builders import (build_algorithm, build_callback,
                                       build_icl_evaluators, build_logger,
                                       build_optimizer, build_scheduler,
                                       build_tokenizer)
from llmfoundry.utils.config_utils import (log_config, pop_config,
                                           process_init_device,
                                           update_batch_size_info)


def validate_config(cfg: DictConfig):
    """Validates compatible model and dataloader selection."""
    loaders = [cfg.train_loader]
    if 'eval_loader' in cfg:
        loaders.append(cfg.eval_loader)
    for loader in loaders:
        if loader.name == 'text':
            if cfg.model.name in ['hf_prefix_lm', 'hf_t5']:
                raise ValueError(
                    f'Model type "{cfg.model.name}" is not supported when using the "text " ' +\
                    f'dataloader. Please use the "text_denoising" dataloader to pre-train that model type.')
        elif loader.name == 'text_denoising':
            if cfg.model.name == 'hf_causal_lm':
                raise ValueError(
                    f'Model type "{cfg.model.name}" is not supported when using the "text_denoising" ' +\
                    f'dataloader. Please use the "text" dataloader to pre-train that model type.')
            if loader.mixture_of_denoisers.decoder_only_format and cfg.model.name == 'hf_t5':
                warnings.warn(
                    'Model type "hf_t5" requires `decoder_only_format` to be ``False``. ' +\
                    'Overriding `decoder_only_format` from ``True`` to ``False``.')
                loader.mixture_of_denoisers.decoder_only_format = False
            if (not loader.mixture_of_denoisers.decoder_only_format
               ) and cfg.model.name == 'hf_prefix_lm':
                warnings.warn(
                    'Model type "hf_prefix_lm" requires `decoder_only_format` to be ``True``. ' +\
                    'Overriding `decoder_only_format` from ``False`` to ``True``.')
                loader.mixture_of_denoisers.decoder_only_format = True

    if 'icl_tasks' in cfg:
        if cfg.model.name == 'hf_t5':
            raise ValueError(
                'ICL evaluation does not currently support Encoder-Decoder models, such as "hf_t5".'
            )

    if (cfg.model.get('fc_type', 'torch') != 'te' and 'te' not in cfg.model.get(
            'ffn_config', {}).get('ffn_type', 'mptmlp') and
            'fp8' in cfg.precision):
        warnings.warn(
            "fp8 only supported for te.Linear layers. Either set `cfg.model.fc_typ='te'` or "
            +
            "`cfg.model.ffn_config.ffn_type='te_ln_mlp'` to enable layers using fp8 precision."
        )

    if (cfg.model.get('fc_type', 'torch') == 'te' or
            'te' in cfg.model.get('ffn_config', {}).get('ffn_type', 'mptmlp')):
        fsdp_config = cfg.get('fsdp_config', None)
        act_ckpt = fsdp_config.get('activation_checkpointing', False)
        act_ckpt_reentrant = fsdp_config.get(
            'activation_checkpointing_reentrant', True)
        if fsdp_config is not None and act_ckpt == True and act_ckpt_reentrant == False:
            warnings.warn(
                '`te.Linear` layers do not support activation_checkpointing with '
                + '`activation_checkpointing_reentrant = False`. ' +
                'Setting cfg.fsdp_config.activation_checkpointing_reentrant=True.'
            )
            cfg.fsdp_config.activation_checkpointing_reentrant = True

    if 'te' in cfg.model.get('ffn_config', {}).get('ffn_type', 'mptmlp'):
        warnings.warn(
            '`te.LayerNormMLP` requires has issues with torch._dynamo. ' +
            'Setting `torch._dynamo.config.suppress_errors = True` and falling back to eager.'
        )
        torch._dynamo.config.suppress_errors = True  # type: ignore


def build_composer_model(model_cfg: DictConfig,
                         tokenizer: PreTrainedTokenizerBase):
    warnings.filterwarnings(
        action='ignore',
        message='Torchmetrics v0.9 introduced a new argument class property')
    if model_cfg.name not in COMPOSER_MODEL_REGISTRY:
        raise ValueError(
            f'Not sure how to build model with name={model_cfg.name}')
    return COMPOSER_MODEL_REGISTRY[model_cfg.name](model_cfg, tokenizer)


def build_composer_peft_model(
        pretrained_model_name_or_path: str, lora_args: Dict[str, Any],
        tokenizer: PreTrainedTokenizerBase) -> ComposerHFCausalLM:
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as e:
        raise ImportError(
            'Error importing from peft. Please verify that peft and peft utils '
            +
            'are installed by running `pip install -e .[peft]` from `llm-foundry/`. '
            + f'Error encountered: {e}')

    # 1) loads a hf model, 2) adds peft modules, 3) wraps it in a ComposerHFCausalLM.
    print('Building Lora config...')
    lora_cfg = LoraConfig(**lora_args)

    print('Building model from HuggingFace checkpoint...')
    model = MPTForCausalLM.from_pretrained(pretrained_model_name_or_path,
                                           trust_remote_code=True)
    print('Model built!')

    print('Adding Lora modules...')
    model = get_peft_model(model, lora_cfg)
    print('Lora modules added!')

    model = ComposerHFCausalLM(model, tokenizer)

    return model


def print_trainable_parameters(model: torch.nn.Module) -> None:
    # Prints the number of trainable parameters in the model.
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f'trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}'
    )


def build_dataloader(cfg: DictConfig, tokenizer: PreTrainedTokenizerBase,
                     device_batch_size: int):
    if cfg.name == 'text':
        return build_text_dataloader(
            cfg,
            tokenizer,
            device_batch_size,
        )
    elif cfg.name == 'text_denoising':
        return build_text_denoising_dataloader(
            cfg,
            tokenizer,
            device_batch_size,
        )
    elif cfg.name == 'finetuning':
        return build_finetuning_dataloader(
            cfg,
            tokenizer,
            device_batch_size,
        )
    else:
        raise ValueError(f'Not sure how to build dataloader with config: {cfg}')


def main(cfg: DictConfig):
    # Filter deprecation warning from torch internal usage
    warnings.filterwarnings(
        action='ignore',
        category=UserWarning,
        message=
        'torch.distributed.*_base is a private function and will be deprecated.*'
    )

    # Check for incompatibilities between the model and data loaders
    validate_config(cfg)

    # Resolve all interpolation variables as early as possible
    om.resolve(cfg)

    # Create copy of config for logging
    logged_cfg: DictConfig = copy.deepcopy(cfg)

    # Get max split size mb
    max_split_size_mb: Optional[int] = cfg.pop('max_split_size_mb', None)
    if max_split_size_mb is not None:
        os.environ[
            'PYTORCH_CUDA_ALLOC_CONF'] = f'max_split_size_mb:{max_split_size_mb}'

    # Set seed first
    seed: int = pop_config(cfg, 'seed', must_exist=True)
    reproducibility.seed_all(seed)

    # Initialize pytorch distributed training process groups
    dist_timeout: Union[int, float] = pop_config(cfg,
                                                 'dist_timeout',
                                                 must_exist=False,
                                                 default_value=600.0)
    dist.initialize_dist(get_device(None), timeout=dist_timeout)

    # Get global and device batch size information from distributed/single node setting
    cfg = update_batch_size_info(cfg)
    logged_cfg.update(cfg, merge=True)

    # Mandatory model training configs
    model_config: DictConfig = pop_config(cfg, 'model', must_exist=True)
    tokenizer_config: DictConfig = pop_config(cfg, 'tokenizer', must_exist=True)
    optimizer_config: Dict[str, Any] = pop_config(cfg,
                                                  'optimizer',
                                                  must_exist=True,
                                                  convert=True)
    scheduler_config: Dict[str, Any] = pop_config(cfg,
                                                  'scheduler',
                                                  must_exist=True,
                                                  convert=True)
    train_loader_config: DictConfig = pop_config(cfg,
                                                 'train_loader',
                                                 must_exist=True)

    # Optional fsdp data, fine-tuning, and eval configs
    fsdp_config: Optional[Dict[str, Any]] = pop_config(cfg,
                                                       'fsdp_config',
                                                       must_exist=False,
                                                       default_value=None,
                                                       convert=True)
    lora_config: Optional[Dict[str, Any]] = pop_config(cfg,
                                                       'lora',
                                                       must_exist=False,
                                                       default_value=None,
                                                       convert=True)
    eval_loader_config: Optional[DictConfig] = pop_config(cfg,
                                                          'eval_loader',
                                                          must_exist=False,
                                                          default_value=None)
    icl_tasks_config: Optional[ListConfig] = pop_config(cfg,
                                                        'icl_tasks',
                                                        must_exist=False,
                                                        default_value=None)

    # Optional logging, evaluation and callback configs
    logger_configs: Optional[DictConfig] = pop_config(cfg,
                                                      'loggers',
                                                      must_exist=False,
                                                      default_value=None)
    callback_configs: Optional[DictConfig] = pop_config(cfg,
                                                        'callbacks',
                                                        must_exist=False,
                                                        default_value=None)
    algorithm_configs: Optional[DictConfig] = pop_config(cfg,
                                                         'algorithms',
                                                         must_exist=False,
                                                         default_value=None)

    # Mandatory hyperparameters for training
    device_train_batch_size: int = pop_config(cfg,
                                              'device_train_batch_size',
                                              must_exist=True)
    device_eval_batch_size: int = pop_config(cfg,
                                             'device_eval_batch_size',
                                             must_exist=True)
    max_duration: Union[int, str] = pop_config(cfg,
                                               'max_duration',
                                               must_exist=True)
    eval_interval: Union[int, str] = pop_config(cfg,
                                                'eval_interval',
                                                must_exist=True)
    precision: str = pop_config(cfg, 'precision', must_exist=True)
    max_seq_len: int = pop_config(cfg, 'max_seq_len', must_exist=True)

    # Optional parameters will be set to default values if not specified.
    default_run_name: str = os.environ.get('RUN_NAME', 'llm')
    run_name: str = pop_config(cfg,
                               'run_name',
                               must_exist=False,
                               default_value=default_run_name)
    save_folder: Optional[str] = pop_config(cfg,
                                            'save_folder',
                                            must_exist=False,
                                            default_value=None)
    save_latest_filename: str = pop_config(cfg,
                                           'save_latest_filename',
                                           must_exist=False,
                                           default_value='latest-rank{rank}.pt')
    save_overwrite: bool = pop_config(cfg,
                                      'save_overwrite',
                                      must_exist=False,
                                      default_value=False)
    save_weights_only: bool = pop_config(cfg,
                                         'save_weights_only',
                                         must_exist=False,
                                         default_value=False)
    save_filename: str = pop_config(
        cfg,
        'save_filename',
        must_exist=False,
        default_value='ep{epoch}-ba{batch}-rank{rank}.pt')
    save_interval: Union[str, int] = pop_config(cfg,
                                                'save_interval',
                                                must_exist=False,
                                                default_value='1000ba')
    save_num_checkpoints_to_keep: int = pop_config(
        cfg, 'save_num_checkpoints_to_keep', must_exist=False, default_value=-1)
    progress_bar = pop_config(cfg,
                              'progress_bar',
                              must_exist=False,
                              default_value=False)
    log_to_console: bool = pop_config(cfg,
                                      'log_to_console',
                                      must_exist=False,
                                      default_value=True)
    python_log_level: str = pop_config(cfg,
                                       'python_log_level',
                                       must_exist=False,
                                       default_value='debug')
    console_log_interval: Union[int, str] = pop_config(cfg,
                                                       'console_log_interval',
                                                       must_exist=False,
                                                       default_value='1ba')
    device_train_microbatch_size: Union[str, int] = pop_config(
        cfg,
        'device_train_microbatch_size',
        must_exist=False,
        default_value='auto')
    eval_subset_num_batches: int = pop_config(cfg,
                                              'eval_subset_num_batches',
                                              must_exist=False,
                                              default_value=-1)
    eval_first: bool = pop_config(cfg,
                                  'eval_first',
                                  must_exist=False,
                                  default_value=False)
    load_path: str = pop_config(cfg,
                                'load_path',
                                must_exist=False,
                                default_value=None)
    load_weights_only: bool = pop_config(cfg,
                                         'load_weights_only',
                                         must_exist=False,
                                         default_value=False)
    load_ignore_keys: Optional[List[str]] = pop_config(cfg,
                                                       'load_ignore_keys',
                                                       must_exist=False,
                                                       default_value=None)
    # Enable autoresume from model checkpoints if possible
    autoresume_default: bool = False
    if logged_cfg.get('run_name', None) is not None \
        and save_folder is not None \
        and not save_overwrite \
        and not save_weights_only:
        print('As run_name, save_folder, and save_latest_filename are set, \
                changing autoresume default to True...')
        autoresume_default = True
    autoresume: bool = pop_config(cfg,
                                  'autoresume',
                                  must_exist=False,
                                  default_value=autoresume_default)

    # Pop known unused parameters that are used as interpolation variables or
    # created by update_batch_size_info.
    pop_config(cfg, 'data_local', must_exist=False)
    pop_config(cfg, 'data_remote', must_exist=False)
    pop_config(cfg, 'global_seed', must_exist=False)
    pop_config(cfg, 'global_train_batch_size', must_exist=False)
    pop_config(cfg, 'n_gpus', must_exist=False)
    pop_config(cfg, 'device_train_grad_accum', must_exist=False)

    # Warn users for unused parameters
    for key in cfg:
        warnings.warn(
            f'Unused parameter {key} found in cfg. Please check your yaml to ensure this parameter is necessary.'
        )

    # Warn if fsdp is enabled but user only has 1 GPU
    if dist.get_world_size() == 1 and fsdp_config is not None:
        warnings.warn(
            'FSDP is not applicable for single-GPU training. Reverting to DDP.')
        fsdp_config = None

    # Initialize context
    init_context = process_init_device(model_config, fsdp_config)
    logged_cfg.update({'fsdp_config': fsdp_config}, merge=True)

    # Build tokenizer
    tokenizer = build_tokenizer(tokenizer_config)

    # Build Model
    print('Initializing model...')
    with init_context:
        if lora_config is not None:  # frozen model + trainable lora modules
            model: ComposerHFCausalLM = build_composer_peft_model(
                model_config.pretrained_model_name_or_path, lora_config['args'],
                tokenizer)
            print_trainable_parameters(model)  # should not be 100%
        else:  # standard model
            model = build_composer_model(model_config, tokenizer)

    # Log number of parameters
    n_params = sum(p.numel() for p in model.parameters())
    logged_cfg.update({'n_params': n_params})

    # Optimizer
    optimizer_name: str = optimizer_config.pop('name')
    optimizer = build_optimizer(model, optimizer_name, optimizer_config)

    # Scheduler
    scheduler_name: str = scheduler_config.pop('name')
    scheduler = build_scheduler(scheduler_name, scheduler_config)

    # Loggers
    loggers = [
        build_logger(str(name), logger_cfg)
        for name, logger_cfg in logger_configs.items()
    ] if logger_configs else None

    # Callbacks
    callbacks = [
        build_callback(str(name), callback_cfg)
        for name, callback_cfg in callback_configs.items()
    ] if callback_configs else None

    # Algorithms
    algorithms = [
        build_algorithm(str(name), algorithm_cfg)
        for name, algorithm_cfg in algorithm_configs.items()
    ] if algorithm_configs else None

    # Dataloaders
    print('Building train loader...')
    train_loader = build_dataloader(
        train_loader_config,
        tokenizer,
        device_train_batch_size,
    )

    ## Evaluation
    print('Building eval loader...')
    evaluators = []
    if eval_loader_config is not None:
        assert model.train_metrics is not None
        eval_dataloader = build_dataloader(eval_loader_config, tokenizer,
                                           device_eval_batch_size)
        eval_metric_names = list(model.train_metrics.keys())
        eval_loader = Evaluator(label='eval',
                                dataloader=eval_dataloader,
                                metric_names=eval_metric_names)
        evaluators.append(eval_loader)

    if icl_tasks_config is not None:
        icl_evaluators, _ = build_icl_evaluators(icl_tasks_config, tokenizer,
                                                 max_seq_len,
                                                 device_eval_batch_size)
        evaluators.extend(icl_evaluators)

    # Build the Trainer
    print('Building trainer...')
    trainer = Trainer(
        run_name=run_name,
        seed=seed,
        model=model,
        train_dataloader=train_loader,
        eval_dataloader=evaluators,
        optimizers=optimizer,
        schedulers=scheduler,
        max_duration=max_duration,
        eval_interval=eval_interval,
        eval_subset_num_batches=eval_subset_num_batches,
        progress_bar=progress_bar,
        log_to_console=log_to_console,
        console_log_interval=console_log_interval,
        loggers=loggers,
        callbacks=callbacks,
        precision=precision,
        algorithms=algorithms,
        device_train_microbatch_size=device_train_microbatch_size,
        fsdp_config=fsdp_config,  # type: ignore
        save_folder=save_folder,
        save_filename=save_filename,
        save_latest_filename=save_latest_filename,
        save_interval=save_interval,
        save_num_checkpoints_to_keep=save_num_checkpoints_to_keep,
        save_overwrite=save_overwrite,
        save_weights_only=save_weights_only,
        load_path=load_path,
        load_weights_only=load_weights_only,
        load_ignore_keys=load_ignore_keys,
        autoresume=autoresume,
        python_log_level=python_log_level,
        dist_timeout=dist_timeout,
    )

    print('Logging config')
    log_config(logged_cfg)
    torch.cuda.empty_cache()

    # Eval first if requested
    if eval_first and trainer.state.timestamp.batch.value == 0:
        trainer.eval()

    print('Starting training...')
    trainer.fit()

    print('Done.')


if __name__ == '__main__':
    yaml_path, args_list = sys.argv[1], sys.argv[2:]
    with open(yaml_path) as f:
        yaml_cfg = om.load(f)
    cli_cfg = om.from_cli(args_list)
    cfg = om.merge(yaml_cfg, cli_cfg)
    assert isinstance(cfg, DictConfig)
    main(cfg)
