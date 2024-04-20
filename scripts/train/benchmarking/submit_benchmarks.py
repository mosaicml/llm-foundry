# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0
import argparse
import math
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import requests
import yaml

from mcli import RunConfig, SchedulingConfig, create_run, get_clusters


def _get_cluster_info():
    clusters = get_clusters()
    cluster_info = {}

    for cluster in clusters:
        cluster_info[cluster.name] = [(ci.gpu_type, max(ci.gpu_nums))
                                      for ci in cluster.cluster_instances
                                      if ci.gpu_type is not None]

    return cluster_info


CLUSTER_INFO = _get_cluster_info()


def str_to_bool(value: Union[bool, str]):
    # helper fn
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')


def parse_args():
    parser = argparse.ArgumentParser(
        description=
        'Generate and run configurations to test MPT training throughput on the MosaicML platform.'
    )

    parser.add_argument('--project', type=str, default='tput')
    parser.add_argument(
        '--image',
        type=str,
        default='mosaicml/pytorch:1.13.1_cu117-python3.10-ubuntu20.04')
    parser.add_argument('--git_branch',
                        type=str,
                        default=None,
                        help='what git branch to use.')
    parser.add_argument('--git_commit',
                        type=str,
                        default=None,
                        help='what git commit to use.')
    parser.add_argument('-t',
                        '--precisions',
                        '--types',
                        type=str,
                        default=['bf16'],
                        nargs='+',
                        choices=['bf16', 'fp16', 'fp8'])
    parser.add_argument('--fsdp_config_mixed_precision',
                        type=str,
                        default='PURE')
    parser.add_argument('--fsdp_config_activation_checkpointing',
                        type=str_to_bool,
                        nargs='?',
                        const=True,
                        default=None)
    parser.add_argument('--fsdp_config_shard_strategy',
                        type=str,
                        nargs='?',
                        const=True,
                        default=None)
    parser.add_argument('--fsdp_config_limit_all_gathers',
                        type=str_to_bool,
                        nargs='?',
                        const=True,
                        default=None)
    parser.add_argument('--fsdp_config_forward_prefetch',
                        type=str_to_bool,
                        nargs='?',
                        const=True,
                        default=None)
    parser.add_argument('--fsdp_config_backward_prefetch',
                        type=str,
                        nargs='?',
                        const=True,
                        default=None)
    parser.add_argument('--activation_cpu_offload',
                        type=str_to_bool,
                        nargs='?',
                        const=True,
                        default=None)
    parser.add_argument(
        '-s',
        '--seq_len_exp',
        type=int,
        default=[11, 11],
        nargs=2,
        help='exponent of seq lengths to be tested (default: [11, 11] = 2048)')
    parser.add_argument(
        '-b',
        '--batch_size_exp',
        type=int,
        default=None,
        nargs=2,
        help=
        'exponent of batch size (in tokens) to be tested (default: [19, 23] = 2^19 to 2^23)'
    )
    parser.add_argument(
        '--batch_sizes',
        type=int,
        nargs='+',
        default=[],
        help='batch sizes to run.',
    )
    parser.add_argument(
        '--accum',
        type=int,
        default=None,
        help='batch sizes multiplier (accumulations before step).',
    )
    parser.add_argument('-m',
                        '--model_yamls',
                        type=str,
                        default=[
                            '125m.yaml', '350m.yaml', '760m.yaml', '1b.yaml',
                            '3b.yaml', '7b.yaml', '13b.yaml', '30b.yaml',
                            '70b.yaml'
                        ],
                        choices=[
                            '125m.yaml', '350m.yaml', '760m.yaml', '1b.yaml',
                            '3b.yaml', '7b.yaml', '13b.yaml', '30b.yaml',
                            '70b.yaml'
                        ],
                        nargs='+',
                        help='model sizes to test')

    parser.add_argument('--attn_impl', type=str, default='flash')

    parser.add_argument('-c',
                        '--clusters',
                        type=str,
                        default=['r1z1'],
                        nargs='+',
                        choices=CLUSTER_INFO.keys())
    known_args = parser.parse_known_args()[0]
    _gpu_types = get_gpu_types(known_args.clusters)
    parser.add_argument('--gpu_types',
                        type=str,
                        default=['a100_40gb'],
                        nargs='+',
                        choices=_gpu_types)
    known_args = parser.parse_known_args()[0]
    _gpu_nums = get_gpu_nums(known_args.clusters, known_args.gpu_types)
    parser.add_argument('-g',
                        '--gpu_nums',
                        type=int,
                        default=[8],
                        nargs='+',
                        choices=_gpu_nums)

    parser.add_argument('--microbatch_size',
                        type=int,
                        default=None,
                        help='set microbatch_size')

    parser.add_argument('--pad_vocab_multiple', type=int, default=None)

    parser.add_argument('--data_remote',
                        type=str,
                        default=None,
                        help='optional data remote path for streaming data')

    parser.add_argument('--wandb',
                        type=str_to_bool,
                        nargs='?',
                        const=True,
                        default=True)

    parser.add_argument('--priority', type=str, default='lowest')

    parser.add_argument('--RUN',
                        type=str_to_bool,
                        nargs='?',
                        const=True,
                        default=False)
    return parser.parse_args()


def get_max_seq_lens(pows: Optional[List[int]] = None):
    if pows is None:
        pows = [9, 14]
    return [2**n for n in range(pows[0], pows[1] + 1)]


def get_global_train_batch_sizes(max_seq_len: int,
                                 pows: List[int],
                                 batch_sizes: Optional[List[int]] = None):
    if batch_sizes is None:
        batch_sizes = []
    if pows:
        # global batch size in tokens (default: .5M thru 8M)
        global_train_token_counts = [2**n for n in range(pows[0], pows[1] + 1)]
        batch_sizes += [t // max_seq_len for t in global_train_token_counts
                       ]  # global batch size in samples
    return batch_sizes


def get_parameters(yaml_file: str):
    local_yamls = False if 'https' in yaml_file else True
    if local_yamls:
        # Load the YAML into a parameters dictionary
        with open(yaml_file) as f:
            parameters = yaml.safe_load(f)
    else:
        # Download parameter yaml
        req = requests.get(yaml_file)
        # Load the YAML into a parameters dictionary
        parameters = yaml.safe_load(req.text)

    return parameters


def get_cluster_gpu_types(cluster: str):
    return [gpu_info[0] for gpu_info in CLUSTER_INFO[cluster]]


def get_gpu_types(clusters: List[str]):
    gpu_types = set()
    for c in clusters:
        for g in get_cluster_gpu_types(c):
            gpu_types.add(g)
    return gpu_types


def get_gpu_nums(clusters: List[str], gpu_types: List[str]):
    max_gpus_per_run = 1
    for c in clusters:
        for gpu_info in CLUSTER_INFO[c]:
            if gpu_info[0] in gpu_types:
                max_gpus_per_run = max(max_gpus_per_run, gpu_info[1])

    gpu_nums = [1]
    while gpu_nums[-1] < max_gpus_per_run:
        gpu_nums += [2 * gpu_nums[-1]]

    return gpu_nums


def get_valid_gpu_lim(cluster: str, gpu_type: str):
    for gpu_info in CLUSTER_INFO[cluster]:
        if gpu_info[0] == gpu_type:
            return gpu_info[1]
    raise ValueError


def mod_parameters(
    parameters: Dict[str, Any],
    max_seq_len: int,
    global_train_batch_size: int,
    precision: str,
    fsdp_config_mixed_precision: str = 'DEFAULT',
    fsdp_config_activation_checkpointing: Optional[bool] = None,
    fsdp_config_shard_strategy: Optional[str] = None,
    fsdp_config_forward_prefetch: Optional[bool] = None,
    fsdp_config_backward_prefetch: Optional[str] = None,
    fsdp_config_limit_all_gathers: Optional[bool] = None,
    activation_cpu_offload: Optional[bool] = None,
    run_name: str = '',
    data_remote: Optional[str] = None,
    max_duration: str = '30ba',
    eval_interval: int = 0,
    microbatch_size: Optional[Union[int, str]] = None,
    wandb: bool = True,
    pad_vocab_multiple: Optional[int] = None,
):
    if run_name:
        parameters['run_name'] = run_name
    if data_remote is not None:
        parameters['data_remote'] = data_remote
        parameters['train_loader']['dataset']['remote'] = parameters[
            'data_remote']
        parameters['eval_loader']['dataset']['remote'] = parameters[
            'data_remote']

        parameters['data_local'] = '/tmp/c4'
        parameters['train_loader']['dataset']['local'] = parameters[
            'data_local']
        parameters['eval_loader']['dataset']['local'] = parameters['data_local']
    else:
        parameters['train_loader']['dataset'][
            'split'] = 'train_small'  # for throughput testing purposes
        parameters['eval_loader']['dataset'][
            'split'] = 'val_small'  # for throughput testing purposes
    # set max_seq_len
    parameters['max_seq_len'] = max_seq_len
    parameters['model']['max_seq_len'] = max_seq_len

    parameters['model']['attn_config']['attn_impl'] = args.attn_impl

    parameters['model']['norm_type'] = 'low_precision_layernorm'

    # Pad vocab size to multiple of N for A100 perf
    if pad_vocab_multiple:
        vocab_size = parameters['model']['vocab_size']
        parameters['model']['vocab_size'] = math.ceil(
            vocab_size / pad_vocab_multiple) * pad_vocab_multiple

    parameters['tokenizer']['kwargs']['model_max_length'] = max_seq_len
    parameters['train_loader']['dataset']['max_seq_len'] = max_seq_len
    parameters['eval_loader']['dataset']['max_seq_len'] = max_seq_len

    parameters['global_train_batch_size'] = global_train_batch_size
    if microbatch_size is not None:
        parameters['device_train_microbatch_size'] = microbatch_size

    # update eval batch size based on change in seq len
    parameters['device_eval_batch_size'] = max(
        1,
        int(parameters['device_eval_batch_size'] / ((max_seq_len / 2048)**2)))

    parameters['eval_loader'][
        'eval_subset_num_batches'] = 2  # for throughput testing purposes

    parameters['max_duration'] = max_duration
    parameters['eval_interval'] = eval_interval

    parameters['precision'] = precision
    parameters['fsdp_config']['mixed_precision'] = fsdp_config_mixed_precision
    if fsdp_config_activation_checkpointing is not None:
        parameters['fsdp_config'][
            'activation_checkpointing'] = fsdp_config_activation_checkpointing
    if fsdp_config_shard_strategy is not None:
        parameters['fsdp_config'][
            'sharding_strategy'] = fsdp_config_shard_strategy
    if fsdp_config_limit_all_gathers is not None:
        parameters['fsdp_config'][
            'limit_all_gathers'] = fsdp_config_limit_all_gathers
    if fsdp_config_forward_prefetch is not None:
        parameters['fsdp_config'][
            'forward_prefetch'] = fsdp_config_forward_prefetch
    if fsdp_config_backward_prefetch is not None:
        parameters['fsdp_config'][
            'backward_prefetch'] = fsdp_config_backward_prefetch
    if activation_cpu_offload is not None:
        parameters['fsdp_config'][
            'activation_cpu_offload'] = activation_cpu_offload

    if wandb:
        # add wandb
        parameters['loggers'] = {'wandb': {}}

    return parameters


def get_integrations(project: str,
                     git_branch: Optional[str] = None,
                     git_commit: Optional[str] = None,
                     wandb: bool = True):
    integrations = []

    if git_branch and git_commit:
        raise ValueError(f'{git_branch=} and {git_commit=} cannot both be set!')
    git_integration = {
        k: v for k, v in {
            'git_branch': git_branch,
            'git_commit': git_commit,
        }.items() if v is not None
    }
    git_integration.update({
        'integration_type': 'git_repo',
        'git_repo': 'mosaicml/llm-foundry',
        'pip_install': '.[gpu-flash2]'
    })

    integrations = [git_integration]

    if wandb:
        integrations += [{
            'integration_type': 'wandb',
            'entity': 'mosaic-ml',
            'project': project
        }]

    return integrations


def run_config(config: Tuple[str, int, int, str, str, int, str],
               args: argparse.Namespace):
    model_yaml, max_seq_len, global_train_batch_size, cluster, gpu_type, gpu_num, precision = config
    integrations = [
        {
            'integration_type': 'git_repo',
            'git_repo': 'mosaicml/llm-foundry',
            'git_branch': 'main',
            'pip_install': '.[gpu-flash2]',
        },
        {
            'integration_type': 'wandb',
            'entity': 'mosaic-ml',
            'project': args.project
        },
    ]

    command = ''
    if gpu_type == 'h100_80gb' and 'fp8' in precision:  # Required for flash-attn and FP8 training
        command += f"""
        pip install flash-attn==2.4.2 --no-build-isolation
        pip install git+https://github.com/NVIDIA/TransformerEngine.git@v0.10
        pip uninstall install pydantic --yes
        pip install pydantic==1.9.0
        """

    if args.data_remote is None:
        command += f"""
            cd llm-foundry/scripts
            python data_prep/convert_dataset_hf.py --dataset c4 --data_subset en --out_root ./my-copy-c4 --splits train_small val_small --concat_tokens {max_seq_len} --eos_text '<|endoftext|>'
            composer train/train.py /mnt/config/parameters.yaml
            """
    else:
        command += f"""
            cd llm-foundry/scripts
            composer train/train.py /mnt/config/parameters.yaml
            """

    path = os.path.join('../yamls/pretrain', 'mpt-' + model_yaml)
    parameters = get_parameters(path)

    model_name = '-'.join(model_yaml.split('.')[-2].split('/')[-2:]).replace(
        '_', '-')
    model_name = model_name.split('-')
    if 'mosaic' in model_name:
        model_name.pop(model_name.index('mosaic'))
    model_name = ''.join(model_name)
    name = f"{args.project}-{cluster}-{model_name}-{gpu_num}x{gpu_type}-s{max_seq_len}b{global_train_batch_size}{precision.replace('amp_', '')}".replace(
        '_', '-')

    name_len_lim = 54 - 7
    if len(name) > name_len_lim:
        _name = name
        name = name[:name_len_lim]
        print(f'Shortening {_name} to {name} ({name_len_lim} chars)')
    microbatch_size = args.microbatch_size or 'auto'
    assert isinstance(microbatch_size, (int, str))
    parameters = mod_parameters(
        parameters,
        max_seq_len,
        global_train_batch_size,
        'amp_' + precision,
        fsdp_config_activation_checkpointing=args.
        fsdp_config_activation_checkpointing,
        fsdp_config_limit_all_gathers=args.fsdp_config_limit_all_gathers,
        fsdp_config_shard_strategy=args.fsdp_config_shard_strategy,
        fsdp_config_forward_prefetch=args.fsdp_config_forward_prefetch,
        fsdp_config_backward_prefetch=args.fsdp_config_backward_prefetch,
        activation_cpu_offload=args.activation_cpu_offload,
        run_name=name,
        data_remote=args.data_remote,
        microbatch_size=microbatch_size,
        wandb=args.wandb,
        pad_vocab_multiple=args.pad_vocab_multiple,
    )
    if gpu_type == 'h100_80gb' and precision == 'fp8':
        parameters['model']['fc_type'] = 'te'
    # Create run config mcli sdk/api
    config = RunConfig(name=name,
                       compute={
                           'cluster': cluster,
                           'gpu_type': gpu_type,
                           'gpus': gpu_num
                       },
                       image=args.image,
                       integrations=integrations,
                       command=command,
                       parameters=parameters,
                       scheduling=SchedulingConfig(priority=args.priority,
                                                   resumable=True))
    if args.RUN:
        # Create the run from a config
        run = create_run(config)
        print(f'Launching run {run.name}')
    else:
        print(f'run = {name}')
        print(f'{config=}')


def run_check_capacity(model_yaml: str,
                       gpu_num: int,
                       gpu_type: str,
                       p_multiplier: int = 16):
    _params = model_yaml.replace('.yaml', '')
    params, mult = int(_params[:-1]), _params[-1]
    if mult == 'm':
        b_params = params / 1000
    elif mult == 'b':
        b_params = params
    else:
        raise ValueError

    gpu_mem = int(gpu_type.split('_')[-1][:-2])

    if p_multiplier * b_params > gpu_num * gpu_mem:
        print(
            f'WARNING: will not be running {model_yaml=} on {gpu_num=} {gpu_type=} since it probably will not fit into memory'
        )
        return False
    return True


def run_check_dtms(num_gpus: int, dtms: int, batch_size: int):
    if num_gpus * dtms > batch_size:
        print(
            f'WARNING: Cannot run with {batch_size=} on {num_gpus=} with {dtms=} ({num_gpus*dtms=}).'
        )
        return False
    return True


if __name__ == '__main__':
    args = parse_args()
    n_jobs = 0
    for max_seq_len in get_max_seq_lens(args.seq_len_exp):
        for cluster in args.clusters:
            for gpu_type in get_cluster_gpu_types(cluster):
                ng_lim = get_valid_gpu_lim(cluster, gpu_type)
                _gpu_nums = [ng for ng in args.gpu_nums if ng <= ng_lim]
                for gpu_num in _gpu_nums:
                    global_train_batch_sizes = get_global_train_batch_sizes(
                        max_seq_len, args.batch_size_exp, args.batch_sizes)
                    if not global_train_batch_sizes and args.microbatch_size is not None:
                        accum = args.accum or 1
                        global_train_batch_sizes = [
                            accum * gpu_num * args.microbatch_size
                        ]

                    for global_train_batch_size in global_train_batch_sizes:
                        for precision in args.precisions:
                            for model_yaml in args.model_yamls:

                                run = run_check_capacity(model_yaml,
                                                         gpu_num,
                                                         gpu_type,
                                                         p_multiplier=4)
                                if args.microbatch_size is not None:
                                    run = run and run_check_dtms(
                                        gpu_num, args.microbatch_size,
                                        global_train_batch_size)

                                if run:
                                    config: Tuple[str, int, int, str, str, int,
                                                  str] = (
                                                      model_yaml, max_seq_len,
                                                      global_train_batch_size,
                                                      cluster, gpu_type,
                                                      gpu_num, precision)
                                    run_config(config, args)
                                    n_jobs += 1

    print(f'{n_jobs=}')
