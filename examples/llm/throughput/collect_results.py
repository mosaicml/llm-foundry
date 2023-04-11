# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

import argparse
import csv
import math
from typing import Any, Dict

from mcli import sdk as msdk

GPU_AVAILABLE_FLOPS = 312_000_000_000_000


def str_to_bool(value):
    # helper fn
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')


def parse_args():
    parser = argparse.ArgumentParser(description="""
        Parse run configs to get MosaicGPT training throughput.
        MFU and HFU are defined in https://arxiv.org/abs/2205.05198
        All FLOP calculations do not include norm, act, residual, etc.
        """)

    parser.add_argument('--project', type=str, default='tput')
    parser.add_argument('--filters', type=str, default=[], nargs='+')
    parser.add_argument('-s',
                        '--save-path',
                        type=str,
                        default='benchmark_results')
    parser.add_argument('-p',
                        '--print-results',
                        type=str_to_bool,
                        nargs='?',
                        const=True,
                        default=False)

    return parser.parse_args()


def get_runs(args):
    runs = [r for r in msdk.get_runs() if args.project in r.name]
    for filter in args.filters:
        runs = [r for r in runs if filter in r.name]

    def sort_key(r):
        model_name = r.name.split('-')[2]
        num_gpu = r.config.gpu_num
        if model_name[-1] == 'm':
            model_name_size = 1e6
        elif model_name[-1] == 'b':
            model_name_size = 1e9
        else:
            print(model_name)
            raise ValueError
        model_size = int(model_name[:-1])
        return (model_name_size, model_size, r.config.parameters['max_seq_len'],
                num_gpu, r.config.parameters['global_train_batch_size'])

    runs.sort(reverse=True, key=sort_key)

    return runs


def filter_runs(runs):
    pop_runs = []
    for run in runs:
        if run.status == msdk.RunStatus('FAILED'):
            print(
                f"run {run.name} has FAILED (likely due to OOM error but we'd recommend checking.)"
            )
            pop_runs.append(run)

    for run in pop_runs:
        runs.pop(runs.index(run))

    pop_runs = []
    for run in runs:
        if run.status in [
                msdk.RunStatus('FAILED_PULL'),
                msdk.RunStatus('PENDING'),
                msdk.RunStatus('QUEUED'),
                msdk.RunStatus('RUNNING'),
                msdk.RunStatus('SCHEDULED'),
                msdk.RunStatus('STARTING'),
                msdk.RunStatus('STOPPED'),
                msdk.RunStatus('STOPPING'),
                msdk.RunStatus('TERMINATING'),
        ]:
            print(f'run {run.name} has run status {run.status}')
            pop_runs.append(run)
    for run in pop_runs:
        runs.pop(runs.index(run))

    return runs


def parse_run(run) -> Dict[str, Any]:
    n_params = micro_batchsize = throughput = -1

    model_name = run.name.split('-')[2]
    gpu_num = run.config.gpu_num
    gpu_type = run.config.gpu_type

    fsdp_config = run.config.parameters['fsdp_config']

    seq_len = run.config.parameters['max_seq_len']
    global_train_batch_size = run.config.parameters['global_train_batch_size']
    activation_checkpointing = fsdp_config['activation_checkpointing']

    logs = msdk.get_run_logs(run)
    lines = ''
    for line in logs:
        lines += line
    lines = lines.split('\n')

    for line in lines:
        if line.startswith('n_params'):
            n_params = int(line.split(' ')[-1])
            break

    lines.reverse()

    for line in lines:
        if 'trainer/device_train_microbatch_size' in line:
            micro_batchsize = int(line.split(' ')[-1])
            break

    for line in lines:
        if 'throughput/samples_per_sec' in line:
            throughput = float(line.split(' ')[-1])
            break

    d_model = run.config.parameters['model']['d_model']
    n_layers = run.config.parameters['model']['n_layers']

    # mfu is approximated using thoughtput and param count
    # the number of paramters is approximately the number of multiply-accumulates (MAC) in the network
    # each MAC has 2 FLOPs - we multiply by 2 ie 2 * n_param
    # there are 3 passes of a NN (fwd, bwd, delta) - we multiply by 3 ie 2 * 3 * n_param
    # this gets us FLOPs / token
    flops_per_token = 2 * n_params
    flops_per_seq = flops_per_token * seq_len

    # there are 2 FLOPS per mac; there is A=Q*K^T and out=A*V ops (ie mult by 2)
    attn_flops_per_seq = n_layers * 2 * 2 * (d_model * (seq_len**2))
    # there are 2 ops in bwd pass and 1 in fwd pass so we mult by 3
    mfu_w_attn = (3 * flops_per_seq + 3 * attn_flops_per_seq) * throughput / (
        gpu_num * GPU_AVAILABLE_FLOPS)

    if activation_checkpointing:
        hfu_w_attn = (4 * flops_per_seq + 4 * attn_flops_per_seq
                     ) * throughput / (gpu_num * GPU_AVAILABLE_FLOPS)
    else:
        hfu_w_attn = mfu_w_attn

    return {
        'Model':
            model_name,
        'SeqLen (T)':
            seq_len,
        '# GPUs':
            gpu_num,
        'GPU':
            gpu_type,
        'MFU':
            round(mfu_w_attn * 100, 2),
        'HFU':
            round(hfu_w_attn * 100, 2),
        'MicroBatchSize':
            micro_batchsize,
        'GradAccum':
            math.ceil(global_train_batch_size / gpu_num / micro_batchsize),
        'GlobalBatchSize':
            global_train_batch_size,
        'Throughput (S/s)':
            int(throughput),
        'Throughput (T/s)':
            int(throughput * seq_len),
        'Throughput (T/s/GPU)':
            int(throughput * seq_len / gpu_num),
        'GlobalBatchSize (T)':
            global_train_batch_size * seq_len,
        'Precision':
            run.config.parameters['precision'],
        'MP Mode':
            fsdp_config['mixed_precision'],
        'Sharding Strategy':
            fsdp_config['sharding_strategy'],
        'Activation Checkpointing':
            activation_checkpointing,
        'Activation CPUOffload':
            str(fsdp_config['activation_cpu_offload']),
        'NumParams':
            n_params,
    }


def main(args):
    runs = get_runs(args)
    runs = filter_runs(runs)

    results = []
    for run in runs:
        try:
            results.append(parse_run(run))
        except Exception as e:
            print(f'{run.name=} not parsed')
            print(e)

    if results:
        csv_name = args.save_path + '.csv'
        with open(csv_name, 'w') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            for result in results:
                writer.writerow(result)

        md_name = args.save_path + '.md'
        fieldnames = results[0].keys()
        with open(md_name, 'w') as f:
            fmt = '| ' + ' {} |' * len(fieldnames) + '\n'

            f.write(fmt.format(*fieldnames))
            f.write(fmt.format(*['---' for _ in fieldnames]))
            if args.print_results:
                print(fmt.format(*fieldnames), end='')
                print(fmt.format(*['---' for _ in fieldnames]), end='')
            for result in results:
                if args.print_results:
                    print(fmt.format(*result.values()), end='')
                f.write(fmt.format(*result.values()))
    else:
        print('WARNING: No results parsed.')


if __name__ == '__main__':
    args = parse_args()
    main(args)
