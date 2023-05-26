# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import random


def format_sample(a, b, c, operator='+'):
    # TODO: add aliases here so that the model gets points even if it is "close"
    return {'context': f'{a}+{b}=', 'answer': str(c), 'aliases': [str(c)]}


# generates a jsonl file of "a+b=c" samples where a, b are integers with num_digits digits
def make_arithmetic_dataset(out_filename,
                            num_samples=1000,
                            num_digits=3,
                            random_subset=False):
    with open(out_filename, 'w', encoding='utf8') as f:
        if random_subset:
            # then just pick num_samples randomly
            for idx in range(num_samples):
                # TODO: handle duplicates
                max_val = 10**num_digits
                a = random.randint(0, max_val)
                b = random.randint(0, max_val)
                c = a + b
                row = format_sample(a, b, c=c, operator='+')
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
        else:
            # consider all possible addition samples of num_digits
            for a in range(10**num_digits):
                for b in range(10**num_digits):
                    row = format_sample(a, b, c=a + b, operator='+')
                    f.write(json.dumps(row, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Script to generate addition dataset for eval on LLMFoundry'
    )
    parser.add_argument('--num-digits',
                        type=int,
                        default=3,
                        metavar='d',
                        help='max number of digits for operands (default: 3)')
    parser.add_argument('--random-subset',
                        action='store_true',
                        default=False,
                        help='chooses a random subset of the possible samples')
    parser.add_argument('--out-filename',
                        type=str,
                        default='addition.jsonl',
                        help='name of output file (default: addition.jsonl)')
    parser.add_argument(
        '--num-samples',
        type=int,
        default=1000,
        metavar='n',
        help=
        'number of random samples to choose if random_subset=True (default: 1000)'
    )

    parser_args = parser.parse_args()

    if not parser_args.random_subset:
        parser_args.num_samples = 10**(2 * parser_args.num_digits)
    elif parser_args.num_samples > (10**(2 * parser_args.num_digits)):
        print(
            f'Warning: num_samples={parser_args.num_samples} is greater than the number of possible samples for {parser_args.num_digits} digits.'
        )
        print(f'Duplicate samples will be generated.')
    print(
        f'Generating addition dataset with with {parser_args.num_samples} samples of up to {parser_args.num_digits} digits'
    )
    make_arithmetic_dataset(parser_args.out_filename,
                            num_samples=parser_args.num_samples,
                            num_digits=parser_args.num_digits,
                            random_subset=parser_args.random_subset)
