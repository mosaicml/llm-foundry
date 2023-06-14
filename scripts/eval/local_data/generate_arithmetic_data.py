# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import random


# this is for qa tasks
def format_sample_old(a, b, c, operator='+'):
    # TODO: add aliases here so that the model gets points even if it is "close"
    return {'context': f'{a}+{b}=', 'answer': str(c), 'aliases': [str(c)]}


# this is for a language modeling task
def format_sample(operands, c, operator='+', with_spaces=False):
    if with_spaces:
        operator = f' {operator} '
        equal_to_str = ' ='
        continuation_str = f' {str(c)}'
    else:
        equal_to_str = '='
        continuation_str = str(c)
    context_str = operator.join([str(operand) for operand in operands])
    return {'context': f'{context_str}{equal_to_str}', 'continuation': continuation_str}


# generates a jsonl file of "a+b=c" samples where a, b are integers with num_digits digits
def make_arithmetic_dataset(out_filename,
                            num_samples=1000,
                            num_digits=3,
                            random_subset=False,):
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


# generates a jsonl file of "a_1 \op a_2 \op ... a_n=c" samples where a_i are integers with num_digits digits
# n is chosen randomly between 2 and 4
def make_arithmetic_dataset_v2(out_filename,
                            num_samples=1000,
                            num_digits=3,
                            random_subset=False,
                            max_num_operands=5,
                            operators=['+', '*', '-'],
                            with_spaces=False,):
    with open(out_filename, 'w', encoding='utf8') as f:
        if random_subset:
            # then just pick num_samples randomly
            for idx in range(num_samples):
                # choose operator randomly from operators
                operator = random.choice(operators)
                num_operands = random.choice(range(2, max_num_operands + 1))
                max_val = 10**num_digits
                # adding a hack to keep the multiplication samples small
                if operator == '*':
                    max_val = 10**(num_digits-1)
                    num_operands = random.choice([2, 3])
                a_1 = random.randint(0, max_val)
                operands = [a_1]
                c = a_1
                for i in range(num_operands-1):
                    a_i = random.randint(0, max_val)
                    operands.append(a_i)
                    if operator == '+':
                        c += a_i
                    elif operator == '-':
                        c -= a_i
                    elif operator == '*':
                        c *= a_i
                    else:
                        print(f"Operators {operator} not supported")
                        raise NotImplementedError
                row = format_sample(operands, c=c, operator=operator, with_spaces=with_spaces)
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
        else:
            print("This is too many. Not generating all samples.")
            exit()
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
    parser.add_argument('--max-num-operands',
                        type=int,
                        default=2,
                        metavar='d',
                        help='max number of operands to use (default: 2)')
    parser.add_argument('--with-spaces',
                        action='store_true',
                        default=False,
                        help='add spaces around operators (default: False)')

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
    # make_arithmetic_dataset(parser_args.out_filename,
    #                         num_samples=parser_args.num_samples,
    #                         num_digits=parser_args.num_digits,
    #                         random_subset=parser_args.random_subset)

    random.seed(42)
    make_arithmetic_dataset_v2(parser_args.out_filename,
                            num_samples=parser_args.num_samples,
                            num_digits=parser_args.num_digits,
                            random_subset=parser_args.random_subset,
                            max_num_operands=parser_args.max_num_operands,
                            operators=['+', '*', '-'],
                            with_spaces=parser_args.with_spaces)


# example runs

# python generate_arithmetic_data.py \
#   --out-filename dummy_arithmetic_nospaces_data.jsonl \
#   --num-samples 10 \
#   --num-digits 3 \
#   --random-subset \
#   --max-num-operands 3

# python generate_arithmetic_data.py \
#   --out-filename dummy_arithmetic_withspaces_data.jsonl \
#   --num-samples 10 \
#   --num-digits 3 \
#   --random-subset \
#   --max-num-operands 3 \
#   --with-spaces

# python generate_arithmetic_data.py \
#   --out-filename simple_arithmetic_nospaces_data.jsonl \
#   --num-samples 1000 \
#   --num-digits 3 \
#   --random-subset \
#   --max-num-operands 5

# python generate_arithmetic_data.py \
#   --out-filename simple_arithmetic_withspaces_data.jsonl \
#   --num-samples 1000 \
#   --num-digits 3 \
#   --random-subset \
#   --max-num-operands 5 \
#   --with-spaces
