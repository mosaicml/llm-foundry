# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""For doing batch model inference and caching results to a file.

This will be faster than evaluating one at a time.
"""

import argparse
import json
from time import sleep

from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--name_or_path', type=str, required=True)
    parser.add_argument('--eval_data', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument('--top_p', type=float, default=0.95)
    parser.add_argument('--top_k', type=int, default=50)
    parser.add_argument('--temperature', type=float, default=0.2)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_length', type=int, default=2048)
    parser.add_argument('--trust_remote_code', action='store_true')
    parser.add_argument('--dtype',
                        choices=['fp32', 'fp16', 'bf16'],
                        default='bf16')
    return parser.parse_args()


def load_eval_in_batches(fname: str, batch_size: int):
    """Load the eval data in batches."""
    with open(fname) as f:
        l = []
        for line in f:
            l.append(json.loads(line)['prompt'])
    batches = []
    for i in range(0, len(l), batch_size):
        batch = l[i:i + batch_size]
        batch = [line for line in batch]
        batches.append(batch)
    return batches


def main():
    args = parse_args()
    if 'http' in args.name_or_path:
        from mcli import predict

        with open(args.output_file, 'w') as f:
            for batch in tqdm(
                    load_eval_in_batches(args.eval_data, args.batch_size)):
                # print(batch)
                preds = predict(
                    args.name_or_path, {
                        'input_strings': batch,
                        'temperature': args.temperature,
                        'top_p': args.top_p,
                        'max_length': args.max_length,
                    })['data']
                sleep(1.5)  # to avoid rate limiting
                for prompt, resp in zip(batch, preds):
                    # save as jsonl
                    f.write(
                        json.dumps({
                            'prompt': prompt,
                            'response':
                                resp[len(prompt):]  # remove the prompt
                        }) + '\n')

    else:
        import torch
        import transformers

        dtype = torch.float32 if args.dtype == 'fp32' else torch.float16 if args.dtype == 'fp16' else torch.bfloat16
        model = transformers.AutoModelForCausalLM.from_pretrained(
            args.name_or_path,
            trust_remote_code=args.trust_remote_code,
            torch_dtype=dtype)
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            args.name_or_path, padding_side='left')
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.to('cuda')
        model.eval()
        with open(args.output_file, 'w') as f:
            for batch in tqdm(
                    load_eval_in_batches(args.eval_data, args.batch_size)):
                inputs = tokenizer(batch,
                                   return_tensors='pt',
                                   padding=True,
                                   truncation=True,
                                   max_length=args.max_length)
                inputs = {k: v.to('cuda') for k, v in inputs.items()}
                outputs = model.generate(**inputs,
                                         top_p=args.top_p,
                                         top_k=args.top_k,
                                         temperature=args.temperature,
                                         do_sample=args.temperature > 0,
                                         max_length=args.max_length)
                # slice outputs to remove the input
                outputs = outputs[:, inputs['input_ids'].shape[1]:]
                preds = [
                    tokenizer.decode(output, skip_special_tokens=True)
                    for output in outputs
                ]
                for prompt, resp in zip(batch, preds):
                    # save as jsonl
                    f.write(
                        json.dumps({
                            'prompt': prompt,
                            'response': resp
                        }) + '\n')


if __name__ == '__main__':
    main()
