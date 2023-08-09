# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

# get file name
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--file_name',
                    type=str,
                    default='mpt-7b-instruct-responses-seed1',
                    help='file name')

args = parser.parse_args()
file_name = args.file_name

lima_test_prompts = []
lima_test_responses = []
# load the prompt response file
with open(file_name + '.txt', 'r') as f:
    # split based on '### End of prompt-response pair ###\n'
    prompt_response_pair = f.read().split(
        '### End of prompt-response pair ###\n')
    # now for every prompt response pair, split based on '### Response:\n'
    for pair in prompt_response_pair[:-1]:
        prompt_response = pair.split('### Response:\n', 1)
        # from the first element of the list, split based on '### Instruction:\n'
        instruction = prompt_response[0].split('### Instruction:\n')[1]
        lima_test_prompts.append(instruction)
        # append the response to the response list
        lima_test_responses.append(prompt_response[1])

# check length of both lists
print(len(lima_test_prompts))
print(len(lima_test_responses))

dict_lists = []
for i in range(len(lima_test_prompts)):
    dict_lists.append({
        'instruction': lima_test_prompts[i],
        'output': lima_test_responses[i],
        'generator': file_name,
        'dataset': 'lima',
        'datasplit': 'test'
    })

# write this dict_list to a json which is nicely formatted with indent=4
with open(file_name + '.json', 'w') as f:
    json.dump(dict_lists, f, indent=4)
