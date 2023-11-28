import time
import torch
import transformers
from transformers import AutoTokenizer
from transformers import pipeline

import habana_frameworks.torch.core as htcore

device = torch.device("hpu")

name = 'mosaicml/mpt-7b'
config = transformers.AutoConfig.from_pretrained(name, trust_remote_code=True)

model = transformers.AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16, trust_remote_code=True)

model.to(device)

tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')

pipe = pipeline('text-generation', model=model, tokenizer=tokenizer, device=device)

prompts = ['Here is a recipe for vegan banana bread:\n', 'Here is a recipe for burgers:\n', 'Here is the recipe of chines rice\n', 'Here is a recipe for italian pizza:\n', 'Here is a recipe for vegan banana bread:\n', 'Here is a recipe for burgers:\n', 'Here is the recipe of chines rice\n', 'Here is a recipe for italian pizza:\n', 'Here is a recipe for vegan banana bread:\n', 'Here is a recipe for burgers:\n', 'Here is the recipe of chines rice\n', 'Here is a recipe for italian pizza:\n']
inference_times = []
answers = []

for promt in prompts:
    start_time = time.time()
    answer = pipe(promt, max_new_tokens=100, do_sample=True, use_cache=True)
    end_time = time.time()
    inference_times.append(end_time - start_time)
    answers.append(answer[0]['generated_text'])

total_inference = 0
total_tokens = 0
for inference_time  in inference_times:
    total_inference += inference_time

print("Avg inference time: {}".format(total_inference /  len(prompts)))

for answer in answers:
    total_tokens += len(answer)

print("Avg tokens per sec: {}".format(total_tokens /  total_inference))