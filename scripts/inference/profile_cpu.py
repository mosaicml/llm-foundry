import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "mosaicml/mpt-7b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, low_cpu_mem_usage=False,trust_remote_code=True)

import time

timea = time.time()
prompt = "A lion is"
inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
outputs = model.generate(
    **inputs, max_new_tokens=20, do_sample=True, temperature=0.75 , return_dict_in_generate=True
)
token = outputs.sequences[0]
output_str = tokenizer.decode(token)
print(output_str)
print("timea = time.time()",-timea + time.time())
