#!/usr/bin/env python
'''LLM demo
'''
from transformers import AutoModelForCausalLM, AutoTokenizer


model_name = 'microsoft/phi-3-mini-4k-instruct'
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map='auto',
    torch_dtype='auto',
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
)


prompt = 'This is a demo of the LLM. Please tell me what is LLM?<|assistant|>'

input_ids = tokenizer.encode(prompt, return_tensors='pt').input_ids

generation_output = model.generate(
    input_ids=input_ids,
    max_new_tokens=20,
)


print(tokenizer.decode(generation_output[0]))