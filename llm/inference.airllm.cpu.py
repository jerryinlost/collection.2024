## Use powershell or cmd instead of cygwin 
## path not interpreted correctly with cygwin

# import os
# os.environ['HF_HOME'] = '/blabla/cache/'

import gradio as gr
from airllm import AutoModel
import torch

MAX_LENGTH = 128

# could use hugging face model repo id:
model = AutoModel.from_pretrained(
    # "~/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/c4a54320a52ed5f88b7a2f84496903ea4ff07b45",
    "C:\\Users\\kojy\\.cache\\huggingface\\hub\\models--meta-llama--Meta-Llama-3-8B-Instruct\\snapshots\\c4a54320a52ed5f88b7a2f84496903ea4ff07b45",
    device="cpu",
    # dtype=torch.float32,
    profiling_mode=True)

# print(output)
def generate_text(input_text):
    input_ids = model.tokenizer.encode(input_text, return_tensors="pt")
    output = model.generate(input_ids, max_length=100)
    return model.tokenizer.decode(output[0])

iface = gr.Interface(
    fn=generate_text, 
    inputs=gr.Textbox(placeholder="Enter prompt..."),
    outputs="text",
    title="LLaMA 3 70B Text Generation"
)

iface.launch(server_name="0.0.0.0", server_port=7860)


# import torch
# from airllm import AirLLMLlama2

# MAX_LENGTH = 128
# model = AirLLMLlama2(
#     "/Users/xxx/.cache/huggingface/hub/models--garage-bAInd--Platypus2-7B/snapshots/c27aff7201e611f301c0e19f351cbe74b1a9f1f1",
#     device="cpu",
#     dtype=torch.float32,
#     profiling_mode=True)
# input_text = [
#     'What is the capital of United States?',
# ]
# while True:
#     input_text = input("Input: ")
#     input_tokens = model.tokenizer(input_text,
#                                    return_tensors="pt",
#                                    return_attention_mask=False,
#                                    truncation=True,
#                                    max_length=MAX_LENGTH,
#                                    padding=False)

#     generation_output = model.generate(
#         input_tokens['input_ids'],
#         max_new_tokens=20,
#         use_cache=True,
#         return_dict_in_generate=True)

#     output = model.tokenizer.decode(generation_output.sequences[0])
    
#     print(output)