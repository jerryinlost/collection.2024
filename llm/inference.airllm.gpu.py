model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

import gradio as gr
from airllm import AutoModel

MAX_LENGTH = 128

# could use hugging face model repo id:
model = AutoModel.from_pretrained(model_id)

# or use model's local path...
# model_path = f"F:\\Models\\llm\\llama3\\Meta-Llama-3-8B\\"
#              #"/home/ubuntu/.cache/huggingface/hub/models--garage-bAInd--Platypus2-70B-instruct/snapshots/b585e74bcaae02e52665d9ac6d23f4d0dbc81a0f"
# model = AutoModel.from_pretrained(model_path)

# input_text = [
#         'What is the capital of United States?',
#         #'I like',
#     ]

# input_tokens = model.tokenizer(input_text,
#     return_tensors="pt", 
#     return_attention_mask=False, 
#     truncation=True, 
#     max_length=MAX_LENGTH, 
#     padding=False)
           
# generation_output = model.generate(
#     input_tokens['input_ids'].cuda(), 
#     max_new_tokens=20,
#     use_cache=True,
#     return_dict_in_generate=True)

# output = model.tokenizer.decode(generation_output.sequences[0])

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
