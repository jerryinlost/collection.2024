import gradio as gr
from transformers import pipeline
from PIL import Image
import io
import numpy as np
import torch

# # from diffusers import DiffusionPipeline
# # https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0?text=a+photo+of+an+astronaut+riding+a+horse+on+mars
# # https://you.com/search?q=layernormkernelimpl+not+implemented+for+%27half%27+stable+diffusion&tbm=youchat&cfr=chatb&cid=c2_314078a2-2905-4e3c-8556-c42927e00cb4

# pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float32, use_safetensors=True, variant="fp32")
# # pipe.to("cuda")

# https://huggingface.co/OFA-Sys/small-stable-diffusion-v0
from diffusers import StableDiffusionPipeline

model_id = "stablediffusionapi/realistic-vision-v51"#"nota-ai/bk-sdm-small"#"OFA-Sys/small-stable-diffusion-v0"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)#,safety_checker=None) #for disable nsfw checking

