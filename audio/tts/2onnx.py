# Install the required packages:
#pip install torch onnx transformers

# Load the model and tokenizer:
import torch
from melo.api import TTS


# Speed is adjustable
speed = 1.0

# CPU is sufficient for real-time inference.
# You can set it manually to 'cpu' or 'cuda' or 'cuda:0' or 'mps'
device = 'auto' # Will automatically use GPU if available

# English 
text = "Did you ever hear a folk tale about a giant turtle?"
model = TTS(language='EN', device=device)

model.eval()

# #Prepare a dummy input:
# text = "Hello, this is a test."
# inputs = tokenizer(text, return_tensors="pt")

# #Export the model to ONNX:
# torch.onnx.export(
#     model, 
#     (inputs["input_ids"], inputs["attention_mask"]), 
#     "melotts.onnx", 
#     input_names=["input_ids", "attention_mask"], 
#     output_names=["logits"], 
#     opset_version=11,
#     dynamic_axes={
#         "input_ids": {0: "batch_size", 1: "sequence_length"},
#         "attention_mask": {0: "batch_size", 1: "sequence_length"},
#         "logits": {0: "batch_size", 1: "sequence_length"}
#     }
# )