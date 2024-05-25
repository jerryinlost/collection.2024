import torch
import onnx
from denoiser import pretrained
from denoiser.dsp import convert_audio

model = pretrained.dns64().cpu()

# Create dummy input matching the model's input shape
dummy_input = torch.randn(1, 1, 16000)  # Adjust dimensions as needed

# Export the model to ONNX
torch.onnx.export(model, dummy_input, "model.onnx")