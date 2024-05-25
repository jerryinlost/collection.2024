import torch
from diffusers import StableDiffusionPipeline
import onnx

# Load the model using diffusers
model_name = "stablediffusionapi/realistic-vision-v51"
pipeline = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float32)
model = pipeline.unet

# Set the model to evaluation mode
model.eval()

# Create a dummy input (adjust shape based on model's requirement)
# For Stable Diffusion, this is typically a latent tensor of shape (batch_size, 4, height//8, width//8)
dummy_input = torch.randn(1, 4, 64, 64, device='cpu')

# Define the ONNX export path
onnx_model_path = "realistic-vision-v51_unet.onnx"

# Export the model
torch.onnx.export(
    model,                           # Model to be exported
    dummy_input,                     # Dummy input
    onnx_model_path,                 # Path where the model will be saved
    export_params=True,              # Store the trained parameter weights inside the model file
    opset_version=11,                # ONNX version to export the model to
    input_names=['latent'],          # Name of the input node
    output_names=['output'],         # Name of the output node
    dynamic_axes={'latent': {0: 'batch_size', 2: 'height', 3: 'width'}, 'output': {0: 'batch_size', 2: 'height', 3: 'width'}}  # Dynamic axes for variable-length inputs
)

print(f"Model has been converted to ONNX and saved at {onnx_model_path}")