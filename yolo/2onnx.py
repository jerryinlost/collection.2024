import torch
# Load YOLOv8 model
from ultralytics import YOLO

model_path = 'models/keremberke/yolov8n-blood-cell-detection/best.pt'

model = YOLO(model_path)  # Load a pre-trained YOLOv8 model

# Set the model to evaluation mode
model.model.eval()

# Create a dummy input tensor with the appropriate dimensions
dummy_input = torch.randn(1, 3, 640, 640)

# Specify the output file name
output_onnx_file = "yolov8n_blood_cell_detection.onnx"

# Export the model to ONNX
torch.onnx.export(
    model.model, 
    dummy_input, 
    output_onnx_file, 
    opset_version=11,  # Use an appropriate opset version
    input_names=["input"], 
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},  # Enable dynamic batching
)

print(f"Model successfully converted to {output_onnx_file}")