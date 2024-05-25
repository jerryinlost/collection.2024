import numpy as np
import onnx
import onnxruntime as ort
import torch
import torchaudio
from denoiser import pretrained

test_arr = np.random.randn(10, 3, 224, 224).astype(np.float32)
model = pretrained.dns64().cpu()

torch_output = model(torch.from_numpy(test_arr))

# input_names = ["input"]
# output_names = ["output"]
# torch.onnx.export(model, 
#                   dummy_input, 
#                   "mobilenet_v2.onnx", 
#                   verbose=False, 
#                   input_names=input_names, 
#                   output_names=output_names)

# model = onnx.load("mobilenet_v2.onnx")
# ort_session = ort.InferenceSession('mobilenet_v2.onnx')
# onnx_outputs = ort_session.run(None, {'input': test_arr})
# print('Export ONNX!')

