import tensorflow as tf
import onnx
import tf2onnx

# Load the ONNX model
onnx_model = onnx.load("model.onnx")

# Convert to TensorFlow model
tf_rep = tf2onnx.tfonnx.process_tf_graph(tf.import_graph_def(onnx_model.graph), input_names=['input'], output_names=['output'])

# Convert to TFLite model
converter = tf.lite.TFLiteConverter.from_frozen_graph(tf_rep.graph_def, ['input'], ['output'])
tflite_model = converter.convert()

# Save the TFLite model
with open("model.tflite", "wb") as f:
    f.write(tflite_model)