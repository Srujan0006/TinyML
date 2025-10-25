import tensorflow as tf
import torch as pt
import onnx
from scklearn.linear_model import LogisticRegression
import joblib

def parse_tensorflow_model(model_path):
    model = tf.keras.models.load_model(model_path)
    print(f"Model loaded: {model_path}")
    for layer in model.layers:
        print(f"Layer: {layer.name} | Type: {type(layer).__name__} | Output Shape: {layer.output_shape} | Data Type: {layer.dtype}")

def parse_pytorch_model(model_path):
    model = pt.load(model_path, map_location=pt.device('cpu'))
    print(f"Model loaded: {model_path}")
    for name, module in model.named_modules():
        print(f"Layer: {name} | Type: {type(module).__name__}")

def parse_onnx_model(model_path):
    model = onnx.load(model_path)
    print(f"Model loaded: {model_path}")
    for node in model.graph.node:
        print(f"Op Name: {node.name} | Op Type: {node.op_type}")

    print("--- Inputs ---")
    for input_tensor in model.graph.input:
        print(f"{input_tensor.name}: {input_tensor.type}")
    print("--- Outputs ---")
    for output_tensor in model.graph.output:
        print(f"{output_tensor.name}: {output_tensor.type}")
