import torch
import tensorflow as tf
import onnx
import numpy as np
from typing import Dict, Any, List, Tuple
import json
import os
from pathlib import Path

class ModelParser:
    def __init__(self):
        self.supported_formats = {
            '.pth': self._parse_pytorch,
            '.pt': self._parse_pytorch,
            '.h5': self._parse_keras,
            '.keras': self._parse_keras,
            '.pb': self._parse_tensorflow,
            '.onnx': self._parse_onnx
        }
    
    def parse_model(self, file_path: str) -> Dict[str, Any]:
        file_path = Path(file_path)
        file_extension = file_path.suffix.lower()
        if file_extension not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_extension}")
        try:
            parser_func = self.supported_formats[file_extension]
            return parser_func(file_path)
        except Exception as e:
            raise Exception(f"Error parsing model: {str(e)}")
    
    def _parse_pytorch(self, file_path: Path) -> Dict[str, Any]:
        try:
            checkpoint = torch.load(file_path, map_location='cpu')
            analysis = {
                'format': 'PyTorch',
                'file_size': file_path.stat().st_size,
                'file_path': str(file_path)
            }
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                    analysis['model_type'] = 'state_dict'
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                    analysis['model_type'] = 'state_dict'
                else:
                    state_dict = checkpoint
                    analysis['model_type'] = 'direct_state_dict'
                analysis['num_parameters'] = sum(p.numel() for p in state_dict.values() if isinstance(p, torch.Tensor))
                analysis['layers'] = list(state_dict.keys())
                analysis['layer_count'] = len(analysis['layers'])
                if 'optimizer_state_dict' in checkpoint:
                    analysis['has_optimizer'] = True
                if 'epoch' in checkpoint:
                    analysis['epoch'] = checkpoint['epoch']
                if 'loss' in checkpoint:
                    analysis['loss'] = checkpoint['loss']
            else:
                analysis['model_type'] = 'model_object'
                analysis['model_class'] = str(type(checkpoint).__name__)
                if hasattr(checkpoint, 'parameters'):
                    analysis['num_parameters'] = sum(p.numel() for p in checkpoint.parameters())
                if hasattr(checkpoint, 'modules'):
                    analysis['layers'] = [name for name, _ in checkpoint.named_modules()]
                    analysis['layer_count'] = len(analysis['layers'])
            return analysis
        except Exception as e:
            raise Exception(f"Error parsing PyTorch model: {str(e)}")
    
    def _parse_keras(self, file_path: Path) -> Dict[str, Any]:
        try:
            model = tf.keras.models.load_model(file_path)
            analysis = {
                'format': 'Keras',
                'file_size': file_path.stat().st_size,
                'file_path': str(file_path),
                'model_type': 'keras_model',
                'num_parameters': model.count_params(),
                'layer_count': len(model.layers),
                'layers': [layer.name for layer in model.layers],
                'input_shape': model.input_shape,
                'output_shape': model.output_shape
            }
            summary_lines = []
            model.summary(print_fn=lambda x: summary_lines.append(x))
            analysis['summary'] = '\n'.join(summary_lines)
            return analysis
        except Exception as e:
            raise Exception(f"Error parsing Keras model: {str(e)}")
    
    def _parse_tensorflow(self, file_path: Path) -> Dict[str, Any]:
        try:
            model = tf.saved_model.load(str(file_path))
            analysis = {
                'format': 'TensorFlow',
                'file_size': file_path.stat().st_size,
                'file_path': str(file_path),
                'model_type': 'saved_model'
            }
            if hasattr(model, 'signatures'):
                signatures = list(model.signatures.keys())
                analysis['signatures'] = signatures
                analysis['signature_count'] = len(signatures)
            try:
                concrete_func = model.signatures['serving_default']
                analysis['input_signature'] = str(concrete_func.inputs)
                analysis['output_signature'] = str(concrete_func.outputs)
            except:
                pass
            return analysis
        except Exception as e:
            raise Exception(f"Error parsing TensorFlow model: {str(e)}")
    
    def _parse_onnx(self, file_path: Path) -> Dict[str, Any]:
        try:
            model = onnx.load(str(file_path))
            analysis = {
                'format': 'ONNX',
                'file_size': file_path.stat().st_size,
                'file_path': str(file_path),
                'model_type': 'onnx_model',
                'ir_version': model.ir_version,
                'producer_name': model.producer_name,
                'producer_version': model.producer_version
            }
            graph = model.graph
            analysis['node_count'] = len(graph.node)
            analysis['input_count'] = len(graph.input)
            analysis['output_count'] = len(graph.output)
            inputs = []
            outputs = []
            for input_tensor in graph.input:
                input_info = {
                    'name': input_tensor.name,
                    'type': str(input_tensor.type.tensor_type.elem_type),
                    'shape': [dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim]
                }
                inputs.append(input_info)
            for output_tensor in graph.output:
                output_info = {
                    'name': output_tensor.name,
                    'type': str(output_tensor.type.tensor_type.elem_type),
                    'shape': [dim.dim_value for dim in output_tensor.type.tensor_type.shape.dim]
                }
                outputs.append(output_info)
            analysis['inputs'] = inputs
            analysis['outputs'] = outputs
            node_types = {}
            for node in graph.node:
                node_type = node.op_type
                node_types[node_type] = node_types.get(node_type, 0) + 1
            analysis['node_types'] = node_types
            return analysis
        except Exception as e:
            raise Exception(f"Error parsing ONNX model: {str(e)}")
    
    def get_model_info_summary(self, analysis: Dict[str, Any]) -> str:
        summary = f"""
**Model Analysis Summary**

**Format:** {analysis.get('format', 'Unknown')}
**File Size:** {analysis.get('file_size', 0) / (1024*1024):.2f} MB
**Model Type:** {analysis.get('model_type', 'Unknown')}

"""
        if 'num_parameters' in analysis:
            summary += f"**Parameters:** {analysis['num_parameters']:,}\n"
        if 'layer_count' in analysis:
            summary += f"**Layers:** {analysis['layer_count']}\n"
        if 'input_shape' in analysis:
            summary += f"**Input Shape:** {analysis['input_shape']}\n"
        if 'output_shape' in analysis:
            summary += f"**Output Shape:** {analysis['output_shape']}\n"
        if 'node_count' in analysis:
            summary += f"**Nodes:** {analysis['node_count']}\n"
        if 'node_types' in analysis:
            summary += "\n**Node Types:**\n"
            for node_type, count in analysis['node_types'].items():
                summary += f"- {node_type}: {count}\n"
        return summary
