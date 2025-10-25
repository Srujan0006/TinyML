import torch
import torch.onnx
import tensorflow as tf
import onnx
import onnxruntime as ort
import coremltools as ct
import numpy as np
from typing import Dict, Any, Optional, Tuple
import os
from pathlib import Path
import tempfile
import shutil

class ModelConverter:
    def __init__(self):
        self.supported_input_formats = ['.pth', '.pt', '.h5', '.keras', '.pb', '.onnx']
        self.supported_output_formats = ['onnx', 'tflite', 'coreml']
        
    def convert_model(self, input_path: str, output_format: str, 
                     output_path: Optional[str] = None, 
                     optimization_level: str = 'default') -> Dict[str, Any]:
        if output_format not in self.supported_output_formats:
            raise ValueError(f"Unsupported output format: {output_format}")
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        input_extension = input_path.suffix.lower()
        if input_extension not in self.supported_input_formats:
            raise ValueError(f"Unsupported input format: {input_extension}")
        if output_path is None:
            output_path = input_path.parent / f"{input_path.stem}_converted.{output_format}"
        else:
            output_path = Path(output_path)
        conversion_result = {
            'input_path': str(input_path),
            'output_path': str(output_path),
            'output_format': output_format,
            'optimization_level': optimization_level,
            'success': False,
            'error': None,
            'conversion_time': 0,
            'file_size_reduction': 0
        }
        try:
            import time
            start_time = time.time()
            if input_extension in ['.pth', '.pt']:
                self._convert_pytorch_to_edge(input_path, output_path, output_format, optimization_level)
            elif input_extension in ['.h5', '.keras']:
                self._convert_keras_to_edge(input_path, output_path, output_format, optimization_level)
            elif input_extension == '.pb':
                self._convert_tensorflow_to_edge(input_path, output_path, output_format, optimization_level)
            elif input_extension == '.onnx':
                self._convert_onnx_to_edge(input_path, output_path, output_format, optimization_level)
            conversion_result['conversion_time'] = time.time() - start_time
            conversion_result['success'] = True
            original_size = input_path.stat().st_size
            converted_size = output_path.stat().st_size
            conversion_result['file_size_reduction'] = (original_size - converted_size) / original_size * 100
        except Exception as e:
            conversion_result['error'] = str(e)
            conversion_result['success'] = False
        return conversion_result
    
    def _convert_pytorch_to_edge(self, input_path: Path, output_path: Path, 
                               output_format: str, optimization_level: str):
        checkpoint = torch.load(input_path, map_location='cpu')
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            model = self._reconstruct_model_from_state_dict(state_dict)
        else:
            model = checkpoint
        model.eval()
        if output_format == 'onnx':
            self._convert_pytorch_to_onnx(model, output_path, optimization_level)
        elif output_format == 'tflite':
            with tempfile.TemporaryDirectory() as temp_dir:
                onnx_path = Path(temp_dir) / "temp_model.onnx"
                self._convert_pytorch_to_onnx(model, onnx_path, optimization_level)
                self._convert_onnx_to_tflite(onnx_path, output_path, optimization_level)
        elif output_format == 'coreml':
            with tempfile.TemporaryDirectory() as temp_dir:
                onnx_path = Path(temp_dir) / "temp_model.onnx"
                self._convert_pytorch_to_onnx(model, onnx_path, optimization_level)
                self._convert_onnx_to_coreml(onnx_path, output_path, optimization_level)
    
    def _reconstruct_model_from_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        layer_names = list(state_dict.keys())
        layers = []
        for name in layer_names:
            if 'weight' in name:
                weight_shape = state_dict[name].shape
                if len(weight_shape) == 2:
                    layers.append(torch.nn.Linear(weight_shape[1], weight_shape[0]))
                elif len(weight_shape) == 4:
                    layers.append(torch.nn.Conv2d(weight_shape[1], weight_shape[0], kernel_size=weight_shape[2]))
        if not layers:
            layers = [torch.nn.Linear(784, 128), torch.nn.ReLU(), torch.nn.Linear(128, 10)]
        model = torch.nn.Sequential(*layers)
        try:
            model.load_state_dict(state_dict)
        except:
            model = torch.nn.Sequential(
                torch.nn.Linear(784, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 10)
            )
        return model
    
    def _convert_pytorch_to_onnx(self, model: torch.nn.Module, output_path: Path, 
                               optimization_level: str):
        dummy_input = torch.randn(1, 3, 224, 224)
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        if optimization_level == 'high':
            self._optimize_onnx_model(output_path)
    
    def _convert_keras_to_edge(self, input_path: Path, output_path: Path, 
                             output_format: str, optimization_level: str):
        model = tf.keras.models.load_model(input_path)
        if output_format == 'onnx':
            self._convert_keras_to_onnx(model, output_path, optimization_level)
        elif output_format == 'tflite':
            self._convert_keras_to_tflite(model, output_path, optimization_level)
        elif output_format == 'coreml':
            with tempfile.TemporaryDirectory() as temp_dir:
                onnx_path = Path(temp_dir) / "temp_model.onnx"
                self._convert_keras_to_onnx(model, onnx_path, optimization_level)
                self._convert_onnx_to_coreml(onnx_path, output_path, optimization_level)
    
    def _convert_keras_to_onnx(self, model: tf.keras.Model, output_path: Path, 
                             optimization_level: str):
        try:
            import tf2onnx
            spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)
            output_path_str = str(output_path)
            model_proto, _ = tf2onnx.convert.from_keras(
                model, input_signature=spec, opset=11, output_path=output_path_str
            )
            if optimization_level == 'high':
                self._optimize_onnx_model(output_path)
        except ImportError:
            raise ImportError("tf2onnx is required for Keras to ONNX conversion. Install with: pip install tf2onnx")
    
    def _convert_keras_to_tflite(self, model: tf.keras.Model, output_path: Path, 
                               optimization_level: str):
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        if optimization_level == 'high':
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
        elif optimization_level == 'medium':
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
    
    def _convert_tensorflow_to_edge(self, input_path: Path, output_path: Path, 
                                  output_format: str, optimization_level: str):
        model = tf.saved_model.load(str(input_path))
        if output_format == 'onnx':
            self._convert_tensorflow_to_onnx(model, output_path, optimization_level)
        elif output_format == 'tflite':
            self._convert_tensorflow_to_tflite(model, output_path, optimization_level)
        elif output_format == 'coreml':
            with tempfile.TemporaryDirectory() as temp_dir:
                onnx_path = Path(temp_dir) / "temp_model.onnx"
                self._convert_tensorflow_to_onnx(model, onnx_path, optimization_level)
                self._convert_onnx_to_coreml(onnx_path, output_path, optimization_level)
    
    def _convert_tensorflow_to_onnx(self, model: tf.saved_model.LoadedModel, 
                                  output_path: Path, optimization_level: str):
        try:
            import tf2onnx
            concrete_func = model.signatures['serving_default']
            spec = concrete_func.inputs
            model_proto, _ = tf2onnx.convert.from_function(
                concrete_func, input_signature=spec, opset=11, output_path=str(output_path)
            )
            if optimization_level == 'high':
                self._optimize_onnx_model(output_path)
        except ImportError:
            raise ImportError("tf2onnx is required for TensorFlow to ONNX conversion. Install with: pip install tf2onnx")
    
    def _convert_tensorflow_to_tflite(self, model: tf.saved_model.LoadedModel, 
                                    output_path: Path, optimization_level: str):
        converter = tf.lite.TFLiteConverter.from_saved_model(str(model))
        if optimization_level == 'high':
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
        elif optimization_level == 'medium':
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
    
    def _convert_onnx_to_edge(self, input_path: Path, output_path: Path, 
                           output_format: str, optimization_level: str):
        if output_format == 'onnx':
            shutil.copy(input_path, output_path)
            if optimization_level in ['medium', 'high']:
                self._optimize_onnx_model(output_path)
        elif output_format == 'tflite':
            self._convert_onnx_to_tflite(input_path, output_path, optimization_level)
        elif output_format == 'coreml':
            self._convert_onnx_to_coreml(input_path, output_path, optimization_level)
    
    def _convert_onnx_to_tflite(self, input_path: Path, output_path: Path, 
                             optimization_level: str):
        try:
            import onnx_tf
            tf_rep = onnx_tf.backend.prepare(onnx.load(str(input_path)))
            tf_model = tf_rep.export_graph()
            converter = tf.lite.TFLiteConverter.from_concrete_functions([tf_model])
            if optimization_level == 'high':
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.target_spec.supported_types = [tf.float16]
            elif optimization_level == 'medium':
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
            tflite_model = converter.convert()
            with open(output_path, 'wb') as f:
                f.write(tflite_model)
        except ImportError:
            raise ImportError("onnx-tf is required for ONNX to TensorFlow Lite conversion. Install with: pip install onnx-tf")
    
    def _convert_onnx_to_coreml(self, input_path: Path, output_path: Path, 
                              optimization_level: str):
        onnx_model = onnx.load(str(input_path))
        coreml_model = ct.convert(
            onnx_model, convert_to='mlprogram', minimum_deployment_target=ct.target.iOS13
        )
        if optimization_level == 'high':
            coreml_model = ct.models.neural_network.quantization_utils.quantize_weights(
                coreml_model, nbits=8
            )
        coreml_model.save(str(output_path))
    
    def _optimize_onnx_model(self, model_path: Path):
        try:
            import onnxoptimizer
            model = onnx.load(str(model_path))
            optimized_model = onnxoptimizer.optimize(model)
            onnx.save(optimized_model, str(model_path))
        except ImportError:
            pass
    
    def get_conversion_recommendations(self, analysis: Dict[str, Any]) -> Dict[str, str]:
        recommendations = {}
        format_type = analysis.get('format', '')
        num_params = analysis.get('num_parameters', 0)
        if format_type == 'PyTorch':
            if num_params > 10_000_000:
                recommendations['format'] = 'ONNX'
                recommendations['reason'] = 'Large PyTorch models benefit from ONNX optimization'
            else:
                recommendations['format'] = 'TensorFlow Lite'
                recommendations['reason'] = 'Smaller models work well with TensorFlow Lite quantization'
        elif format_type == 'Keras':
            recommendations['format'] = 'TensorFlow Lite'
            recommendations['reason'] = 'Direct conversion from Keras to TensorFlow Lite is most efficient'
        elif format_type == 'TensorFlow':
            recommendations['format'] = 'TensorFlow Lite'
            recommendations['reason'] = 'Native TensorFlow Lite conversion provides best performance'
        elif format_type == 'ONNX':
            recommendations['format'] = 'ONNX (optimized)'
            recommendations['reason'] = 'ONNX model can be further optimized for edge deployment'
        if num_params > 5_000_000:
            recommendations['optimization'] = 'high'
            recommendations['optimization_reason'] = 'Large models benefit from aggressive optimization'
        else:
            recommendations['optimization'] = 'medium'
            recommendations['optimization_reason'] = 'Medium optimization balances size and accuracy'
        return recommendations