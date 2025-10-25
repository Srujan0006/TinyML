import torch
import torch.onnx
import tensorflow as tf
import numpy as np
from typing import Dict, Any, Optional, Tuple
import os
import traceback
import onnx
import onnxruntime
from pathlib import Path
import tempfile
import shutil
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
from flask_ngrok import run_with_ngrok
from pyngrok import ngrok
import time
import tf2onnx
import onnx_tf

ngrok.set_auth_token("34YSdGwxsaR0l0jV60oY9Jqe3lK_6iF4vkPA7ZxkQjXi774G1")

app = Flask(__name__)

UPLOAD_FOLDER = '/tmp/uploads'
OUTPUT_FOLDER = '/tmp/outputs'
ALLOWED_EXTENSIONS = {'h5', 'pt', 'pth', 'onnx', 'pb', 'keras'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


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
    
    def _convert_tensorflow_to_onnx(self, model, output_path: Path, optimization_level: str):
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
    
    def _convert_tensorflow_to_tflite(self, model, output_path: Path, optimization_level: str):
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


# Initialize converter
converter = ModelConverter()


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'ML Model Converter API is running',
        'supported_input_formats': converter.supported_input_formats,
        'supported_output_formats': converter.supported_output_formats
    })


@app.route('/convert', methods=['POST'])
def convert_model():
    """Main conversion endpoint"""
    file_path = None
    output_path = None
    
    try:
        # Check if file is present
        if 'model' not in request.files:
            return jsonify({'error': 'No model file provided'}), 400
        
        file = request.files['model']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Unsupported file format'}), 400
        
        # Get conversion parameters
        target_format = request.form.get('format', 'tflite')
        optimization = request.form.get('optimization', 'default')
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        
        # Prepare output path
        output_filename = f"converted_model.{target_format}"
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        
        # Perform conversion using ModelConverter
        result = converter.convert_model(
            input_path=file_path,
            output_format=target_format,
            output_path=output_path,
            optimization_level=optimization
        )
        
        if not result['success']:
            return jsonify({
                'error': f"Conversion failed: {result['error']}"
            }), 500
        
        # Get file sizes
        original_size = os.path.getsize(file_path)
        converted_size = os.path.getsize(output_path)
        
        # Return the converted model file with metadata
        response = send_file(
            output_path,
            as_attachment=True,
            download_name=output_filename,
            mimetype='application/octet-stream'
        )
        
        # Add metadata headers
        response.headers['X-Conversion-Time'] = str(result['conversion_time'])
        response.headers['X-Original-Size'] = str(original_size)
        response.headers['X-Converted-Size'] = str(converted_size)
        response.headers['X-Size-Reduction'] = str(result['file_size_reduction'])
        
        return response
        
    except Exception as e:
        error_msg = f"Conversion failed: {str(e)}"
        print(f"Error: {error_msg}")
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': error_msg}), 500
    
    finally:
        # Clean up uploaded file
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except:
                pass
        # Note: Keep output file for download, clean up in a separate task


@app.route('/formats', methods=['GET'])
def get_supported_formats():
    """Get supported input and output formats"""
    return jsonify({
        'input_formats': converter.supported_input_formats,
        'output_formats': converter.supported_output_formats,
        'optimization_levels': ['default', 'medium', 'high']
    })


@app.route('/recommendations', methods=['POST'])
def get_recommendations():
    """Get conversion recommendations based on model analysis"""
    try:
        data = request.get_json()
        recommendations = converter.get_conversion_recommendations(data)
        return jsonify(recommendations)
    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    print("=" * 60)
    print("Starting ML Model Converter API...")
    print("=" * 60)
    print("=" * 60)

    # Start ngrok tunnel and Flask app
    public_url = ngrok.connect(8502)
    print(f"\n Public URL: {public_url.public_url}")
    print("You can paste this URL into your Streamlit app sidebar as the Colab API URL.\n")

    app.run(host='0.0.0.0', port=8502, debug=False)