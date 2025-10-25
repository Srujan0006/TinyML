#!/usr/bin/env python3
"""
Example script demonstrating how to use the AI Model Edge Converter programmatically.
"""

import torch
import torch.nn as nn
from parser import ModelParser
from converter import ModelConverter
import tempfile
import os

def create_sample_model():
    """Create a sample PyTorch model for demonstration."""
    
    class SimpleCNN(nn.Module):
        def __init__(self):
            super(SimpleCNN, self).__init__()
            self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(64 * 56 * 56, 128)
            self.fc2 = nn.Linear(128, 10)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.5)
        
        def forward(self, x):
            x = self.pool(self.relu(self.conv1(x)))
            x = self.pool(self.relu(self.conv2(x)))
            x = x.view(-1, 64 * 56 * 56)
            x = self.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x
    
    model = SimpleCNN()
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Save model
    with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp_file:
        torch.save(model.state_dict(), tmp_file.name)
        return tmp_file.name

def main():
    """Main demonstration function."""
    
    print("🤖 AI Model Edge Converter - Example Usage")
    print("=" * 50)
    
    # Create a sample model
    print("📦 Creating sample PyTorch model...")
    model_path = create_sample_model()
    print(f"✅ Model saved to: {model_path}")
    
    try:
        # Parse the model
        print("\n🔍 Parsing model...")
        parser = ModelParser()
        analysis = parser.parse_model(model_path)
        
        print("📊 Model Analysis Results:")
        print(f"  Format: {analysis['format']}")
        print(f"  Model Type: {analysis['model_type']}")
        print(f"  Parameters: {analysis['num_parameters']:,}")
        print(f"  Layers: {analysis['layer_count']}")
        print(f"  File Size: {analysis['file_size'] / (1024*1024):.2f} MB")
        
        # Get conversion recommendations
        print("\n💡 Getting conversion recommendations...")
        converter = ModelConverter()
        recommendations = converter.get_conversion_recommendations(analysis)
        
        print("🎯 Recommendations:")
        for key, value in recommendations.items():
            print(f"  {key}: {value}")
        
        # Convert to ONNX
        print("\n🔄 Converting to ONNX...")
        onnx_result = converter.convert_model(
            model_path, 
            'onnx', 
            optimization_level='medium'
        )
        
        if onnx_result['success']:
            print("✅ ONNX conversion successful!")
            print(f"  Output: {onnx_result['output_path']}")
            print(f"  Conversion time: {onnx_result['conversion_time']:.2f}s")
            print(f"  Size reduction: {onnx_result['file_size_reduction']:.1f}%")
        else:
            print(f"❌ ONNX conversion failed: {onnx_result['error']}")
        
        # Convert to TensorFlow Lite
        print("\n🔄 Converting to TensorFlow Lite...")
        tflite_result = converter.convert_model(
            model_path, 
            'tflite', 
            optimization_level='medium'
        )
        
        if tflite_result['success']:
            print("✅ TensorFlow Lite conversion successful!")
            print(f"  Output: {tflite_result['output_path']}")
            print(f"  Conversion time: {tflite_result['conversion_time']:.2f}s")
            print(f"  Size reduction: {tflite_result['file_size_reduction']:.1f}%")
        else:
            print(f"❌ TensorFlow Lite conversion failed: {tflite_result['error']}")
        
        print("\n🎉 Example completed successfully!")
        print("\nTo run the web interface:")
        print("  streamlit run app.py")
        
    except Exception as e:
        print(f"❌ Error during demonstration: {str(e)}")
    
    finally:
        # Cleanup
        try:
            os.unlink(model_path)
            print(f"\n🧹 Cleaned up temporary file: {model_path}")
        except:
            pass

if __name__ == "__main__":
    main()

