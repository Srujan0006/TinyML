import os
import argparse
import torch
import onnx
import tensorflow as tf
from onnxruntime.quantization import quantize_dynamic, QuantType


class ModelConverter:
    def _init_(self, model_path, output_dir="converted", target="tflite", quantize=None):
        self.model_path = model_path
        self.output_dir = output_dir
        self.target = target.lower()
        self.quantize = quantize
        os.makedirs(self.output_dir, exist_ok=True)

    def convert(self):
        ext = os.path.splitext(self.model_path)[1].lower()

        if ext in [".h5", ".pb"]:
            return self._convert_tf_model()

        elif ext in [".pt", ".pth"]:
            return self._convert_pytorch_to_onnx_then_tflite()

        elif ext == ".onnx":
            return self._convert_onnx_model()

        else:
            raise ValueError(f"Unsupported model format: {ext}")

    # ----------------- #
    # TensorFlow → TFLite
    # ----------------- #
    def _convert_tf_model(self):
        model = tf.keras.models.load_model(self.model_path)
        converter = tf.lite.TFLiteConverter.from_keras_model(model)

        if self.quantize == "float16":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
        elif self.quantize == "int8":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]

        tflite_model = converter.convert()
        out_path = os.path.join(self.output_dir, "model.tflite")
        with open(out_path, "wb") as f:
            f.write(tflite_model)

        print(f"✅ TensorFlow → TFLite conversion successful: {out_path}")
        return out_path

    # ----------------- #
    # PyTorch → ONNX → TFLite
    # ----------------- #
    def _convert_pytorch_to_onnx_then_tflite(self):
        print("⏳ Loading PyTorch model...")
        model = torch.load(self.model_path, map_location="cpu")
        model.eval()

        dummy_input = torch.randn(1, 3, 224, 224)
        onnx_path = os.path.join(self.output_dir, "model.onnx")

        print("🔄 Exporting PyTorch → ONNX...")
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=17,
            input_names=["input"],
            output_names=["output"]
        )

        print(f"✅ PyTorch → ONNX saved at: {onnx_path}")

        if self.target == "onnx":
            return onnx_path
        else:
            return self._convert_onnx_to_tflite(onnx_path)

    # ----------------- #
    # ONNX → (quantized) ONNX
    # ----------------- #
    def _convert_onnx_model(self):
        model = onnx.load(self.model_path)
        onnx.checker.check_model(model)

        if self.quantize:
            print("⚙ Applying ONNX dynamic quantization...")
            out_path = os.path.join(self.output_dir, "model_quantized.onnx")
            quant_type = QuantType.QInt8 if self.quantize == "int8" else QuantType.QUInt8
            quantize_dynamic(self.model_path, out_path, weight_type=quant_type)
            print(f"✅ Quantized ONNX model saved at: {out_path}")
            return out_path

        print("✅ ONNX model verified successfully.")
        return self.model_path

    # ----------------- #
    # ONNX → TFLite (via TensorFlow)
    # ----------------- #
    def _convert_onnx_to_tflite(self, onnx_path):
        import onnx_tf.backend as backend
        print("🔁 Converting ONNX → TensorFlow graph...")
        tf_rep = backend.prepare(onnx.load(onnx_path))
        tf_path = os.path.join(self.output_dir, "temp_tf_model")
        tf_rep.export_graph(tf_path)

        print("⚙ Converting TensorFlow → TFLite...")
        converter = tf.lite.TFLiteConverter.from_saved_model(tf_path)

        if self.quantize == "float16":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
        elif self.quantize == "int8":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]

        tflite_model = converter.convert()
        out_path = os.path.join(self.output_dir, "model.tflite")

        with open(out_path, "wb") as f:
            f.write(tflite_model)

        print(f"✅ ONNX → TensorFlow → TFLite complete: {out_path}")
        return out_path


# ----------------- #
# CLI Entry Point
# ----------------- #
def main():
    parser = argparse.ArgumentParser(description="Universal ML Model Converter")
    parser.add_argument("model_path", help="Path to model (.pt, .onnx, .h5, .pb)")
    parser.add_argument("--output", default="converted", help="Output directory")
    parser.add_argument("--target", default="tflite", choices=["tflite", "onnx"], help="Target format")
    parser.add_argument("--quantize", choices=["float16", "int8"], help="Quantization type")
    args = parser.parse_args()

    converter = ModelConverter(
        model_path=args.model_path,
        output_dir=args.output,
        target=args.target,
        quantize=args.quantize,
    )
    converter.convert()


if _name_ == "_main_":
    main()