import streamlit as st
import os
import tempfile
from pathlib import Path
import time
from typing import Dict, Any, List

try:
    from model_parser import ModelParser
    from model_converter import ModelConverter
except ImportError as e:
    st.error(f"ML libraries not installed: {e}")
    st.info("Please install ML frameworks: pip install torch tensorflow onnx")
    st.stop()

from utils.file_handler import (
    save_uploaded_file, cleanup_temp_files, format_file_size, 
    validate_model_file, create_download_zip
)
from utils.visualization import (
    display_model_summary, display_conversion_results,
    create_model_size_comparison, create_optimization_recommendations_chart
)

st.set_page_config(
    page_title="AI Model Edge Converter",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #667eea;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
    }
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #f5c6cb;
    }
    .info-box {
        background-color: #d1ecf1;
        color: #0c5460;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #bee5eb;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def check_dependencies():
    available = {}
    try:
        import torch
        available['pytorch'] = True
    except ImportError:
        available['pytorch'] = False
    try:
        import tensorflow as tf
        available['tensorflow'] = True
    except ImportError:
        available['tensorflow'] = False
    try:
        import onnx
        available['onnx'] = True
    except ImportError:
        available['onnx'] = False
    try:
        import coremltools
        available['coreml'] = True
    except ImportError:
        available['coreml'] = False
    return available

def main():
    st.markdown('<h1 class="main-header">AI Model Edge Converter</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <p style="font-size: 1.2rem; color: #666;">
            Convert your AI models to formats optimized for edge devices like Raspberry Pi
        </p>
    </div>
    """, unsafe_allow_html=True)

    dependencies = check_dependencies()

    st.markdown("### Framework Status")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.success("PyTorch") if dependencies['pytorch'] else st.error("PyTorch")
    with col2:
        st.success("TensorFlow") if dependencies['tensorflow'] else st.error("TensorFlow")
    with col3:
        st.success("ONNX") if dependencies['onnx'] else st.error("ONNX")
    with col4:
        st.success("CoreML") if dependencies['coreml'] else st.error("CoreML")

    if not any(dependencies.values()):
        st.markdown("""
        <div class="info-box">
            <h4>Installation Required</h4>
            <p>To use full functionality, install ML frameworks:</p>
            <pre><code>pip install torch torchvision tensorflow onnx coremltools</code></pre>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### Demo Mode")
        st.info("""
        You're currently in demo mode. Some features are limited.
        """)

    if 'uploaded_file_path' not in st.session_state:
        st.session_state.uploaded_file_path = None
    if 'model_analysis' not in st.session_state:
        st.session_state.model_analysis = None
    if 'conversion_results' not in st.session_state:
        st.session_state.conversion_results = []
    if 'temp_files' not in st.session_state:
        st.session_state.temp_files = []

    with st.sidebar:
        st.header("Upload Model")
        uploaded_file = st.file_uploader(
            "Choose a model file",
            type=['pth', 'pt', 'h5', 'keras', 'pb', 'onnx']
        )
        
        if uploaded_file:
            validation = validate_model_file(uploaded_file)
            if validation['valid']:
                st.success(f"Valid {validation['file_type']} file")
                st.info(f"Size: {format_file_size(validation['file_size'])}")
                if st.button("Process Model", type="primary"):
                    with st.spinner("Processing model..."):
                        file_path = save_uploaded_file(uploaded_file)
                        if file_path:
                            st.session_state.uploaded_file_path = file_path
                            st.session_state.temp_files.append(file_path)
                            try:
                                parser = ModelParser()
                                analysis = parser.parse_model(file_path)
                                st.session_state.model_analysis = analysis
                                st.success("Model analyzed successfully.")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error analyzing model: {str(e)}")
            else:
                st.error(validation['error'])

        if st.button("Clear All", type="secondary"):
            cleanup_temp_files(st.session_state.temp_files)
            st.session_state.uploaded_file_path = None
            st.session_state.model_analysis = None
            st.session_state.conversion_results = []
            st.session_state.temp_files = []
            st.rerun()

    if st.session_state.model_analysis is None:
        st.markdown("""
        ## Welcome to AI Model Edge Converter

        Convert AI models for edge devices such as Raspberry Pi, phones, and other constrained systems.

        ### Features
        - Multi-format support (PyTorch, TensorFlow, ONNX)
        - Edge optimization for ONNX, TensorFlow Lite, and CoreML
        - Detailed model analysis and conversion metrics
        """)

        if not any(dependencies.values()):
            st.markdown("""
            ### Installation Guide
            ```bash
            pip install torch torchvision tensorflow keras onnx coremltools
            ```
            """)
    else:
        analysis = st.session_state.model_analysis
        display_model_summary(analysis)
        st.divider()

        st.subheader("Model Conversion")
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("#### Conversion Settings")
            output_formats = {'ONNX': 'onnx', 'TensorFlow Lite': 'tflite', 'CoreML': 'coreml'}
            selected_format = st.selectbox("Output format:", list(output_formats.keys()))
            optimization_level = st.selectbox("Optimization level:", ['default', 'medium', 'high'])
            convert_all = st.checkbox("Convert to all formats")

        with col2:
            try:
                converter = ModelConverter()
                recommendations = converter.get_conversion_recommendations(analysis)
                st.markdown("#### Recommendations")
                if recommendations:
                    st.info(f"Format: {recommendations.get('format', 'Unknown')}")
                    st.info(f"Reason: {recommendations.get('reason', 'No reason provided')}")
                    st.info(f"Optimization: {recommendations.get('optimization', 'default')}")
            except Exception as e:
                st.warning(f"Recommendations unavailable: {str(e)}")

        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("Convert Model", type="primary"):
                if st.session_state.uploaded_file_path:
                    with st.spinner("Converting..."):
                        try:
                            converter = ModelConverter()
                            results = []
                            if convert_all:
                                for fmt in ['onnx', 'tflite', 'coreml']:
                                    try:
                                        result = converter.convert_model(
                                            st.session_state.uploaded_file_path,
                                            fmt,
                                            optimization_level=optimization_level
                                        )
                                        results.append(result)
                                        if result['success'] and os.path.exists(result['output_path']):
                                            st.session_state.temp_files.append(result['output_path'])
                                    except Exception as e:
                                        results.append({'output_format': fmt, 'success': False, 'error': str(e)})
                                st.session_state.conversion_results = results
                            else:
                                result = converter.convert_model(
                                    st.session_state.uploaded_file_path,
                                    output_formats[selected_format],
                                    optimization_level=optimization_level
                                )
                                st.session_state.conversion_results = [result]
                                if result['success'] and os.path.exists(result['output_path']):
                                    st.session_state.temp_files.append(result['output_path'])
                            st.success("Conversion completed.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Conversion failed: {str(e)}")
        with c2:
            if st.button("Show Recommendations"):
                try:
                    converter = ModelConverter()
                    rec = converter.get_conversion_recommendations(analysis)
                    if rec:
                        fig = create_optimization_recommendations_chart(rec)
                        st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Could not generate recommendations: {str(e)}")
        with c3:
            if st.button("Clear Results"):
                st.session_state.conversion_results = []
                st.rerun()

        if st.session_state.conversion_results:
            st.divider()
            display_conversion_results(st.session_state.conversion_results)
            st.subheader("Download Converted Models")

            successful = [
                r for r in st.session_state.conversion_results 
                if r.get('success') and os.path.exists(r.get('output_path', ''))
            ]

            if successful:
                if len(successful) == 1:
                    r = successful[0]
                    with open(r['output_path'], 'rb') as f:
                        st.download_button(
                            label=f"Download {r['output_format'].upper()} Model",
                            data=f.read(),
                            file_name=f"converted_model.{r['output_format']}",
                            mime="application/octet-stream"
                        )
                else:
                    files = {f"model_{r['output_format']}.{r['output_format']}": r['output_path'] for r in successful}
                    zip_path = create_download_zip(files, "converted_models.zip")
                    st.session_state.temp_files.append(zip_path)
                    with open(zip_path, 'rb') as f:
                        st.download_button(
                            label="Download All Models (ZIP)",
                            data=f.read(),
                            file_name="converted_models.zip",
                            mime="application/zip"
                        )
                if len(successful) > 1:
                    st.subheader("Size Comparison")
                    orig = analysis.get('file_size', 0)
                    sizes = {r['output_format']: os.path.getsize(r['output_path']) for r in successful}
                    fig = create_model_size_comparison(orig, sizes)
                    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        <p>AI Model Edge Converter | Built with Streamlit</p>
        <p>Convert AI models for edge deployment with ease</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
