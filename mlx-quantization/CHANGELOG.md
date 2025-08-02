# Changelog

All notable changes to the MLX Quantization Toolkit will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-01-30

### Added
- Initial release of MLX Quantization Toolkit
- Universal MLX converter for any Hugging Face model
- DeepSeek-R1 AWQ to MLX conversion support
- AWQ (Activation-aware Weight Quantization) implementation
- DWQ (Distilled Weight Quantization) implementation  
- Dynamic mixed-precision quantization support
- Comprehensive Jupyter notebook workflow
- Automated model download and upload to Hugging Face
- Performance benchmarking and validation tools
- Apple Silicon optimization for M1/M2/M3/M4 devices
- Robust error handling and fallback mechanisms
- Complete documentation and usage examples

### Features
- **5 Quantization Methods**: Universal, DeepSeek-R1, AWQ, DWQ, Dynamic
- **Apple Silicon Optimized**: Native MLX framework integration
- **Automated Workflows**: End-to-end conversion pipelines
- **Performance Testing**: Built-in benchmarking tools
- **Hugging Face Integration**: Seamless model upload/download
- **Error Recovery**: Multiple fallback conversion methods

### Technical Specifications
- **Python**: 3.8+ required
- **Hardware**: Apple Silicon (M1/M2/M3/M4) required
- **Storage**: 50GB+ recommended for large models
- **Memory**: 16GB+ RAM recommended
- **MLX Version**: 0.12.0+ supported

### Supported Models
- Any Hugging Face transformer model
- DeepSeek-R1 AWQ models (specialized support)
- Large language models up to 70B+ parameters
- Various architectures: Llama, Mistral, Qwen, etc.

### Performance Metrics
- Model size reduction: 60-80%
- Inference speed improvement: 2-4x on Apple Silicon
- Quality retention: 95-99% of original performance
- Memory usage reduction: 50-75%

### Documentation
- Comprehensive README with setup instructions
- Individual notebook documentation
- Usage examples and best practices
- Troubleshooting guide
- Performance benchmarking results