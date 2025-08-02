# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains Jupyter notebooks for converting and quantizing large language models using Apple's MLX framework. The project focuses on optimizing models for Apple Silicon devices through various quantization techniques.

## Repository Structure

The repository is organized around Jupyter notebooks that handle different aspects of MLX model conversion and quantization:

- **deepseek_r1_mlx_conversion.ipynb**: Converts DeepSeek-R1 AWQ models to MLX format
- **universal_mlx_converter.ipynb**: Universal converter for any Hugging Face model to MLX format
- **dwq_quantization.ipynb**: Distilled Weight Quantization implementation
- **awq_quantization.ipynb**: Activation-aware Weight Quantization implementation  
- **dynamic_quantization.ipynb**: Dynamic quantization with mixed precision
- **models/**: Directory for storing downloaded and converted models

## Core Dependencies

All notebooks require these essential packages:
- `mlx-lm`: Apple's MLX language model framework
- `transformers`: Hugging Face transformers library
- `torch`: PyTorch (for compatibility)
- `huggingface_hub`: For model download/upload
- `datasets`, `accelerate`, `sentencepiece`, `protobuf`: Supporting libraries

## Common Workflow Pattern

Each notebook follows a consistent structure:
1. Environment setup and dependency installation
2. MLX import testing and verification
3. Model configuration and directory setup  
4. Original model download from Hugging Face
5. Model conversion/quantization using MLX tools
6. Converted model testing and validation
7. Optional performance comparison
8. Optional Hugging Face upload
9. Cleanup and summary

## MLX Conversion Commands

The project uses MLX command-line tools for conversions:

### Basic Conversion
```bash
python -m mlx_lm.convert --hf-path <source_dir> --mlx-path <target_dir>
```

### DWQ Quantization
```bash
python -m mlx_lm.dwq --model <model_path> --mlx-path <output_path> --bits 4 --num-samples 1024
```

### AWQ Quantization  
```bash
python -m mlx_lm.awq --model <model_path> --mlx-path <output_path> --bits 4 --num-samples 32
```

### Dynamic Quantization
```bash
python -m mlx_lm.dynamic_quant --model <model_path> --mlx-path <output_path> --target-bpw 4.0
```

## Environment Setup Requirements

**Critical**: This project requires macOS with Apple Silicon (M1/M2/M3/M4). The notebooks include specific handling for:
- numpy/gfortran library conflicts in JupyterLab Desktop
- MLX framework import verification
- Automatic package installation with error handling
- Kernel restart recommendations for import issues

## Model Storage Architecture

The project uses a standardized directory structure:
- `models/`: Root directory for all model storage
- `models/<model_name_sanitized>/`: Original downloaded models
- `models/<model_name>_<method>_<precision>/`: Quantized outputs
- `sensitivities/`: Layer sensitivity analysis files (for dynamic quantization)

## Error Handling Patterns

All notebooks implement robust error handling:
- Multiple conversion method attempts with fallbacks
- Comprehensive import testing before execution
- File existence checks and cleanup procedures
- Detailed error reporting with troubleshooting guidance

## Hugging Face Integration

The notebooks include full Hugging Face workflow:
- Secure token-based authentication
- Model download with resume capability
- Automatic model card generation
- Repository creation and file upload
- Upload verification and file listing

## Performance Testing

Each quantization method includes:
- Model loading and inference testing
- Multi-prompt validation
- Performance timing comparisons
- Size reduction calculations
- Quality evaluation options using standard datasets

## Important Notes

- AWQ models require dequantization before MLX conversion (`--dequantize` flag)
- Directory paths must be absolute, not relative
- Large models require significant disk space (50GB+ for full-size models)
- Model conversion can be time-intensive depending on model size
- Always test converted models before deployment or upload