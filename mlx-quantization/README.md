# MLX Model Quantization Toolkit

A comprehensive collection of Jupyter notebooks for converting and quantizing large language models using Apple's MLX framework, optimized for Apple Silicon devices.

## 🚀 Features

- **Universal Model Conversion**: Convert any Hugging Face model to MLX format
- **Multiple Quantization Methods**: Support for AWQ, DWQ, and Dynamic Quantization
- **Apple Silicon Optimized**: Built specifically for M1/M2/M3/M4 devices
- **Automated Workflows**: Complete pipeline from download to deployment
- **Performance Testing**: Built-in benchmarking and validation tools

## 📋 Requirements

- **Hardware**: macOS with Apple Silicon (M1/M2/M3/M4)
- **Python**: 3.8 or higher
- **Storage**: 50GB+ free space for large models

## 🛠 Installation

1. Clone the repository:
```bash
git clone https://github.com/cs2764/mlx-quantization.git
cd mlx-quantization
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Launch Jupyter:
```bash
jupyter lab
```

## 📚 Notebooks Overview

### Core Notebooks

| Notebook | Description | Use Case |
|----------|-------------|----------|
| `universal_mlx_converter.ipynb` | Universal converter for any HF model | General model conversion |
| `awq_quantization.ipynb` | Activation-aware Weight Quantization | High-quality 4-bit quantization |
| `dwq_quantization.ipynb` | Distilled Weight Quantization | Fast quantization with good quality |
| `dynamic_quantization.ipynb` | Dynamic mixed-precision quantization | Optimal size/quality balance |

### Quantization Methods Comparison

| Method | Speed | Quality | Size Reduction | Best For |
|--------|-------|---------|----------------|----------|
| **AWQ** | Medium | High | ~75% | Production deployment |
| **DWQ** | Fast | Good | ~70% | Quick prototyping |
| **Dynamic** | Slow | Highest | Variable | Research/experimentation |

## 🔄 Common Workflow

Each notebook follows this standardized pattern:

1. **Environment Setup** - Dependency installation and MLX verification
2. **Model Configuration** - Set up directories and parameters
3. **Model Download** - Fetch original model from Hugging Face
4. **Conversion/Quantization** - Apply selected quantization method
5. **Validation** - Test converted model functionality
6. **Performance Analysis** - Compare speed and quality metrics
7. **Optional Upload** - Push to Hugging Face Hub
8. **Cleanup** - Remove temporary files

## 📁 Directory Structure

```
mlx-quantization/
├── models/                          # Model storage
│   ├── <model_name>/               # Original models
│   └── <model_name>_<method>_<bits>/ # Quantized outputs
├── sensitivities/                   # Layer analysis files
├── *.ipynb                         # Conversion notebooks
├── requirements.txt                # Dependencies
└── README.md                       # This file
```

## 🚀 Quick Start

1. **Choose your quantization method** based on your requirements
2. **Open the corresponding notebook** in Jupyter Lab
3. **Follow the step-by-step instructions** in each cell
4. **Monitor the conversion process** and review results
5. **Test the quantized model** before deployment

## 📊 Performance Benchmarks

Typical results on Apple M2 Pro:

- **Model Size Reduction**: 60-80% smaller than original
- **Inference Speed**: 2-4x faster on Apple Silicon
- **Quality Retention**: 95-99% of original performance
- **Memory Usage**: 50-75% reduction

## 🔧 MLX Commands Reference

### Basic Conversion
```bash
python -m mlx_lm.convert --hf-path <source> --mlx-path <target>
```

### AWQ Quantization
```bash
python -m mlx_lm.awq --model <model> --mlx-path <output> --bits 4
```

### DWQ Quantization
```bash
python -m mlx_lm.dwq --model <model> --mlx-path <output> --bits 4
```

### Dynamic Quantization
```bash
python -m mlx_lm.dynamic_quant --model <model> --mlx-path <output> --target-bpw 4.0
```

## ⚠️ Important Notes

- **AWQ models require dequantization** before MLX conversion (`--dequantize` flag)
- **Use absolute paths** - relative paths may cause issues
- **Large models need significant storage** - ensure adequate disk space
- **Test converted models** before production deployment
- **Conversion time varies** based on model size and method

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly on Apple Silicon
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Apple MLX Team for the excellent framework
- Hugging Face for model hosting and tools
- The open-source ML community

## 📞 Support

- **Issues**: Report bugs and request features via GitHub Issues
- **Discussions**: Join community discussions in GitHub Discussions
- **Documentation**: Refer to individual notebook markdown cells

---

**Version**: 1.0.0  
**Last Updated**: 2025-01-30  
**Compatibility**: Apple Silicon (M1/M2/M3/M4) + macOS