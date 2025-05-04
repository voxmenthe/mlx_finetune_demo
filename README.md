# 🚀 MLX Finetuning Demo Project

✨ A complete guide to setting up and running the MLX finetuning pipeline for your custom datasets

## 🛠️ Setup Instructions

1. **Create and activate virtual environment**
   ```bash
   python -m venv mlx_venv
   source mlx_venv/bin/activate  # Linux/Mac
   # OR
   mlx_venv\Scripts\activate    # Windows
   ```

2. **Install dependencies**
   ```bash
   sh project_setup.sh
   ```

3. **Prepare your dataset**
   - Place your source data file (long text format) in `/data/raw/`
   - Example dataset structure:
     ```
     data/
       raw/
         my_dataset.txt
     ```

## 🔄 Data Processing Pipeline

1. **Run semantic chunker**
   ```bash
   python src/data_processing/semantic_chunker.py \
     --book_path data/raw/your_book.txt \
     --target 480 \
     --output_path data/processed/chunks.json
   ```
   - `--target`: Target word count per chunk (default: 480)
   - Uses `lightonai/modernbert-embed-large` model by default

2. **Prepare training data**
   ```bash
   python src/data_processing/prepare_training_data.py \
     --input_files data/processed/chunks.json \
     --output_dir data/final \
     --train_ratio 0.85
   ```
   - Creates `train.jsonl` and `valid.jsonl` files
   - Each sample contains a prompt/continuation pair
   - Default 85/15 train/validation split

## ⚙️ Configuration

Edit `lora_config.yaml` with your settings:
```yaml
model_name: "bert-base-uncased"
lora_rank: 8
target_modules: ["query", "value"]
learning_rate: 3e-4
batch_size: 32
num_epochs: 10
```

## 🏋️ Training

Start finetuning:
```bash
sh src/finetuning/finetune_qwen3.sh \
  --tune-type dora \
  --config src/finetuning/lora_config.yaml
```

Key parameters (edit in script):
- `MODEL_PATH`: Path to MLX model directory
- `DATA_PATH`: Directory containing `train.jsonl` and `valid.jsonl`
- `ADAPTER_PATH`: Where to save adapters
- `ITERS`: Number of training iterations (default: 5600)
- `BATCH_SIZE`: Batch size (default: 1)

## 📊 Evaluation

Run evaluations:
```bash
python run_evaluations.py \
  --model-path mlx_models/Qwen3-14B-mlx \
  --adapter-path ADAPTERS/qwen3_14b_dora_sacredhunger_multi \
  --valid-jsonl-path data/final/valid.jsonl \
  --output-dir eval_outputs \
  --num-examples 50
```

Evaluation parameters:
- `--temp`: Sampling temperature (default: 0.75)
- `--top-p`: Top-p sampling (default: 0.95)
- `--repetition-penalty`: Penalty for repeated tokens (default: 1.1)

## 📌 Tips

- Monitor training with `tensorboard --logdir outputs/logs`
- For large datasets, consider using `--num_workers` in data preparation
- Adjust batch size based on your GPU memory

💡 For questions or issues, please open an issue in this repository!