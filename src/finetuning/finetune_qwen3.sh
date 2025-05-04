#!/bin/bash

# Script to fine-tune Qwen3 models using mlx_lm.lora

# --- Configuration ---

# Default fine-tuning type
FINE_TUNE_TYPE="dora" # "lora" # Default: "lora". Can be overridden with --tune-type flag ("lora", "dora", "full")
CONFIG_PATH="src/finetuning/lora_config.yaml"         # Optional path to a YAML config file for detailed LoRA/optimizer settings

# --- Argument Parsing ---
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --tune-type) FINE_TUNE_TYPE="$2"; shift ;;
        --config) CONFIG_PATH="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Validate FINE_TUNE_TYPE
if [[ "$FINE_TUNE_TYPE" != "lora" && "$FINE_TUNE_TYPE" != "dora" && "$FINE_TUNE_TYPE" != "full" ]]; then
    echo "Error: Invalid --tune-type specified: '$FINE_TUNE_TYPE'. Must be 'lora', 'dora', or 'full'."
    exit 1
fi
echo "Using Fine-Tuning Type: $FINE_TUNE_TYPE"

# !!! IMPORTANT: Set this path to your *local* converted MLX model directory !!!
# Example: ../../mlx_models/Qwen3-4B-mlx
MODEL_PATH="mlx_models/Qwen3-14B-mlx"

# !!! IMPORTANT: Path to the directory containing your training data files !!!
# This directory MUST contain files named exactly 'train.jsonl' and 'valid.jsonl'.
# Optionally, it can contain 'test.jsonl' if RUN_TEST=true.
# This path is relative to where you run the script from.
# See mlx-lm/mlx_lm/LORA.md#data for format details.
DATA_PATH="DATA/SACREDHUNGER"

# Directory to save the LoRA adapters (relative to where you run the script)
ADAPTER_PATH="ADAPTERS/qwen3_14b_${FINE_TUNE_TYPE}_sacredhunger_multi" # _atkm_multi" # Example, adjust as needed

# Training parameters (adjust as needed)
ITERS=5600          # Number of training iterations
BATCH_SIZE=1      # Batch size (reduce if hitting memory limits)
LEARNING_RATE=1e-5 # Learning rate
SAVE_EVERY=100     # Save adapter weights every N iterations
NUM_LAYERS=-1 # 16      # Number of layers to apply LoRA to (-1 for all)
MAX_SEQ_LENGTH=3827 # Max sequence length model can handle

# Evaluation parameters (optional)
RUN_TEST=false     # Set to true to run evaluation on test.jsonl after training
VAL_BATCHES=25     # Number of validation batches during training (-1 for full validation set)
TEST_BATCHES=100   # Number of test batches if RUN_TEST=true (-1 for full test set)

# --- Safety Checks ---
if [ "$MODEL_PATH" == "../../mlx_models/<your_model_name>-mlx" ]; then
    echo "Error: Please set the MODEL_PATH variable in the script to your specific MLX model directory."
    exit 1
fi

# Basic check if DATA_PATH directory exists
if [ ! -d "$DATA_PATH" ]; then
    echo "Error: Data directory '$DATA_PATH' not found."
    exit 1
fi

if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model directory '$MODEL_PATH' not found."
    echo "Did you run a conversion script?"
    exit 1
fi

# Check for the required files within DATA_PATH
if [ ! -f "$DATA_PATH/train.jsonl" ] || [ ! -f "$DATA_PATH/valid.jsonl" ]; then
    echo "Error: Could not find required 'train.jsonl' or 'valid.jsonl' in directory '$DATA_PATH'."
    echo "Please ensure your training files are named correctly and placed in this directory."
    exit 1
fi

# Add mlx-lm to Python path (adjust if your structure differs)
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PROJECT_ROOT=$( realpath "$SCRIPT_DIR/../.." )
MLX_LM_PATH="$PROJECT_ROOT/mlx-lm"
export PYTHONPATH="$PYTHONPATH:$MLX_LM_PATH"

echo "Using MLX Model: $MODEL_PATH"
echo "Using Data Dir : $DATA_PATH"
echo "Saving Adapters to: $ADAPTER_PATH"
echo "PYTHONPATH set to: $PYTHONPATH"

# --- Build Command --- 
# Uses the DATA_PATH directory as input for the --data argument
CMD=(
    "python"
    "-m"
    "mlx_lm"
    "lora"
    "--model" "$MODEL_PATH"
    "--train"
    "--data" "$DATA_PATH"
    "--adapter-path" "$ADAPTER_PATH"
    "--iters" "$ITERS"
    "--batch-size" "$BATCH_SIZE"
    # "--learning-rate" "$LEARNING_RATE" # Let config file override this if provided
    "--save-every" "$SAVE_EVERY"
    "--num-layers" "$NUM_LAYERS"
    "--max-seq-length" "$MAX_SEQ_LENGTH"
    "--val-batches" "$VAL_BATCHES"
    "--fine-tune-type" "$FINE_TUNE_TYPE"
    # Add optional flags
    # "--grad-checkpoint" # Use gradient checkpointing to save memory
    # "--mask-prompt"     # Ignore prompt tokens in loss calculation
)

# Add config file if provided
if [ -n "$CONFIG_PATH" ]; then
  if [ ! -f "$CONFIG_PATH" ]; then
      echo "Error: Config file specified but not found: '$CONFIG_PATH'"
      exit 1
  fi
  echo "Using config file: $CONFIG_PATH"
  CMD+=("--config" "$CONFIG_PATH")
fi

if [ "$RUN_TEST" = true ]; then
    if [ ! -f "$DATA_PATH/test.jsonl" ]; then # Check in DATA_PATH directory
        echo "Warning: RUN_TEST is true but 'test.jsonl' not found in '$DATA_PATH'. Skipping test evaluation."
    else
        CMD+=("--test" "--test-batches" "$TEST_BATCHES")
    fi
fi

# --- Run Training --- 
echo "Running command:"
printf "%s " "${CMD[@]}"
echo "\n"

"${CMD[@]}"

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "Fine-tuning completed successfully."
    echo "Adapters saved in: $ADAPTER_PATH"
else
    echo "Fine-tuning failed with exit code $EXIT_CODE."
fi

exit $EXIT_CODE 