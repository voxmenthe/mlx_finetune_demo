import argparse
import copy
import glob
import json
import shutil
import sys
from pathlib import Path
from typing import Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

# Add project root to path to find mlx_lm and src.copies
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

# Add mlx-lm path specifically for its imports
mlx_lm_path = project_root / "mlx-lm"
sys.path.insert(1, str(mlx_lm_path))

# Import necessary functions from original mlx-lm
from mlx_lm.utils import (
    fetch_from_hub,
    get_model_path,
    make_shards,
    save_config,
    save_weights,
    upload_to_hub,
    get_model_path
)
from mlx_lm.tokenizer_utils import load_tokenizer

# Import our *custom* Qwen3 model definition
# Need to ensure src/copies/models has an __init__.py if running as script directly
# Or rely on PYTHONPATH containing project root
try:
    from src.copies.models.qwen3 import Model, ModelArgs
except ModuleNotFoundError:
    print("Error: Could not import custom Qwen3 model from src.copies.models.qwen3")
    print("Ensure the project root is in PYTHONPATH or run this script as a module.")
    sys.exit(1)


def get_custom_model_classes(config: dict) -> Tuple[nn.Module, ModelArgs]:
    """Returns our custom Qwen3 model classes if model_type matches."""
    model_type = config.get("model_type")
    if model_type == "qwen3":
        return Model, ModelArgs
    else:
        # Fallback or error for other model types if needed
        raise ValueError(f"This custom converter only supports model_type 'qwen3', got {model_type}")


def custom_load_model(
    model_path: Path,
    lazy: bool = False,
    strict: bool = True,
) -> nn.Module:
    """Loads model using the custom qwen3 class.
    
    Replicates logic from mlx_lm.utils.load_model but uses get_custom_model_classes.
    """
    try:
        with open(model_path / "config.json", "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"Config file not found in {model_path}")
        # Fallback for older HF format
        try:
            with open(model_path / "params.json", "r") as f:
                config = json.load(f)
                config["model_type"] = config["architectures"][0].lower()
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Neither config.json nor params.json found in {model_path}"
            )

    weight_files = glob.glob(str(model_path / "*.safetensors"))
    if not weight_files:
        # Fallback for older pytorch format
        weight_files = glob.glob(str(model_path / "*.bin"))
    if not weight_files:
        raise FileNotFoundError(f"No weights found in {model_path}")

    weights = {}
    for wf in weight_files:
        weights.update(mx.load(wf))

    # --- Debug: Print initial dtypes ---
    print("\n--- Initial loaded weight dtypes (sample) ---")
    for k, v in list(weights.items())[:5]: # Print first 5
        print(f"{k}: {v.dtype}")
    if len(weights) > 5:
         for k, v in list(weights.items())[-5:]: # Print last 5
            print(f"{k}: {v.dtype}")
    print("---------------------------------------------")
    # ------------------------------------

    # Use our custom class getter
    model_class, model_args_class = get_custom_model_classes(config=config)

    model_args = model_args_class.from_dict(config)
    model = model_class(model_args)

    # The sanitize method is part of our custom Model class
    if hasattr(model, "sanitize"):
        print("Sanitizing weights using custom model...")
        weights = model.sanitize(weights)

    # Quantization is not expected/handled for FP8 conversion here
    if (quantization := config.get("quantization", None)) is not None:
         print("Warning: Found quantization config, but this custom script doesn't apply it.")

    model.load_weights(list(weights.items()), strict=strict)

    if not lazy:
        mx.eval(model.parameters())

    model.eval()
    return model, config


def custom_convert(
    hf_path: str,
    mlx_path: str = "mlx_model",
    dtype: str = "float16",
    upload_repo: str = None,
):
    """Converts HF Qwen3 model using the custom sanitize logic."""
    print(f"[INFO] Loading model from HF path: {hf_path}")
    model_path = get_model_path(hf_path)
    mlx_path = Path(mlx_path)
    mlx_path.mkdir(parents=True, exist_ok=True)

    # Load model using our custom loader which calls the custom sanitize
    model, config = custom_load_model(model_path, lazy=True)
    
    # Load tokenizer using standard mlx-lm function
    tokenizer = load_tokenizer(model_path)

    weights = dict(tree_flatten(model.parameters()))
    
    # --- Debug: Print dtypes before astype conversion ---
    print("\n--- Weight dtypes before astype() (sample) ---")
    param_list = list(weights.items())
    for k, v in param_list[:5]: # Print first 5
        print(f"{k}: {v.dtype}")
    if len(param_list) > 5:
        for k, v in param_list[-5:]: # Print last 5
            print(f"{k}: {v.dtype}")
    print("---------------------------------------------")
    # --------------------------------------------------
    
    dtype = getattr(mx, dtype)
    print(f"[INFO] Casting weights to target dtype: {dtype}")
    weights = {k: v.astype(dtype) for k, v in weights.items()}

    print("[INFO] Saving weights")
    save_weights(mlx_path, weights)

    # Save tokenizer
    shutil.copyfile(
        str(model_path / "tokenizer.json"), str(mlx_path / "tokenizer.json")
    )
    if (model_path / "vocab.json").is_file(): # Qwen specific?
        shutil.copyfile(
            str(model_path / "vocab.json"), str(mlx_path / "vocab.json")
        ) 
    if (model_path / "merges.txt").is_file(): # Qwen specific?
         shutil.copyfile(
            str(model_path / "merges.txt"), str(mlx_path / "merges.txt")
        ) 
    if (model_path / "tokenizer_config.json").is_file():
        shutil.copyfile(
            str(model_path / "tokenizer_config.json"),
            str(mlx_path / "tokenizer_config.json"),
        )
        
    # Save config -- make sure dtype is updated if changed
    config["mlx_lm"] = {"dtype": str(dtype).split(".")[-1]}
    save_config(config, config_path=mlx_path / "config.json")

    if upload_repo:
        upload_to_hub(mlx_path, upload_repo, hf_path)
        
    print(f"[INFO] Conversion complete. Model saved to: {mlx_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert Qwen3 HF weights to MLX format using custom sanitize."
    )
    parser.add_argument(
        "--hf-path",
        type=str,
        required=True,
        help="Path to the Hugging Face model directory or repo ID (e.g., Qwen/Qwen3-14B).",
    )
    parser.add_argument(
        "--mlx-path",
        type=str,
        required=True,
        help="Path to save the MLX model (e.g., mlx_models/Qwen3-14B-mlx). This path should NOT exist.",
    )
    parser.add_argument(
        "--dtype",
        help="Type to save the parameters, ignored for quantized models. Options: float16, bfloat16, float32.",
        type=str,
        default="float16", # Keep original precision if possible, but HF FP8 might load as float16/32 initially
    )
    parser.add_argument(
        "--upload-repo",
        help="The Hugging Face repo to upload the model to.",
        type=str,
        default=None,
    )

    args = parser.parse_args()
    
    # Basic check for mlx_path existence (convert expects it not to exist)
    if Path(args.mlx_path).exists():
        print(f"Error: Output path {args.mlx_path} already exists. Please remove it first.")
        sys.exit(1)
        
    custom_convert(**vars(args)) 