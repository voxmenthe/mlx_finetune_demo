import argparse
import subprocess
import sys
from pathlib import Path

# Add mlx-lm to the Python path
# Assumes the script is run from the project root or src/finetuning
project_root = Path(__file__).resolve().parents[2]
mlx_lm_path = project_root / "mlx-lm"
sys.path.insert(0, str(mlx_lm_path))

def download_and_convert(hf_repo_id: str, output_dir_base: Path):
    """
    Downloads a model from Hugging Face and converts it to MLX format.

    Args:
        hf_repo_id: The Hugging Face repository ID (e.g., 'Qwen/Qwen3-14B').
        output_dir_base: The base directory to save the converted MLX model within.
                         A model-specific subdirectory will be created here.
    """
    # Ensure the base directory exists
    output_dir_base.mkdir(parents=True, exist_ok=True)

    # Construct the final model-specific path
    model_name = hf_repo_id.split("/")[-1] + "-mlx" 
    final_model_path = output_dir_base / model_name

    print(f"Converting {hf_repo_id} to MLX format...")
    print(f"Final output directory: {final_model_path}")

    # Check if the final path *already* exists before attempting conversion
    if final_model_path.exists():
        print(f"Error: Target directory {final_model_path} already exists.")
        print("Please remove it or specify a different base directory if conversion is needed again.")
        sys.exit(1)

    try:
        # Use our custom conversion script
        command = [
            sys.executable,  # Use the current Python interpreter
            str(project_root / "src" / "finetuning" / "convert_qwen3_custom.py"), # Path to custom script
            "--hf-path",
            hf_repo_id,
            "--mlx-path",
            str(final_model_path),
            # Potentially add "--dtype" if needed, default is float16
            # "--dtype", "bfloat16"
        ]
        
        # Note: Ensure transformers and huggingface_hub are installed:
        # pip install transformers huggingface_hub sentencepiece tiktoken

        print(f"Running command: {' '.join(command)}")
        subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"Successfully converted {hf_repo_id} and saved to {final_model_path}")

    except subprocess.CalledProcessError as e:
        print(f"Error during conversion:")
        print(f"Command: {' '.join(e.cmd)}")
        print(f"Return code: {e.returncode}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and convert a Hugging Face model to MLX format."
    )
    parser.add_argument(
        "--hf-repo-id",
        type=str,
        default="Qwen/Qwen3-14B",
        help="Hugging Face repository ID of the model to download and convert.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("mlx_models"),  # Base directory
        help="Base directory to save the converted MLX model (model-specific subfolder will be created).",
    )
    args = parser.parse_args()

    # Ensure output base dir is relative to project root if not absolute
    if not args.output_dir.is_absolute():
        args.output_dir = project_root / args.output_dir
        
    download_and_convert(args.hf_repo_id, args.output_dir) 