import argparse
import json
import subprocess
import os
import sys
from pathlib import Path

# Add the project root to the Python path to allow importing from src
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

def run_inference(model_path, adapter_path, prompt_text, temp, top_p, rep_penalty):
    """Runs the inference script with the given parameters."""
    command = [
        "python",
        str(project_root / "src/inference/generate_qwen3.py"),
        "--model-path", model_path,
        "--prompt", "-", # Indicate prompt comes from stdin
        "--temp", str(temp),
        "--top-p", str(top_p),
        "--repetition-penalty", str(rep_penalty),
        # Add other necessary args like max_tokens if needed
        # "--max-tokens", "512",
    ]
    if adapter_path:
        command.extend(["--adapter-path", adapter_path])

    try:
        # Use subprocess.run to execute the command
        # Pass the prompt via stdin
        result = subprocess.run(
            command,
            input=prompt_text,
            text=True,
            capture_output=True,
            check=True, # Raise an exception if the command fails
            encoding='utf-8'
        )
        return result.stdout.strip()
    except FileNotFoundError:
        print(f"Error: The script 'src/inference/generate_qwen3.py' was not found.", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"Error during inference subprocess execution:", file=sys.stderr)
        print(f"Command: {' '.join(e.cmd)}", file=sys.stderr)
        print(f"Return code: {e.returncode}", file=sys.stderr)
        print(f"Stderr: {e.stderr}", file=sys.stderr)
        print(f"Stdout: {e.stdout}", file=sys.stderr)
        return None # Or re-raise the exception if preferred

def main():
    parser = argparse.ArgumentParser(description="Run batch evaluations using generate_qwen3.py")
    parser.add_argument("--model-path", required=True, help="Path to the MLX model directory.")
    parser.add_argument("--adapter-path", default=None, help="Path to the adapter directory (optional).")
    parser.add_argument("--valid-jsonl-path", required=True, help="Path to the validation JSONL file.")
    parser.add_argument("--output-dir", default="eval_outputs", help="Directory to save evaluation outputs.")
    parser.add_argument("--num-examples", type=int, default=10, help="Number of examples to run from the JSONL file.")
    parser.add_argument("--prompt-key", default="prompt", help="The key in the JSONL file containing the prompt text.")
    # Add generation parameters matching the inference script
    parser.add_argument("--temp", type=float, default=0.75, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p sampling.")
    parser.add_argument("--repetition-penalty", type=float, default=1.1, help="Repetition penalty.")

    args = parser.parse_args()

    # Validate paths
    model_path_obj = Path(args.model_path)
    valid_jsonl_path_obj = Path(args.valid_jsonl_path)
    output_dir_obj = Path(args.output_dir)
    adapter_path_obj = Path(args.adapter_path) if args.adapter_path else None

    if not model_path_obj.is_dir():
        print(f"Error: Model path '{args.model_path}' not found or not a directory.", file=sys.stderr)
        sys.exit(1)
    if args.adapter_path and not adapter_path_obj.is_dir():
         print(f"Error: Adapter path '{args.adapter_path}' not found or not a directory.", file=sys.stderr)
         sys.exit(1)
    if not valid_jsonl_path_obj.is_file():
        print(f"Error: Validation JSONL file '{args.valid_jsonl_path}' not found.", file=sys.stderr)
        sys.exit(1)

    # Create output directory
    output_dir_obj.mkdir(parents=True, exist_ok=True)
    output_file = output_dir_obj / "results.jsonl"

    print(f"Running evaluations for {args.num_examples} examples...")
    print(f"Model: {args.model_path}")
    if args.adapter_path:
        print(f"Adapter: {args.adapter_path}")
    print(f"Input: {args.valid_jsonl_path}")
    print(f"Output will be saved to: {output_file}")

    count = 0
    results = []
    try:
        with open(args.valid_jsonl_path, 'r', encoding='utf-8') as infile, \
             open(output_file, 'w', encoding='utf-8') as outfile:
            for line in infile:
                if count >= args.num_examples:
                    break
                try:
                    data = json.loads(line.strip())
                except json.JSONDecodeError:
                    print(f"Warning: Skipping invalid JSON line: {line.strip()}", file=sys.stderr)
                    continue

                if args.prompt_key not in data:
                    print(f"Warning: Prompt key '{args.prompt_key}' not found in JSON line: {line.strip()}. Skipping.", file=sys.stderr)
                    continue

                prompt = data[args.prompt_key]
                print(f"\nRunning example {count + 1}/{args.num_examples}...")
                # print(f"Prompt: {prompt[:100]}...") # Optionally print start of prompt

                generation = run_inference(
                    model_path=args.model_path,
                    adapter_path=args.adapter_path,
                    prompt_text=prompt,
                    temp=args.temp,
                    top_p=args.top_p,
                    rep_penalty=args.repetition_penalty
                )

                if generation is not None:
                    print(f"Generation successful.")
                    result_data = {
                        "prompt": prompt,
                        "generation": generation,
                        "original_data": data # Keep original data for reference
                    }
                    # Write result to output file immediately
                    outfile.write(json.dumps(result_data) + '\\n')
                    outfile.flush() # Ensure it's written in case of interruption
                    results.append(result_data)
                    count += 1
                else:
                     print(f"Generation failed for example {count + 1}.")
                     # Decide if you want to stop or continue
                     # break

    except FileNotFoundError:
        print(f"Error: Could not open validation file '{args.valid_jsonl_path}'", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"\nEvaluation complete. {count} results saved to {output_file}")

if __name__ == "__main__":
    main() 