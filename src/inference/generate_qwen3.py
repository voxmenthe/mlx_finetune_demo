
"""
Usage:

# WITH ADAPTER
cat temp_prompt.txt | python src/inference/generate_qwen3.py \
--model-path mlx_models/Qwen3-4B-mlx \
--adapter-path ADAPTERS/qwen3_4b_lora_sacredhunger \
--prompt "-" \
--repetition-penalty 1.1 \
--temp 0.75 \
--top-p 0.95 

# WITHOUT ADAPTER
cat temp_prompt.txt | python src/inference/generate_qwen3.py \
--model-path mlx_models/Qwen3-4B-mlx \
--prompt "-" \
--repetition-penalty 1.1 \
--temp 0.75 \
"""

import argparse
import sys
import time
from pathlib import Path

import mlx.core as mx

# Add mlx-lm to the Python path
# Assumes the script is run from the project root or src/inference
project_root = Path(__file__).resolve().parents[2]
mlx_lm_path = project_root / "mlx-lm"

if str(mlx_lm_path) not in sys.path:
    sys.path.insert(0, str(mlx_lm_path))

try:
    from mlx_lm.utils import load
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler, make_logits_processors
except ImportError as e:
    print("Error importing mlx_lm. Please ensure mlx-lm is installed and discoverable.")
    print(f"Attempted path: {mlx_lm_path}")
    print(f"PYTHONPATH: {sys.path}")
    print(f"ImportError: {e}")
    sys.exit(1)


def main(args):
    mx.random.seed(args.seed)

    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Error: Model path does not exist: {model_path}")
        print("Make sure you have downloaded and converted the model using a script like")
        print("src/finetuning/download_qwen3.py")
        sys.exit(1)

    adapter_path = args.adapter_path
    if adapter_path:
        adapter_path = Path(adapter_path)
        if not adapter_path.exists():
            print(f"Error: Adapter path does not exist: {adapter_path}")
            sys.exit(1)
        print(f"Loading model from {model_path} with adapter from {adapter_path}...")
    else:
        print(f"Loading model from {model_path}...")

    try:
        model, tokenizer = load(args.model_path, adapter_path=str(adapter_path) if adapter_path else None)
    except Exception as e:
        print(f"Error loading the model or adapter: {e}")
        sys.exit(1)
    
    print("Model loaded.")

    # Handle prompt input (stdin or argument)
    if args.prompt == "-":
        print("Reading prompt from stdin...")
        try:
            prompt_input = sys.stdin.read()
        except EOFError:
            print("Error: Reached end of input while reading from stdin.")
            sys.exit(1)
        if not prompt_input:
             print("Error: Received empty prompt from stdin.")
             sys.exit(1)
    else:
        # Replace escaped newlines/tabs if coming from command line
        prompt_input = args.prompt.replace("\\n", "\n").replace("\\t", "\t")

    # Prepare the prompt (apply chat template if requested and available)
    if args.use_chat_template and hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None:
        print("Applying chat template...")
        messages = []
        if args.system_prompt:
            messages.append({"role": "system", "content": args.system_prompt})
        messages.append({"role": "user", "content": prompt_input})
        
        try:
            prompt_str = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            # Encode the templated prompt, template likely includes BOS/EOS handling
            encoded_prompt = tokenizer.encode(prompt_str, add_special_tokens=False)
            print(f"Using templated prompt: {prompt_str[:200]}...") # Print start of templated prompt
        except Exception as e:
            print(f"Error applying chat template: {e}")
            print("Falling back to raw prompt encoding.")
            # Fallback if template application fails
            encoded_prompt = tokenizer.encode(prompt_input, add_special_tokens=True) # Use original input

    else:
        if not args.use_chat_template:
            print("Chat template disabled by user.")
        elif not hasattr(tokenizer, 'chat_template') or tokenizer.chat_template is None:
            print("No chat template found in tokenizer.")
        print("Encoding raw prompt...")
        # Encode raw prompt, assuming default BOS/EOS handling is desired
        encoded_prompt = tokenizer.encode(prompt_input, add_special_tokens=True) # Use original input
    
    encoded_prompt = mx.array(encoded_prompt)

    print("Generating response...")
    start_time = time.time()
    
    # Create the sampler instance
    sampler = make_sampler(temp=args.temp, top_p=args.top_p)
    
    # Create the logits processors list
    logits_processors = make_logits_processors(
        repetition_penalty=args.repetition_penalty,
        repetition_context_size=args.repetition_context_size
    )
    
    # Pass sampler and processors instead of individual args
    response = generate(
        model,
        tokenizer,
        prompt=encoded_prompt, # Use the processed prompt
        max_tokens=args.max_tokens,
        sampler=sampler,
        logits_processors=logits_processors,
        verbose=args.verbose  
    )
    
    # If verbose=False, generate returns the full string, otherwise it prints token by token and returns None
    if not args.verbose:
        print(response)
        
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"\n\nGeneration complete in {elapsed_time:.2f} seconds.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with a converted MLX model, optionally using LoRA adapters.")
    
    # --- Model and Prompt Arguments ---
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the directory containing the converted MLX model files (e.g., <project_root>/mlx_models/Qwen3-14B-mlx)."
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help="Optional path to the directory containing the trained LoRA adapter weights (`adapters.safetensors`) and config (`adapter_config.json`)."
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="The input prompt for the model, or '-' to read from stdin."
    )
    parser.add_argument(
        "--use-chat-template",
        action="store_true",
        default=True,
        help="Use the tokenizer's chat template if available. Set --no-use-chat-template to disable."
    )
    parser.add_argument(
        "--no-use-chat-template",
        action="store_false",
        dest="use_chat_template",
        help="Disable the use of the tokenizer's chat template."
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help="Optional system prompt to prepend when using the chat template."
    )
    
    # --- Generation Control Arguments ---
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=3200, #32768,
        help="Maximum number of tokens to generate. [Default: 3200]"
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=0.6,
        help=(
            "Sampling temperature. Controls randomness. Lower values (e.g., 0.1) make the output "
            "more deterministic, higher values (e.g., 1.0) make it more random. [Default: 0.6]"
        )
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help=(
            "Top-p (nucleus) sampling probability. Selects tokens from the smallest set whose cumulative "
            "probability exceeds top_p. A value of 1.0 considers all tokens. Lower values (e.g., 0.9) "
            "restrict sampling to more likely tokens. [Default: 1.0]"
        )
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=None,
        help=(
            "Penalty applied to repeated tokens. Values > 1.0 discourage repetition. "
            "A value of 1.0 means no penalty. [Default: None (no penalty)]"
        )
    )
    parser.add_argument(
        "--repetition-context-size",
        type=int,
        default=20,
        help=(
            "The number of previous tokens to consider for the repetition penalty. "
            "[Default: 20]"
        )
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=0, 
        help="Seed for the random number generator. [Default: 0]"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Stream the generated text token by token instead of printing the full output at the end."
    )

    args = parser.parse_args()
    main(args)

    # --- Example Test Cases ---
    # (Replace paths as needed)
    # 
    # 1. Basic generation (no adapter):
    # python src/inference/generate_qwen3.py \
    #   --model-path mlx_models/Qwen3-4B-mlx \
    #   --prompt "Tell me a short story about a brave knight."
    # 
    # 2. Generation with an adapter:
    # python src/inference/generate_qwen3.py \
    #   --model-path mlx_models/Qwen3-4B-mlx \
    #   --adapter-path ADAPTERS/qwen3_4b_lora_sacredhunger \
    #   --prompt "Write a paragraph in the style of the finetuning data."
    # 
    # 2b. Generation with adapter, reading long prompt from stdin:
    # cat prompt.txt | python src/inference/generate_qwen3.py \\
    #   --model-path mlx_models/Qwen3-4B-mlx \\
    #   --adapter-path ADAPTERS/qwen3_4b_lora_sacredhunger \\
    #   --prompt "-"
    # (Where prompt.txt contains your long prompt)
    #
    # 2c. Generation with adapter, using a 'here document' for prompt:
    # python src/inference/generate_qwen3.py \\
    #   --model-path mlx_models/Qwen3-4B-mlx \\
    #   --adapter-path ADAPTERS/qwen3_4b_lora_sacredhunger \\
    #   --prompt "-" <<EOF
    # This is a very long prompt
    # that spans multiple lines.
    # The shell will pass this text to the script's stdin.
    # EOF
    #
    # 3. More creative output (higher temperature):
    # python src/inference/generate_qwen3.py \
    #   --model-path mlx_models/Qwen3-4B-mlx \
    #   --prompt "What if the moon was made of cheese?" \
    #   --temp 1.0 \
    #   --max-tokens 150
    #
    # 4. More focused output (lower temperature, top-p):
    # python src/inference/generate_qwen3.py \
    #   --model-path mlx_models/Qwen3-4B-mlx \
    #   --prompt "Explain the concept of photosynthesis in simple terms." \
    #   --temp 0.3 \
    #   --top-p 0.9 \
    #   --max-tokens 200
    #
    # 5. Discourage repetition:
    # python src/inference/generate_qwen3.py \
    #   --model-path mlx_models/Qwen3-4B-mlx \
    #   --prompt "List the planets in our solar system, starting from the sun." \
    #   --repetition-penalty 1.2 \
    #   --max-tokens 50
    #
    # 6. Stream output token by token (with adapter):
    # python src/inference/generate_qwen3.py \
    #   --model-path mlx_models/Qwen3-4B-mlx \
    #   --adapter-path ADAPTERS/qwen3_4b_lora_sacredhunger \
    #   --prompt "Write a haiku about a cat." \
    #   --verbose