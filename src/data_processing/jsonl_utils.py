import json
import random
from pathlib import Path
from typing import List, Any, Dict

def shuffle_jsonl(input_path: str | Path, output_path: str | Path, seed: int) -> None:
    """
    Reads a JSONL file, shuffles its lines randomly using a seed,
    and writes the shuffled lines to a new JSONL file.

    Args:
        input_path: Path to the input JSONL file.
        output_path: Path to save the shuffled output JSONL file.
        seed: Random seed for shuffling.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines: List[Dict[str, Any]] = []
    with open(input_path, "r", encoding="utf-8") as infile:
        for line in infile:
            try:
                lines.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                print(f"Warning: Skipping invalid JSON line in {input_path}: {line.strip()}")

    print(f"Read {len(lines)} lines from {input_path}.")

    random.seed(seed)
    random.shuffle(lines)
    print(f"Shuffled {len(lines)} lines using seed {seed}.")

    with open(output_path, "w", encoding="utf-8") as outfile:
        for item in lines:
            outfile.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Saved shuffled data to {output_path}.")


def concatenate_and_shuffle_jsonl(
    input_paths: List[str | Path], output_path: str | Path, seed: int
) -> None:
    """
    Loads multiple JSONL files, concatenates their contents, shuffles the combined
    lines randomly using a seed, and saves them as a single JSONL file.

    Args:
        input_paths: A list of paths to the input JSONL files.
        output_path: Path to save the concatenated and shuffled output JSONL file.
        seed: Random seed for shuffling.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_lines: List[Dict[str, Any]] = []
    total_lines_read = 0
    for input_path_str in input_paths:
        input_path = Path(input_path_str)
        if not input_path.is_file():
            print(f"Warning: Input file not found, skipping: {input_path}")
            continue

        current_file_lines: List[Dict[str, Any]] = []
        with open(input_path, "r", encoding="utf-8") as infile:
            for line_num, line in enumerate(infile, 1):
                stripped_line = line.strip()
                if not stripped_line: # Skip empty lines
                    continue
                try:
                    obj = json.loads(stripped_line)
                    # Assuming primary use case is JSON objects per line
                    if isinstance(obj, dict):
                         current_file_lines.append(obj)
                    else:
                        # Handle cases like top-level arrays, strings, numbers if needed,
                        # but typically JSONL has objects. Warn if not an object.
                        print(f"Warning: Skipping non-object JSON on line {line_num} in {input_path}: Type={type(obj)}")
                        # Optionally append if other types are expected: current_file_lines.append(obj)

                except json.JSONDecodeError as e:
                    # Provide more context about the error
                    print(f"Warning: Skipping invalid JSON on line {line_num} in {input_path}.")
                    print(f"  Error: {e}")
                    # Optionally print the problematic line (or part of it)
                    max_len = 150 # Increased preview length
                    line_preview = stripped_line[:max_len] + ('...' if len(stripped_line) > max_len else '')
                    print(f"  Line content (preview): {line_preview}")
        print(f"Read {len(current_file_lines)} valid JSON objects from {input_path}.")
        all_lines.extend(current_file_lines)
        total_lines_read += len(current_file_lines) # Count only successfully read objects

    # Adjust total lines read message to reflect valid objects
    print(f"Read a total of {total_lines_read} valid JSON objects from {len(input_paths)} files.")

    if not all_lines:
        print("Warning: No lines were read from any input file. Output file will be empty.")
        # Create an empty file
        with open(output_path, "w", encoding="utf-8") as outfile:
            pass
        print(f"Created empty output file at {output_path}.")
        return

    random.seed(seed)
    random.shuffle(all_lines)
    print(f"Shuffled {len(all_lines)} combined lines using seed {seed}.")

    # Revert to default newline handling by removing newline=''
    with open(output_path, "w", encoding="utf-8") as outfile:
        for i, item in enumerate(all_lines):
             # Add a check just in case non-dict items slipped through (shouldn't happen with current read logic)
            if not isinstance(item, dict):
                print(f"Error: Item at index {i} after shuffle is not a dict: {type(item)}. Skipping.")
                continue
            try:
                # Write the JSON object followed by a newline
                outfile.write(json.dumps(item, ensure_ascii=False) + "\n")
            except Exception as e:
                # Catch potential errors during dumping (e.g., complex objects not serializable)
                print(f"Error dumping item at index {i} to JSON. Skipping.")
                print(f"  Item Preview: {str(item)[:100]}...") # Avoid printing huge items
                print(f"  Error: {e}")

    print(f"Saved concatenated and shuffled data to {output_path}.")

if __name__ == '__main__':
    # Example Usage (replace with your actual file paths and desired seed)

    # --- Example 1: Shuffle a single file ---
    # input_single = Path("path/to/your/input.jsonl")
    # output_single = Path("path/to/your/shuffled_output.jsonl")
    # random_seed = 42
    # if input_single.exists():
    #     shuffle_jsonl(input_single, output_single, random_seed)
    # else:
    #     print(f"Example 1 input file not found: {input_single}")

    # --- Example 2: Concatenate and shuffle multiple files ---

    input_multiple = [
        Path("../../DATA/ALLTHEKINGSMEN/train.jsonl"),
        Path("../../DATA/SACREDHUNGER/train.jsonl"),
        # Path("../../DATA/ALLTHEKINGSMEN/valid.jsonl"),
        # Path("../../DATA/SACREDHUNGER/valid.jsonl"),
    ]
    output_multiple = Path("../../DATA/NOVELS/train.jsonl")
    random_seed_concat = 123

    existing_input_multiple = [p for p in input_multiple if p.exists()]
    if not existing_input_multiple:
         print("No input files found.")
    elif len(existing_input_multiple) < len(input_multiple):
        print(f"Found {len(existing_input_multiple)} out of {len(input_multiple)} input files.")


    if existing_input_multiple:
        concatenate_and_shuffle_jsonl(existing_input_multiple, output_multiple, random_seed_concat)

    print("\nScript finished.")
