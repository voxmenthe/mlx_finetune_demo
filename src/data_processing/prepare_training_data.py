import json
import argparse
from pathlib import Path
import random

# Define the prompt template directly
PROMPT_TEMPLATE = '''This is an excerpt from a novel. Write the next excerpt of similar length. Use the same style as the excerpt. Make sure that while stylistically similar, the new section moves the story forward and/or develops the characters and/or adds new information or in some way continues on meaningfully from the previous section.

EXCERPT:
{}'''

def save_to_jsonl(data: list, file_path: Path):
    """Saves a list of strings to a JSONL file, with each string under the 'text' key."""
    with file_path.open("w", encoding="utf-8") as f:
        for item in data:
            # Ensure the output is a valid JSON line
            f.write(json.dumps({"text": item}) + '\n')

def load_chunks_from_file(file_path: Path) -> list:
    """Loads chunks from a single JSON file."""
    if not file_path.is_file():
        print(f"Warning: Input file not found at {file_path}, skipping.")
        return []
    try:
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if "chunks" not in data or not isinstance(data["chunks"], list):
            print(f"Warning: Input file {file_path} does not contain a list under the 'chunks' key, skipping.")
            return []
        return data["chunks"]
    except json.JSONDecodeError:
        print(f"Warning: Could not decode JSON from {file_path}, skipping.")
        return []
    except Exception as e:
        print(f"Warning: An error occurred while reading {file_path}: {e}, skipping.")
        return []


def create_pairs_from_chunks(chunks: list) -> list:
    """Creates prompt/completion pairs from a list of chunks."""
    paired_texts = []
    if len(chunks) < 2:
        return [] # Not enough chunks to form pairs
    # Iterate up to the second-to-last chunk to form pairs
    for i in range(len(chunks) - 1):
        prompt_part = PROMPT_TEMPLATE.format(chunks[i])
        completion_part = chunks[i+1]
        combined_text = f"{prompt_part}\nNEXT EXCERPT:\n{completion_part}"
        paired_texts.append(combined_text)
    return paired_texts

if __name__ == "__main__":
    DEFAULT_INPUT_FILES = ["semantic_chunks.json"] # Default is now a list
    DEFAULT_OUTPUT_DIR = "."
    DEFAULT_TRAIN_RATIO = 0.85
    DEFAULT_SEED = 42

    parser = argparse.ArgumentParser(
        description="Prepare training and validation data from semantic chunks in one or more input files."
    )
    parser.add_argument(
        "--input_files",
        type=str,
        nargs='+', # Accept one or more arguments
        default=DEFAULT_INPUT_FILES,
        help=f"Path(s) to the input JSON file(s) containing semantic chunks (default: {DEFAULT_INPUT_FILES[0]})."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to save the train.jsonl and valid.jsonl files (default: {DEFAULT_OUTPUT_DIR})."
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=DEFAULT_TRAIN_RATIO,
        help=f"Proportion of data to use for the training set (default: {DEFAULT_TRAIN_RATIO})."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for shuffling data (default: {DEFAULT_SEED})."
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True) # Ensure output directory exists

    all_chunks_data = []
    max_chunks_len = 0
    file_with_max_chunks = None

    print("Loading chunks from input files...")
    for file_path_str in args.input_files:
        file_path = Path(file_path_str)
        print(f"  Processing: {file_path}")
        chunks = load_chunks_from_file(file_path)
        if chunks:
            all_chunks_data.append({"path": file_path, "chunks": chunks})
            if len(chunks) > max_chunks_len:
                max_chunks_len = len(chunks)
                file_with_max_chunks = file_path
        else:
            print(f"  No valid chunks loaded from {file_path}.")


    if not all_chunks_data:
        print("Error: No valid chunks loaded from any input file.")
        exit(1)

    if max_chunks_len < 2:
        print("Error: The file with the most chunks has less than 2 chunks. Cannot create pairs.")
        exit(1)

    print(f"\nFile determining shuffle/split order (max {max_chunks_len} chunks): {file_with_max_chunks}")
    max_pairs_len = max_chunks_len - 1 # Number of pairs is one less than chunks

    # Create shuffled indices based on the file with the maximum number of pairs
    print(f"Creating shuffle order based on {max_pairs_len} potential pairs...")
    indices = list(range(max_pairs_len))
    random.seed(args.seed)
    random.shuffle(indices)

    # Determine split point based on the max number of pairs
    split_index = int(max_pairs_len * args.train_ratio)
    train_indices_full = set(indices[:split_index])
    valid_indices_full = set(indices[split_index:])

    print(f"Determined split: {len(train_indices_full)} train indices, {len(valid_indices_full)} valid indices (based on max pairs).")

    combined_train_data = []
    combined_valid_data = []

    print("\nProcessing pairs and splitting for each file...")
    for file_data in all_chunks_data:
        file_path = file_data["path"]
        chunks = file_data["chunks"]
        print(f"  Processing pairs from: {file_path} ({len(chunks)} chunks)")

        paired_texts = create_pairs_from_chunks(chunks)
        num_pairs_in_file = len(paired_texts)

        if num_pairs_in_file == 0:
            print(f"    No pairs created for {file_path}, skipping.")
            continue

        print(f"    Created {num_pairs_in_file} pairs.")

        file_train_data = []
        file_valid_data = []

        # Use the pre-calculated shuffled indices, but only up to the number of pairs available in *this* file
        for i in range(num_pairs_in_file):
            original_index = i # This corresponds to the index in paired_texts for this file
            # Check if this index falls into the train or valid set based on the *overall* shuffled indices
            # We map the position `i` in the *current* file's pairs back to the global shuffled index list `indices`.
            # Find where `i` appears in the shuffled list `indices`.
            # However, a simpler approach is to iterate through the global train/valid indices and check if they are valid for the current file
            pass # Refactoring logic below

        current_train_count = 0
        current_valid_count = 0
        # Iterate through the globally determined shuffled indices
        for idx_in_shuffled_list, original_pair_index in enumerate(indices):
             # Check if this original_pair_index is valid for the current file's pair list
             if original_pair_index < num_pairs_in_file:
                 pair = paired_texts[original_pair_index]
                 # Determine if this index belongs to the train or validation set based on the split point of the *shuffled* list
                 if idx_in_shuffled_list < split_index: # Check position in the shuffled list
                     file_train_data.append(pair)
                     current_train_count += 1
                 else:
                     file_valid_data.append(pair)
                     current_valid_count += 1


        print(f"    Added {current_train_count} pairs to train set, {current_valid_count} pairs to valid set.")
        combined_train_data.extend(file_train_data)
        combined_valid_data.extend(file_valid_data)


    print(f"\nTotal training samples: {len(combined_train_data)}")
    print(f"Total validation samples: {len(combined_valid_data)}")

    # Shuffle the combined sets again for good measure? Optional, but can help ensure randomness if order matters downstream.
    # random.shuffle(combined_train_data)
    # random.shuffle(combined_valid_data)

    train_output_path = output_dir / "train.jsonl"
    valid_output_path = output_dir / "valid.jsonl"

    print(f"\nSaving combined training data to: {train_output_path}")
    save_to_jsonl(combined_train_data, train_output_path)

    print(f"Saving combined validation data to: {valid_output_path}")
    save_to_jsonl(combined_valid_data, valid_output_path)

    print("\nDone.")

"""
Example Usage (Multiple Files):
python src/data_processing/prepare_training_data.py \
    --input_files semantic_chunks_part1.json semantic_chunks_part2.json \
    --output_dir DATA/processed_novel_combined \
    --train_ratio 0.9 \
    --seed 123
""" 