"""
convert_hf_dataset_format.py

Converts Hugging Face datasets from the "conversations" format to the standard "messages" format.
This script handles the conversion between different chat dataset formats, following Hugging Face
best practices for dataset transformation.

Source format: 
- "conversations" field with "from" and "value" keys
- Roles: "human", "gpt", "system"

Target format:
- "messages" field with "role" and "content" keys  
- Roles: "user", "assistant", "system"
- Additional fields: "id", "source"
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datasets import Dataset, load_dataset, DatasetDict, IterableDataset, IterableDatasetDict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Role mapping from source to target format
ROLE_MAPPING = {
    "human": "user",
    "gpt": "assistant", 
    "system": "system"
}

def convert_conversation_format(example: Dict[str, Any], sample_number: int, dataset_name: str, source_name: str) -> Dict[str, Any]:
    """
    Converts a single example from conversations format to messages format.
    
    Args:
        example: Dictionary containing the conversation data
        sample_number: The row number for generating the ID
        dataset_name: Name to use for ID generation
        source_name: Source name for the dataset
        
    Returns:
        Dictionary in the target format with messages, id, and source fields
    """
    if "conversations" not in example:
        logger.warning(f"Sample {sample_number}: No 'conversations' field found")
        return None
    
    conversations = example["conversations"]
    if not isinstance(conversations, list):
        logger.warning(f"Sample {sample_number}: 'conversations' is not a list")
        return None
    
    messages = []
    for turn in conversations:
        if not isinstance(turn, dict) or "from" not in turn or "value" not in turn:
            logger.warning(f"Sample {sample_number}: Invalid conversation turn format")
            continue
            
        role = turn["from"]
        content = turn["value"]
        
        # Skip system messages as specified in requirements
        if role == "system":
            continue
            
        # Map roles according to the specification
        if role in ROLE_MAPPING:
            mapped_role = ROLE_MAPPING[role]
            messages.append({
                "role": mapped_role,
                "content": content
            })
        else:
            logger.warning(f"Sample {sample_number}: Unknown role '{role}', skipping turn")
    
    # Only return valid examples with at least one message
    if not messages:
        logger.warning(f"Sample {sample_number}: No valid messages after conversion")
        return None
    
    return {
        "messages": messages,
        "id": f"{dataset_name}_{sample_number}",
        "source": source_name
    }

def convert_dataset_batch(batch: Dict[str, List[Any]], start_idx: int, dataset_name: str, source_name: str) -> Dict[str, List[Any]]:
    """
    Converts a batch of examples for efficient processing.
    
    Args:
        batch: Dictionary containing batched data
        start_idx: Starting index for ID generation
        dataset_name: Name to use for ID generation
        source_name: Source name for the dataset
        
    Returns:
        Dictionary containing converted batch data
    """
    converted_messages = []
    converted_ids = []
    converted_sources = []
    
    batch_size = len(batch["conversations"])
    
    for i in range(batch_size):
        sample_number = start_idx + i
        example = {key: batch[key][i] for key in batch.keys()}
        
        converted = convert_conversation_format(example, sample_number, dataset_name, source_name)
        
        if converted is not None:
            converted_messages.append(converted["messages"])
            converted_ids.append(converted["id"])
            converted_sources.append(converted["source"])
    
    return {
        "messages": converted_messages,
        "id": converted_ids,
        "source": converted_sources
    }

def convert_single_example(example: Dict[str, Any], idx: int, dataset_name: str, source_name: str) -> Dict[str, Any]:
    """
    Converts a single example (for streaming datasets).
    
    Args:
        example: Dictionary containing the conversation data
        idx: The sample index for generating the ID
        dataset_name: Name to use for ID generation
        source_name: Source name for the dataset
        
    Returns:
        Dictionary in the target format with messages, id, and source fields
    """
    return convert_conversation_format(example, idx, dataset_name, source_name)

def extract_dataset_names(dataset_path: str) -> tuple[str, str]:
    """
    Extracts dataset name and source name from the dataset path.
    
    Args:
        dataset_path: Path to dataset (local path or HF Hub identifier)
        
    Returns:
        Tuple of (dataset_name, source_name)
    """
    # Convert to Path object to handle both local and remote paths
    path = Path(dataset_path)
    
    # Check if it looks like a HuggingFace Hub identifier (contains '/')
    if '/' in dataset_path and not dataset_path.startswith('./') and not dataset_path.startswith('/'):
        # HuggingFace Hub format: "org/repo" or "org/repo/subfolder"
        parts = dataset_path.split('/')
        if len(parts) >= 2:
            # Use the full path as source, just the repo name for ID
            source_name = dataset_path
            dataset_name = parts[1]  # Use just the repo name part
        else:
            # Fallback
            source_name = dataset_path
            dataset_name = dataset_path
    else:
        # Local path - use the filename without extension as both
        stem = path.stem if path.suffix else path.name
        source_name = stem
        dataset_name = stem
    
    return dataset_name, source_name

def load_and_convert_dataset(
    dataset_path: str,
    split: Optional[str] = None,
    streaming: bool = False,
    max_samples: Optional[int] = None
) -> Union[Dataset, DatasetDict, IterableDataset, IterableDatasetDict]:
    """
    Loads and converts a dataset from conversations format to messages format.
    
    Args:
        dataset_path: Path to the dataset (local or HF Hub)
        split: Which split to load (if None, loads all splits)
        streaming: Whether to use streaming mode
        max_samples: Maximum number of samples to process (for testing)
        
    Returns:
        Converted Dataset object
    """
    logger.info(f"Loading dataset from: {dataset_path}")
    
    # Extract dataset names for dynamic ID and source generation
    dataset_name, source_name = extract_dataset_names(dataset_path)
    logger.info(f"Using dataset_name='{dataset_name}', source_name='{source_name}'")
    
    try:
        # Load the dataset
        if split:
            dataset = load_dataset(dataset_path, split=split, streaming=streaming)
        else:
            dataset = load_dataset(dataset_path, streaming=streaming)
        
        if streaming:
            # Handle streaming datasets
            if isinstance(dataset, IterableDatasetDict):
                logger.info(f"Found streaming splits: {list(dataset.keys())}")
                converted_datasets = {}
                
                for split_name, split_dataset in dataset.items():
                    logger.info(f"Converting streaming split: {split_name}")
                    converted_datasets[split_name] = _convert_streaming_dataset(
                        split_dataset, max_samples, dataset_name, source_name
                    )
                
                return IterableDatasetDict(converted_datasets)
            else:
                # Single streaming dataset
                return _convert_streaming_dataset(dataset, max_samples, dataset_name, source_name)
        else:
            # Handle regular datasets
            if isinstance(dataset, DatasetDict):
                logger.info(f"Found splits: {list(dataset.keys())}")
                converted_datasets = {}
                
                for split_name, split_dataset in dataset.items():
                    logger.info(f"Converting split: {split_name}")
                    converted_datasets[split_name] = _convert_regular_dataset(
                        split_dataset, max_samples, dataset_name, source_name
                    )
                
                return DatasetDict(converted_datasets)
            else:
                # Single dataset
                return _convert_regular_dataset(dataset, max_samples, dataset_name, source_name)
            
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise

def _convert_regular_dataset(dataset: Dataset, max_samples: Optional[int] = None, dataset_name: str = "", source_name: str = "") -> Dataset:
    """Helper function to convert a regular (non-streaming) dataset."""
    
    # Limit samples if specified (useful for testing)
    if max_samples:
        logger.info(f"Limiting to {max_samples} samples for testing")
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    logger.info(f"Converting {len(dataset)} samples...")
    
    # Convert using map with batching for efficiency
    converted_dataset = dataset.map(
        lambda batch, idx: convert_dataset_batch(batch, idx, dataset_name, source_name),
        batched=True,
        batch_size=1000,  # Process in batches of 1000
        with_indices=True,
        remove_columns=dataset.column_names,  # Remove original columns
        desc="Converting format"
    )
    
    # Filter out None values (invalid conversions)
    original_size = len(converted_dataset)
    converted_dataset = converted_dataset.filter(
        lambda x: x["messages"] is not None and len(x["messages"]) > 0
    )
    final_size = len(converted_dataset)
    
    logger.info(f"Conversion complete: {final_size}/{original_size} samples retained")
    
    return converted_dataset

def _convert_streaming_dataset(dataset: IterableDataset, max_samples: Optional[int] = None, dataset_name: str = "", source_name: str = "") -> IterableDataset:
    """Helper function to convert a streaming dataset."""
    
    logger.info("Converting streaming dataset...")
    
    # For streaming datasets, we need to use a different approach
    def convert_example_with_index(example, idx):
        converted = convert_single_example(example, idx, dataset_name, source_name)
        return converted
    
    # Convert using map (no batching for streaming datasets)
    converted_dataset = dataset.map(
        convert_example_with_index,
        with_indices=True,
        remove_columns=dataset.column_names if hasattr(dataset, 'column_names') else None
    )
    
    # Filter out None values
    converted_dataset = converted_dataset.filter(
        lambda x: x is not None and "messages" in x and x["messages"] is not None and len(x["messages"]) > 0
    )
    
    # Limit samples if specified
    if max_samples:
        logger.info(f"Limiting to {max_samples} samples for testing")
        converted_dataset = converted_dataset.take(max_samples)
    
    logger.info("Streaming dataset conversion setup complete")
    
    return converted_dataset

def save_dataset(
    dataset: Union[Dataset, DatasetDict, IterableDataset, IterableDatasetDict],
    output_path: str,
    save_format: str = "json"
) -> None:
    """
    Saves the converted dataset to the specified format.
    
    Args:
        dataset: The converted dataset
        output_path: Path where to save the dataset
        save_format: Format to save in ('json', 'jsonl', 'parquet', 'hf_hub')
    """
    output_path = Path(output_path)
    
    if save_format == "hf_hub":
        # Push to Hugging Face Hub
        logger.info(f"Pushing dataset to Hub: {output_path}")
        
        # For streaming datasets, we need to collect the data first
        if isinstance(dataset, (IterableDataset, IterableDatasetDict)):
            logger.info("Converting streaming dataset to regular dataset for upload...")
            if isinstance(dataset, IterableDatasetDict):
                regular_datasets = {}
                for split_name, streaming_split in dataset.items():
                    logger.info(f"Collecting data for split: {split_name}")
                    data = list(streaming_split)
                    logger.info(f"Collected {len(data)} samples for split: {split_name}")
                    regular_datasets[split_name] = Dataset.from_list(data)
                dataset = DatasetDict(regular_datasets)
            else:
                logger.info("Collecting streaming data...")
                data = list(dataset)
                logger.info(f"Collected {len(data)} samples")
                dataset = Dataset.from_list(data)
        
        dataset.push_to_hub(str(output_path))
    else:
        # Save locally
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # For streaming datasets, collect data first
        if isinstance(dataset, (IterableDataset, IterableDatasetDict)):
            logger.info("Converting streaming dataset for local save...")
            if isinstance(dataset, IterableDatasetDict):
                for split_name, streaming_split in dataset.items():
                    split_output = output_path.parent / f"{output_path.name}_{split_name}"
                    data = list(streaming_split)
                    split_dataset = Dataset.from_list(data)
                    _save_single_dataset(split_dataset, split_output, save_format)
                return
            else:
                data = list(dataset)
                dataset = Dataset.from_list(data)
        
        if isinstance(dataset, DatasetDict):
            for split_name, split_dataset in dataset.items():
                split_output = output_path.parent / f"{output_path.name}_{split_name}"
                _save_single_dataset(split_dataset, split_output, save_format)
        else:
            _save_single_dataset(dataset, output_path, save_format)

def _save_single_dataset(dataset: Dataset, output_path: Path, save_format: str):
    """Helper function to save a single dataset."""
    if save_format == "json":
        output_file = output_path.with_suffix('.json')
        logger.info(f"Saving dataset to: {output_file}")
        dataset.to_json(str(output_file), orient="records", force_ascii=False)
        
    elif save_format == "jsonl":
        output_file = output_path.with_suffix('.jsonl')
        logger.info(f"Saving dataset to: {output_file}")
        dataset.to_json(str(output_file), orient="records", lines=True, force_ascii=False)
        
    elif save_format == "parquet":
        output_file = output_path.with_suffix('.parquet')
        logger.info(f"Saving dataset to: {output_file}")
        dataset.to_parquet(str(output_file))
        
    else:
        raise ValueError(f"Unsupported save format: {save_format}")

def validate_converted_sample(sample: Dict[str, Any]) -> bool:
    """
    Validates that a converted sample has the correct format.
    
    Args:
        sample: The converted sample to validate
        
    Returns:
        True if valid, False otherwise
    """
    required_fields = ["messages", "id", "source"]
    
    # Check required fields
    for field in required_fields:
        if field not in sample:
            return False
    
    # Validate messages structure
    messages = sample["messages"]
    if not isinstance(messages, list) or len(messages) == 0:
        return False
    
    # Validate each message
    for message in messages:
        if not isinstance(message, dict):
            return False
        if "role" not in message or "content" not in message:
            return False
        if message["role"] not in ["user", "assistant", "system"]:
            return False
    
    return True

def main():
    parser = argparse.ArgumentParser(
        description="Convert Hugging Face datasets from conversations to messages format"
    )
    parser.add_argument(
        "dataset_path",
        help="Path to dataset (local path or HF Hub identifier)"
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Output path for converted dataset"
    )
    parser.add_argument(
        "--split",
        help="Specific split to convert (default: all splits)"
    )
    parser.add_argument(
        "--format",
        choices=["json", "jsonl", "parquet", "hf_hub"],
        default="jsonl",
        help="Output format (default: jsonl)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Maximum number of samples to process (for testing)"
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use streaming mode for large datasets"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate a sample of converted data"
    )
    
    args = parser.parse_args()
    
    try:
        # Load and convert the dataset
        converted_dataset = load_and_convert_dataset(
            args.dataset_path,
            split=args.split,
            streaming=args.streaming,
            max_samples=args.max_samples
        )
        
        # Validate if requested
        if args.validate:
            logger.info("Validating converted data...")
            
            # Get a sample for validation
            if isinstance(converted_dataset, (DatasetDict, IterableDatasetDict)):
                first_split = list(converted_dataset.keys())[0]
                sample_dataset = converted_dataset[first_split]
            else:
                sample_dataset = converted_dataset
            
            # Get first sample
            if isinstance(sample_dataset, IterableDataset):
                sample = next(iter(sample_dataset))
            else:
                sample = sample_dataset[0]
            
            if validate_converted_sample(sample):
                logger.info("Validation passed ✓")
                logger.info(f"Sample: {json.dumps(sample, indent=2, ensure_ascii=False)}")
            else:
                logger.error("Validation failed ✗")
                return 1
        
        # Save the dataset
        save_dataset(converted_dataset, args.output_path, args.format)
        
        logger.info("Conversion completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())