# semantic_chunker.py
import json, re
import numpy as np
from pathlib import Path
from typing import List
from tqdm import tqdm

import numpy as np
from baseline_chunker import load_paragraphs, chunk_paragraphs
from src.data_processing.bm25_func import bm25_gap_violation

from sentence_transformers import SentenceTransformer

# ==== plug‑in hooks =========================================================
# Removed embed and similarities helper functions as logic is now inline
# ============================================================================

def refine_boundaries(chunks: list[str],
                      tail_len: int = 2,
                      head_len: int = 2,
                      thresh_low: float = 0.15,
                      thresh_high: float = 0.65,
                      char_names: list[str] = [
                          "Erasmus", "Paris", "Thurso", "William Kemp",
                          "Sarah", "Blair", "Calley", "Kireku", "Delblanc", "Daniel",
                          "Liverpool Merchant", "Loango", "Bonny", "Whydah",
                          "Bight of Benin", "Barbados", "Liverpool", "Mersey"
                      ],
                      max_bm25_gap: int = 4,
                      max_size: int = 1200,
                      model: SentenceTransformer = None):
    new_chunks = []
    modification_count = 0  # Initialize modification counter
    modified_current_boundary = False  # Flag to track if current boundary was modified

    # Wrap chunks with tqdm for progress bar
    for i, chunk in enumerate(tqdm(chunks, desc="Refining chunk boundaries")):
        modified_current_boundary = False # Reset flag for each boundary check
        if i == 0:
            new_chunks.append(chunk)
            continue

        prev = new_chunks[-1]
        # candidate sentences near the join
        prev_sents = re.split(r'(?<=[.!?])\\s+', prev)
        next_sents = re.split(r'(?<=[.!?])\\s+', chunk)
        tail = " ".join(prev_sents[-tail_len:])
        head = " ".join(next_sents[:head_len])

        # Encode tail as document, head as query
        tail_embedding = model.encode(f"search_document: {tail}")
        head_embedding = model.encode(f"search_query: {head}")
        # Calculate similarity
        sim = model.similarity(tail_embedding, head_embedding)[0, 0] # Access the single similarity score

        if sim < thresh_low:
            # move first paragraph of current chunk back to previous
            para_split = re.split(r'\\n{2,}', chunk, maxsplit=1)
            if len(para_split) == 2:
                new_chunks[-1] += "\\n\\n" + para_split[0]
                chunk = para_split[1]
                modification_count += 1
                modified_current_boundary = True
        elif sim > thresh_high:
            # duplicate a connecting sentence for continuity
            new_chunks[-1] += " " + head
            modification_count += 1
            modified_current_boundary = True

        # BM25 continuity check (pseudo) - only check if not already modified by similarity
        if not modified_current_boundary:
            for character in char_names:
                if bm25_gap_violation((new_chunks[-1], chunk), character, max_bm25_gap):
                    # pull one paragraph back if gap too wide
                    para_split = re.split(r'\\n{2,}', chunk, maxsplit=1)
                    if len(para_split) == 2:
                        new_chunks[-1] += "\\n\\n" + para_split[0]
                        chunk = para_split[1]
                        modification_count += 1
                        modified_current_boundary = True
                        break # Stop checking characters for this boundary once modified

        # enforce hard upper size
        if len(chunk.split()) > max_size:
            # optional second pass of baseline chunking just on *this* oversize chunk
            mini_chunks = chunk_paragraphs(
                re.split(r'\\n{2,}', chunk), target_words=max_size)
            new_chunks.extend(mini_chunks)
            modification_count += 1 # Count the split as one modification event
            # Don't append the original oversized chunk
        else:
            # Only append if the chunk wasn't replaced by mini_chunks
            new_chunks.append(chunk)

    return new_chunks, modification_count # Return modification count

if __name__ == "__main__":
    import argparse

    DEFAULT_TARGET_WORDS = 350 # 350 480 520 570 680 730 790
    DEFAULT_MODEL_NAME = "lightonai/modernbert-embed-large"
    DEFAULT_BOOK_PATH = "sacredhunger.txt"
    DEFAULT_OUTPUT_PATH = f"sacredhunger_{DEFAULT_TARGET_WORDS}.json"

    ap = argparse.ArgumentParser(
        description="Chunk a book into semantic segments, refining initial paragraph-based chunks."
    )
    ap.add_argument(
        "--book_path",
        type=str,
        default=DEFAULT_BOOK_PATH,
        help=f"Path to the input text file (book). Defaults to '{DEFAULT_BOOK_PATH}'.",
    )
    ap.add_argument(
        "--target",
        type=int,
        default=DEFAULT_TARGET_WORDS,
        help=f"Target number of words per chunk (default: {DEFAULT_TARGET_WORDS}). "
             f"Note: Baseline chunking aims for 90% of this target.",
    )
    ap.add_argument(
        "--model_name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help=f"Name of the SentenceTransformer model to use for embeddings (default: {DEFAULT_MODEL_NAME})."
    )
    ap.add_argument(
        "--output_path",
        type=str,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Path to the output file (default: {DEFAULT_OUTPUT_PATH})."
    )
    args = ap.parse_args()

    if args.target != DEFAULT_TARGET_WORDS:
        args.output_path = f"allthekingsmen_{args.target}.json"
        print(f"Using output path: {args.output_path}")

    print(f"Loading sentence transformer model: {args.model_name}")
    model = SentenceTransformer(args.model_name, trust_remote_code=True)

    print(f"Loading paragraphs from: {args.book_path}")
    paras = load_paragraphs(args.book_path)

    print(f"Creating baseline chunks (target ~{int(args.target * 0.9)} words)...")
    base = chunk_paragraphs(paras, int(args.target * 0.9))

    print(f"Refining {len(base)} baseline chunks...")
    refined, mod_count = refine_boundaries(base, model=model) # Capture modification count
    print(f"Refinement process modified {mod_count} chunk boundaries.") # Print count

    Path(args.output_path).write_text(
        json.dumps({"chunks": refined, "count": len(refined)}, indent=2),
        encoding="utf‑8")
    print(f"Wrote {len(refined)} refined chunks to {args.output_path}")
