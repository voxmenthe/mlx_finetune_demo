#!/usr/bin/env python3
"""
convert_to_markdown.py

Usage
-----
python convert_to_markdown_jsonl.py \
       --input  texts.txt            # one snippet per line
       --output converted.jsonl
"""

import argparse
import json
import re
import tempfile
from pathlib import Path
from typing import List

from markitdown import MarkItDown          # official API :contentReference[oaicite:1]{index=1}


def guess_suffix(text: str) -> str:
    """Very small heuristic to give MarkItDown the right file extension."""
    if re.search(r"<\s*html[^>]*>", text, re.I):
        return ".html"
    if text.lstrip().startswith("#"):
        return ".md"
    return ".txt"


def convert_snippets(snippets: List[str]) -> List[dict]:
    """Return a list of {'raw', 'markdown'} dictionaries."""
    md = MarkItDown(enable_plugins=False)   # one converter for all calls
    records = []

    for snippet in snippets:
        # Write the snippet to a NamedTemporaryFile so MarkItDown
        # can treat it like a real file
        suffix = guess_suffix(snippet)
        with tempfile.NamedTemporaryFile("w+b", suffix=suffix, delete=True) as tf:
            tf.write(snippet.encode("utf-8"))
            tf.flush()                      # ensure bytes are written
            result = md.convert(tf.name)    # convert returns a DocumentResult
            records.append({"raw": snippet, "markdown": result.text_content})
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Bulk convert text → Markdown (JSONL)")
    parser.add_argument("--input", required=True,
                        help="Text file with one snippet per line")
    parser.add_argument("--output", required=True,
                        help="Destination .jsonl file")
    args = parser.parse_args()

    # Load snippets (blank lines are ignored)
    with Path(args.input).expanduser().open(encoding="utf-8") as f:
        snippets = [line.rstrip("\n") for line in f if line.strip()]

    conversions = convert_snippets(snippets)

    # Write JSONL
    with Path(args.output).expanduser().open("w", encoding="utf-8") as out:
        for rec in conversions:
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"✅  Wrote {len(conversions):,} records to {args.output}")


if __name__ == "__main__":
    main()
