# baseline_chunker.py
import re
from pathlib import Path
from typing import List, Iterable

def load_paragraphs(path: str | Path) -> List[str]:
    raw = Path(path).read_text(encoding="utf‑8")
    # collapse Windows/Mac line endings, then split on 2+ newlines
    return [p.strip() for p in re.split(r"\n{2,}", raw) if p.strip()]

def chunk_paragraphs(paragraphs: Iterable[str],
                     target_words: int = 1000,
                     overlap_paragraphs: int = 0) -> List[str]:
    chunks, current, cur_count = [], [], 0
    for p in paragraphs:
        p_words = len(p.split())
        # if adding this paragraph would push us *over* the target,
        # flush what we’ve got (unless empty) and start anew
        if current and cur_count + p_words > target_words:
            chunks.append("\n\n".join(current))
            # start next chunk with optional overlap from the *end*
            current = current[-overlap_paragraphs:] if overlap_paragraphs else []
            cur_count = sum(len(x.split()) for x in current)
        current.append(p)
        cur_count += p_words
    if current:
        chunks.append("\n\n".join(current))
    return chunks

if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("book_path")
    ap.add_argument("--size", type=int, default=1000,
                    help="≈ words per chunk (default 1000)")
    ap.add_argument("--overlap", type=int, default=0,
                    help="paragraphs to repeat between chunks")
    args = ap.parse_args()

    paras = load_paragraphs(args.book_path)
    chunks = chunk_paragraphs(paras, args.size, args.overlap)
    print(json.dumps({"chunks": chunks, "count": len(chunks)}, indent=2))
