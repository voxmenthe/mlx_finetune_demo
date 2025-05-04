import re
try:
    from rank_bm25 import BM25Okapi
    _HAS_BM25 = True
except ImportError:
    _HAS_BM25 = False


# --- utilities --------------------------------------------------------------
def _tokenise(text: str) -> list[str]:
    """Very light tokeniser → lowercase words A‑Z."""
    return re.findall(r"[A-Za-z']+", text.lower())

# ---------------------------------------------------------------------------


def bm25_gap_violation(boundary_chunks: tuple[str, str],
                       entity: str,
                       max_gap: int = 4,
                       bm25_thresh: float = 0.0) -> bool:
    """
    Return ``True`` when *entity* (e.g. “Erasmus”) vanishes for more than
    ``max_gap`` paragraph units **across** the boundary formed by the two
    chunks supplied.

    Parameters
    ----------
    boundary_chunks : (prev_chunk, next_chunk)
        Tuple of the text immediately before and after the boundary.
    entity : str
        Name / term whose continuity we want to keep.
    max_gap : int, default 4
        Maximum allowed paragraph distance without seeing the entity.
    bm25_thresh : float, default 0.0
        Minimum BM25 score regarded as a “hit”.  Leave at 0 to treat mere
        lexical presence as sufficient.

    Notes
    -----
    * If the third‑party package ``rank_bm25`` is present we build a very
      small per‑boundary BM25 index so that inflected or approximate
      mentions (“Mr Kemp”, “Kemp’s”) still register continuity.
    * If the package is missing we fall back to a fast
      case‑insensitive regex exact match.
    """

    prev_chunk, next_chunk = boundary_chunks
    prev_paras = re.split(r"\n{2,}", prev_chunk)
    next_paras = re.split(r"\n{2,}", next_chunk)

    # --------------------- helper to detect "entity present" ---------------
    entity_tokens = _tokenise(entity)
    if not entity_tokens:
        return False  # nothing to look for

    if _HAS_BM25:
        # Build tiny BM25 index over paragraphs that straddle the boundary
        corpus_paras = prev_paras + next_paras
        corpus_tok   = [_tokenise(p) for p in corpus_paras]
        bm25         = BM25Okapi(corpus_tok)
        scores       = bm25.get_scores(entity_tokens)
        # Treat paragraph as containing the entity if BM25 > threshold
        contains     = [s > bm25_thresh for s in scores]
    else:
        # Cheap lexical fallback
        pat = re.compile(rf"\b{re.escape(entity)}\b", flags=re.I)
        contains = [bool(pat.search(p)) for p in prev_paras + next_paras]

    # --------------------- measure the paragraph gap -----------------------
    # Index of *last* hit in the previous chunk
    last_prev_idx = None
    for i in reversed(range(len(prev_paras))):
        if contains[i]:
            last_prev_idx = len(prev_paras) - 1 - i  # distance back from end
            break

    # Index of *first* hit in the next chunk
    offset = len(prev_paras)  # shift into global index
    first_next_idx = None
    for j in range(len(next_paras)):
        if contains[offset + j]:
            first_next_idx = j
            break

    # Compute paragraphs without the entity spanning the join
    if last_prev_idx is None:
        gap_left = len(prev_paras)  # no mention in prev ⇒ full length
    else:
        gap_left = last_prev_idx

    if first_next_idx is None:
        gap_right = len(next_paras)  # no mention in next ⇒ full length
    else:
        gap_right = first_next_idx

    total_gap = gap_left + gap_right + 1  # +1 for the boundary itself

    return total_gap > max_gap
