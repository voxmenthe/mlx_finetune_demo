## 1 · Baseline, purely programmatic splitter

**Core idea**

1. Read the text.
2. Isolate **paragraph units** (book already uses double line‑breaks as separators).
3. Greedily concatenate whole paragraphs until the running word‑count would exceed `target_size`; then start a new chunk.
4. Optionally create a *fixed paragraph overlap* for a little context bleed.

*Configurable knobs*

| parameter   | purpose                              | sensible range |
| ----------- | ------------------------------------ | -------------- |
| `--size`    | target words per chunk               | 300‑2 000      |
| `--overlap` | paragraphs repeated at each boundary | 0‑2            |

---

## 2 · Semantic‑aware splitter (embedding + BM25 refinement)

### Rationale

Even with paragraph‑respecting boundaries you can accidentally:

* cut **mid‑scene** (a named character vanishes between chunks),
* split **dialogue exchanges**, harming retrieval‑QA accuracy.

We therefore **post‑process** the baseline boundaries:

1. **Initial pass** – call the greedy algorithm above with `size ≈ target × 0.9` (gives headroom for later shuffling).

2. **Compute embeddings** for each paragraph (or sentence) using a suitable model (e.g., `lightonai/modernbert-embed-large`).
   *Note: This model requires specific prefixes. The script currently uses `search_document:` for the text segments being compared.*

3. For every tentative boundary `B` between chunk *i* and *i+1*:

   * Take the last `tail_len` sentences of chunk *i* (`S_tail`) and the first `head_len` sentences of chunk *i+1* (`S_head`).
   * `sim = cosine(get_emb(S_tail), get_emb(S_head))`.
   * If `sim < thresh_low`, **shift** `B` *forward* until similarity rises or the size budget is hit.
   * If `sim > thresh_high`, optionally create a *sentence‑level overlap* so the teaser sentence appears in both chunks.

4. **BM25 character check** – build a BM25 index over paragraphs. For each main character name (Erasmus, Paris, Thurso, etc.) ensure that it doesn't disappear for > `gap` paragraphs. If a gap occurs across a boundary, shift the boundary backward by one paragraph.


### Configurable aspects

* `--target` – desired words / chunk
* `tail_len`, `head_len` – size of *join windows*
* `thresh_low`, `thresh_high` – similarity action thresholds
* `char_names` + `max_bm25_gap` – continuity heuristics
* `max_size` – hard cap after refinement

---

**Measuring chunk boundaries**

Measure the distance between chunks by counting paragraphs from the last occurrence of an entity in one chunk to the first in the next. I'll look for the entity in both chunks, calculate the gap, and check if it exceeds a defined max number of paragraphs. If it does, I'll return "True." This will be accomplished using a case-insensitive regex for the entity's location. My approach seems clear, just ensuring I handle both chunks with careful indexing.

```python
# --- utilities --------------------------------------------------------------
def _tokenise(text: str) -> list[str]:
    """Very light tokeniser → lowercase words A‑Z."""
    return re.findall(r"[A-Za-z']+", text.lower())

try:
    from rank_bm25 import BM25Okapi
    _HAS_BM25 = True
except ImportError:
    _HAS_BM25 = False
# ---------------------------------------------------------------------------

## `bm25_gap_violation`
 
### How it works

1. **Tokenisation** – a tiny regex picks out alphabetic tokens and lower‑cases them, enough for BM25.
2. **Dual mode**

   * If `rank_bm25` is available, we calculate paragraph‑level BM25 scores for the entity; any paragraph scoring above `bm25_thresh` (0 → "contains at least one query term") counts as a hit.
   * Without the library, we revert to a fast exact word‑boundary regex.
3. **Gap detection** – walk backward from the end of the *previous* chunk and forward from the start of the *next* chunk to locate the two nearest mentions.  The sum of paragraphs between those two mentions (inclusive of the join) is the **gap**.  If it exceeds `max_gap`, the function flags a violation so your boundary‑refinement logic can shift or duplicate paragraphs.

You can now import the function directly in `semantic_chunker.py`, run the script, and the continuity‑checking step will operate deterministically—optionally strengthened by BM25 when the library is installed.

