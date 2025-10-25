"""Convert a directory of page images into a single markdown document using Qwen VLM."""

from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config


# ---------------------------------------------------------------------------
# Configuration defaults live here to keep tuning data-driven and low blast.
# ---------------------------------------------------------------------------
CONFIG = {
    "model_path": "mlx-community/Qwen3-VL-32B-Instruct-8bit",
    "page_prompt_template": (
        "Put all the text from this page into markdown format."
    ),
    "heading_template": "## Page {page_number}",
    "max_tokens": 3_200,
    "output_suffix": "_combined.md",
    "default_output_filename": "output.md",
    "supported_image_suffixes": {".jpg", ".jpeg", ".png", ".webp"},
}


PAGE_NUMBER_PATTERN = re.compile(r"(\d+)$")


@dataclass(frozen=True)
class PageInferenceResult:
    page_number: int
    image_path: Path
    markdown: str
    elapsed_seconds: float


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a markdown document by processing page images with Qwen VLM.",
    )
    parser.add_argument(
        "image_dir",
        type=Path,
        help="Directory containing per-page images (e.g. *_page_0001.jpg).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to write the markdown output (defaults next to the image directory).",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=CONFIG["page_prompt_template"],
        help="Prompt template applied to each page (supports {page_number} and {image_name}).",
    )
    parser.add_argument(
        "--heading-template",
        type=str,
        default=CONFIG["heading_template"],
        help="Markdown heading template per page (supports {page_number} and {image_name}).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=CONFIG["max_tokens"],
        help="Maximum tokens to generate per page.",
    )

    args = parser.parse_args()

    if args.max_tokens <= 0:
        parser.error("--max-tokens must be positive")

    return args


def _resolve_output_path(image_dir: Path, explicit_output: Path | None) -> Path:
    if explicit_output is not None:
        explicit_output = explicit_output.resolve(strict=False)
        if explicit_output.exists() and explicit_output.is_dir():
            return explicit_output / CONFIG["default_output_filename"]
        if explicit_output.suffix == "":
            return explicit_output / CONFIG["default_output_filename"]
        return explicit_output

    default_name = f"{image_dir.name}{CONFIG['output_suffix']}"
    return image_dir.parent / default_name


def _collect_image_paths(image_dir: Path, suffixes: Sequence[str]) -> List[Path]:
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not image_dir.is_dir():
        raise NotADirectoryError(f"Expected a directory path: {image_dir}")

    suffix_set = {s.lower() for s in suffixes}

    candidates = [
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in suffix_set
    ]

    if not candidates:
        raise FileNotFoundError(
            f"No supported image files found in {image_dir}. Expected suffixes: {sorted(suffix_set)}"
        )

    sortable = []
    unsortable: List[Path] = []
    for path in candidates:
        match = PAGE_NUMBER_PATTERN.search(path.stem)
        if match:
            sortable.append((int(match.group(1)), path))
        else:
            unsortable.append(path)

    if unsortable:
        logging.warning(
            "The following files are missing a trailing page number and will appear last in lexicographic order: %s",
            ", ".join(sorted(p.name for p in unsortable)),
        )

    sorted_paths = [path for _, path in sorted(sortable, key=lambda item: item[0])]
    sorted_paths.extend(sorted(unsortable))
    return sorted_paths


def _load_model(model_path: str):
    logging.info("Loading model %s", model_path)
    model, processor = load(model_path)
    config = load_config(model_path)
    return model, processor, config


def _run_inference(
    model,
    processor,
    config,
    *,
    image_path: Path,
    prompt_template: str,
    heading_template: str,
    page_number: int,
    max_tokens: int,
) -> PageInferenceResult:
    prompt = prompt_template.format(page_number=page_number, image_name=image_path.name)
    formatted_prompt = apply_chat_template(
        processor,
        config,
        prompt,
        num_images=1,
    )

    image_inputs = [image_path.as_posix()]
    start_time = time.perf_counter()
    output = generate(
        model,
        processor,
        formatted_prompt,
        image_inputs,
        verbose=False,
        max_tokens=max_tokens,
    )
    elapsed = time.perf_counter() - start_time

    heading = heading_template.format(page_number=page_number, image_name=image_path.name)
    markdown_body = output.text.strip()
    markdown_block = f"{heading}\n\n{markdown_body}\n"

    logging.info(
        "Page %s processed in %.2f seconds (%s)",
        page_number,
        elapsed,
        image_path.name,
    )

    return PageInferenceResult(
        page_number=page_number,
        image_path=image_path,
        markdown=markdown_block,
        elapsed_seconds=elapsed,
    )


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    args = _parse_arguments()
    image_dir = args.image_dir.resolve()
    output_path = _resolve_output_path(image_dir, args.output)

    image_paths = _collect_image_paths(image_dir, CONFIG["supported_image_suffixes"])
    logging.info("Found %s page images in %s", len(image_paths), image_dir)

    model, processor, config = _load_model(CONFIG["model_path"])

    output_path.parent.mkdir(parents=True, exist_ok=True)

    page_count = 0
    total_time = 0.0
    with output_path.open("w", encoding="utf-8") as output_file:
        for index, image_path in enumerate(image_paths, start=1):
            result = _run_inference(
                model,
                processor,
                config,
                image_path=image_path,
                prompt_template=args.prompt,
                heading_template=args.heading_template,
                page_number=index,
                max_tokens=args.max_tokens,
            )
            output_file.write(result.markdown)
            output_file.flush()
            page_count += 1
            total_time += result.elapsed_seconds

    logging.info(
        "Wrote %s pages to %s in %.2f seconds total - avg %.2f seconds/page",
        page_count,
        output_path,
        total_time,
        total_time / page_count,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
