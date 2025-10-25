"""Split a PDF into per-page JPEGs using selectable backends.

This script provides multiple conversion algorithms so the user can pick the
one that fits their environment. Each algorithm is declared in the CONFIG
section to keep defaults visible and easy to change.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Protocol


# ---------------------------------------------------------------------------
# Configuration: adjust these values to tune defaults without touching logic.
# ---------------------------------------------------------------------------
CONFIG = {
    "default_quality": 90,  # JPEG quality: 1 (worst) .. 100 (best)
    "default_algorithm": "pymupdf",
    "algorithm_settings": {
        # PyMuPDF offers fast native rendering. Requires "pymupdf" to be installed.
        "pymupdf": {
            "dpi": 200,
        },
        # pdf2image leverages Poppler. Requires "pdf2image" and Poppler utilities.
        "pdf2image": {
            "dpi": 200,
            "thread_count": 2,
        },
    },
}


class ConversionError(RuntimeError):
    """Base class for conversion-related failures."""


class MissingDependencyError(ConversionError):
    """Raised when an algorithm cannot run because a dependency is absent."""


class ConversionAlgorithm(Protocol):
    """Callable signature all conversion backends must satisfy."""

    def __call__(
        self,
        pdf_path: Path,
        output_dir: Path,
        *,
        jpeg_quality: int,
        settings: Dict[str, int | float | str | None],
    ) -> List[Path]:
        ...


@dataclass(frozen=True)
class AlgorithmDefinition:
    name: str
    executor: ConversionAlgorithm
    settings: Dict[str, int | float | str | None]


def _convert_with_pymupdf(
    pdf_path: Path,
    output_dir: Path,
    *,
    jpeg_quality: int,
    settings: Dict[str, int | float | str | None],
) -> List[Path]:
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise MissingDependencyError(
            "Algorithm 'pymupdf' requires the 'pymupdf' package."
        ) from exc

    dpi = int(settings.get("dpi", 200) or 200)
    scale = dpi / 72  # PDF points per inch

    saved_paths: List[Path] = []
    doc = fitz.open(pdf_path)
    try:
        for page_index in range(doc.page_count):
            page = doc.load_page(page_index)
            pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
            output_path = output_dir / f"{pdf_path.stem}_page_{page_index + 1:04d}.jpg"
            pix.save(output_path.as_posix(), jpg_quality=jpeg_quality)
            saved_paths.append(output_path)
            logging.debug(
                "Rendered page %s with PyMuPDF at %sdpi to %s",
                page_index + 1,
                dpi,
                output_path,
            )
    finally:
        doc.close()

    return saved_paths


def _convert_with_pdf2image(
    pdf_path: Path,
    output_dir: Path,
    *,
    jpeg_quality: int,
    settings: Dict[str, int | float | str | None],
) -> List[Path]:
    try:
        from pdf2image import convert_from_path
    except ImportError as exc:
        raise MissingDependencyError(
            "Algorithm 'pdf2image' requires the 'pdf2image' package and Poppler."
        ) from exc

    dpi = int(settings.get("dpi", 200) or 200)
    thread_count = settings.get("thread_count")

    images = convert_from_path(
        pdf_path.as_posix(),
        dpi=dpi,
        thread_count=int(thread_count) if thread_count else None,
    )

    saved_paths: List[Path] = []
    for index, image in enumerate(images, start=1):
        output_path = output_dir / f"{pdf_path.stem}_page_{index:04d}.jpg"
        image.save(
            output_path,
            format="JPEG",
            quality=jpeg_quality,
            optimize=True,
        )
        saved_paths.append(output_path)
        logging.debug(
            "Rendered page %s with pdf2image at %sdpi to %s",
            index,
            dpi,
            output_path,
        )

    return saved_paths


ALGORITHM_REGISTRY: Dict[str, ConversionAlgorithm] = {
    "pymupdf": _convert_with_pymupdf,
    "pdf2image": _convert_with_pdf2image,
}


def _build_algorithm_definitions() -> Dict[str, AlgorithmDefinition]:
    definitions: Dict[str, AlgorithmDefinition] = {}
    for name, executor in ALGORITHM_REGISTRY.items():
        settings = CONFIG["algorithm_settings"].get(name, {})
        definitions[name] = AlgorithmDefinition(name=name, executor=executor, settings=settings)
    return definitions


def _parse_arguments(available_algorithms: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split a PDF into JPEG pages using the configured conversion backend.",
    )
    parser.add_argument("pdf_path", type=Path, help="Path to the PDF file to split")
    parser.add_argument(
        "--algorithm",
        choices=sorted(available_algorithms),
        default=CONFIG["default_algorithm"],
        help="Conversion backend to use (default: %(default)s)",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=CONFIG["default_quality"],
        help="JPEG quality between 1-100 (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory (defaults to <pdf_stem>_pages alongside the PDF)",
    )

    args = parser.parse_args()

    if not 1 <= args.quality <= 100:
        parser.error("quality must be between 1 and 100")

    return args


def _prepare_output_dir(pdf_path: Path, explicit_output_dir: Path | None) -> Path:
    if explicit_output_dir is not None:
        output_dir = explicit_output_dir
    else:
        output_dir = pdf_path.with_name(f"{pdf_path.stem}_pages")

    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _ensure_pdf_exists(pdf_path: Path) -> None:
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if not pdf_path.is_file():
        raise ValueError(f"Expected a file path, got: {pdf_path}")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    algorithm_definitions = _build_algorithm_definitions()
    args = _parse_arguments(algorithm_definitions.keys())

    pdf_path: Path = args.pdf_path.resolve()
    _ensure_pdf_exists(pdf_path)

    output_dir = _prepare_output_dir(pdf_path, args.output_dir)
    logging.info("Saving pages to %s", output_dir)

    definition = algorithm_definitions[args.algorithm]

    try:
        saved_paths = definition.executor(
            pdf_path,
            output_dir,
            jpeg_quality=args.quality,
            settings=definition.settings,
        )
    except MissingDependencyError as exc:
        logging.error("%s", exc)
        return 2
    except ConversionError as exc:
        logging.error("Conversion failed: %s", exc)
        return 3
    except Exception as exc:  # pragma: no cover - unexpected failure surface
        logging.exception("Unexpected error during conversion")
        return 4

    logging.info("Wrote %s pages", len(saved_paths))
    for path in saved_paths:
        logging.debug("Created %s", path)

    return 0


if __name__ == "__main__":
    sys.exit(main())

