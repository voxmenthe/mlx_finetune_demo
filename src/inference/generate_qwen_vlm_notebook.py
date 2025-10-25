# %%
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config


# %%
CONFIG = {
    "model_path": "mlx-community/Qwen3-VL-32B-Instruct-3bit",
    #"prompt_template": "Put all the text from this image into markdown format.",
    "prompt_template": "Put the table from this image into markdown format.",
    "max_tokens": 3_200,
    "supported_image_suffixes": {".jpg", ".jpeg", ".png", ".webp"},
}


@dataclass(frozen=True)
class LoadedArtifacts:
    model: object
    processor: object
    config: dict


@dataclass(frozen=True)
class SingleImageResult:
    image_path: Path
    text: str
    elapsed_seconds: float


# %%
def load_vlm_artifacts(model_path: str) -> LoadedArtifacts:
    logging.info("Loading model %s", model_path)
    model, processor = load(model_path)
    config = load_config(model_path)
    return LoadedArtifacts(model=model, processor=processor, config=config)


def validate_image_path(image_path: Path, suffixes: Sequence[str]) -> Path:
    resolved = image_path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Image not found: {resolved}")
    suffix_set = {suffix.lower() for suffix in suffixes}
    if resolved.suffix.lower() not in suffix_set:
        raise ValueError(f"Unsupported image suffix: {resolved.suffix}. Expected one of {sorted(suffix_set)}")
    return resolved


def generate_text_for_image(
    artifacts: LoadedArtifacts,
    *,
    image_path: Path,
    prompt_template: str,
    max_tokens: int,
) -> SingleImageResult:
    validated_path = validate_image_path(image_path, CONFIG["supported_image_suffixes"])
    formatted_prompt = apply_chat_template(
        artifacts.processor,
        artifacts.config,
        prompt_template,
        num_images=1,
    )

    image_inputs = [validated_path.as_posix()]
    start_time = time.perf_counter()
    output = generate(
        artifacts.model,
        artifacts.processor,
        formatted_prompt,
        image_inputs,
        verbose=False,
        max_tokens=max_tokens,
    )
    elapsed = time.perf_counter() - start_time

    return SingleImageResult(image_path=validated_path, text=output.text.strip(), elapsed_seconds=elapsed)


# %%
def convert_pdf_to_images(
    pdf_path: Path,
    output_dir: Path | None = None,
    *,
    algorithm: str = "pymupdf",
    dpi: int = 200,
    jpeg_quality: int = 90,
) -> List[Path]:
    resolved_pdf = pdf_path.expanduser().resolve()
    if not resolved_pdf.exists():
        raise FileNotFoundError(f"PDF not found: {resolved_pdf}")
    if output_dir is None:
        output_dir = resolved_pdf.with_name(f"{resolved_pdf.stem}_pages")
    output_dir.mkdir(parents=True, exist_ok=True)

    if algorithm == "pymupdf":
        try:
            import fitz  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Algorithm 'pymupdf' requires the 'pymupdf' package.") from exc

        scale = dpi / 72
        saved_paths: List[Path] = []
        doc = fitz.open(resolved_pdf)
        try:
            for page_index in range(doc.page_count):
                page = doc.load_page(page_index)
                pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
                output_path = output_dir / f"{resolved_pdf.stem}_page_{page_index + 1:04d}.jpg"
                pix.save(output_path.as_posix(), jpg_quality=jpeg_quality)
                saved_paths.append(output_path)
        finally:
            doc.close()
        return saved_paths

    if algorithm == "pdf2image":
        try:
            from pdf2image import convert_from_path
        except ImportError as exc:
            raise RuntimeError("Algorithm 'pdf2image' requires the 'pdf2image' package and Poppler.") from exc

        images = convert_from_path(resolved_pdf.as_posix(), dpi=dpi)
        saved_paths = []
        for index, image in enumerate(images, start=1):
            output_path = output_dir / f"{resolved_pdf.stem}_page_{index:04d}.jpg"
            image.save(output_path, format="JPEG", quality=jpeg_quality, optimize=True)
            saved_paths.append(output_path)
        return saved_paths

    raise ValueError(f"Unsupported algorithm: {algorithm}")


# %%
PDF_PATH = "../../DATA/PDFS/CVSPharmacyRateSchedule.pdf"
PDF_PATH = "../../DATA/PDFS/Walgreens.pdf"
PDF_OUTPUT_DIR = "../../DATA/PDFS/Walgreens_pages"
PDF_ALGORITHM = "pymupdf"
PDF_DPI = 200
PDF_JPEG_QUALITY = 90

converted_pages: List[Path] = []
if PDF_PATH is not None:
    converted_pages = convert_pdf_to_images(
        Path(PDF_PATH),
        Path(PDF_OUTPUT_DIR) if PDF_OUTPUT_DIR is not None else None,
        algorithm=PDF_ALGORITHM,
        dpi=PDF_DPI,
        jpeg_quality=PDF_JPEG_QUALITY,
    )
converted_pages


# %%
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
artifacts = load_vlm_artifacts(CONFIG["model_path"])


# %%
TARGET_IMAGE_PATH = "../../DATA/TABLE_IMAGES/prior-auth-example.png"
PROMPT_TEMPLATE = CONFIG["prompt_template"]
MAX_TOKENS = CONFIG["max_tokens"]


# %%
if TARGET_IMAGE_PATH is None:
    raise ValueError("Set TARGET_IMAGE_PATH to an actual image path before running this cell.")

result = generate_text_for_image(
    artifacts,
    image_path=Path(TARGET_IMAGE_PATH),
    prompt_template=PROMPT_TEMPLATE,
    max_tokens=MAX_TOKENS,
)

print(result.text)
result.elapsed_seconds

# %%
