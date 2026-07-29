"""
Image preprocessing for the HAM10000 skin-lesion classifier.

The model contract (see models/model_meta.json and the training notebook):

  * Input is a RAW uint8 RGB array of shape (N, 128, 128, 3), values 0-255.
  * Normalisation is done by a `Rescaling(1/127.5, offset=-1.0)` layer INSIDE
    the model graph. Do NOT divide by 255 or call any `preprocess_input` here.
  * Resize with PIL: open -> convert("RGB") -> resize((128, 128), BILINEAR),
    matching training exactly.

Everything a web process needs to turn an uploaded file (path, bytes, or a
directory of files) into arrays the model can consume lives here.
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Iterable, Tuple, Union

import numpy as np
from PIL import Image, UnidentifiedImageError

# ---------------------------------------------------------------------------
# Configuration (no hardcoded absolute paths; overridable by env var)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]

# The model was trained at 128x128. This is echoed by models/model_meta.json
# ("img_size": 128) but we keep a safe default so preprocessing never crashes
# if the meta file is missing.
IMG_SIZE = int(os.environ.get("IMG_SIZE", "128"))

# Reject anything larger than this to avoid a decompression-bomb / OOM upload.
MAX_FILE_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))

# File extensions we are willing to try to decode.
_ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# Subfolder names that encode a class label.
_BENIGN_NAMES = {"benign", "0", "neg", "negative"}
_MALIGNANT_NAMES = {"malignant", "1", "pos", "positive", "mel", "malig"}

ImageSource = Union[str, os.PathLike, bytes, bytearray, io.IOBase]


class PreprocessError(ValueError):
    """Raised for user-facing preprocessing problems (bad file, too big, ...)."""


# ---------------------------------------------------------------------------
# Single image
# ---------------------------------------------------------------------------
def load_image(source: ImageSource, *, max_bytes: int = MAX_FILE_BYTES) -> np.ndarray:
    """Load one image into a raw uint8 RGB array of shape (IMG_SIZE, IMG_SIZE, 3).

    Accepts a filesystem path, raw ``bytes``/``bytearray``, or a binary
    file-like object. The returned array is what the model expects directly:
    values in 0-255, NO normalisation applied here.

    Raises PreprocessError for oversize input, non-images, or unreadable data.
    """
    raw = _read_bytes(source, max_bytes=max_bytes)

    try:
        with Image.open(io.BytesIO(raw)) as img:
            img = img.convert("RGB")
            img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
            arr = np.asarray(img, dtype=np.uint8)
    except (UnidentifiedImageError, OSError) as exc:
        raise PreprocessError(f"File is not a readable image: {exc}") from exc

    if arr.shape != (IMG_SIZE, IMG_SIZE, 3):
        raise PreprocessError(
            f"Decoded image has unexpected shape {arr.shape}, "
            f"expected ({IMG_SIZE}, {IMG_SIZE}, 3)"
        )
    return arr


def load_batch(sources: Iterable[ImageSource]) -> np.ndarray:
    """Stack several images into a (N, IMG_SIZE, IMG_SIZE, 3) uint8 array."""
    imgs = [load_image(s) for s in sources]
    if not imgs:
        raise PreprocessError("No images supplied")
    return np.stack(imgs, axis=0).astype(np.uint8)


# ---------------------------------------------------------------------------
# Directory of images -> (X, y) for retraining
# ---------------------------------------------------------------------------
def build_array_from_dir(
    directory: Union[str, os.PathLike],
) -> Tuple[np.ndarray, np.ndarray]:
    """Walk ``directory`` and build (X, y) arrays for retraining.

    Labels are resolved, in order of preference, from:
      1. the filename prefix ``0_xxx.jpg`` / ``1_xxx.jpg``, or
      2. a parent subfolder named ``benign/`` / ``malignant/`` (also accepts
         ``0``/``1``, ``mel``, etc.).

    Files that are not images, exceed the size cap, or whose label cannot be
    resolved are skipped (they do not abort the whole batch). Raises
    PreprocessError only if nothing usable was found.

    Returns
    -------
    X : np.uint8, shape (N, IMG_SIZE, IMG_SIZE, 3)
    y : np.int64, shape (N,)
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise PreprocessError(f"Not a directory: {directory}")

    xs: list[np.ndarray] = []
    ys: list[int] = []
    skipped = 0

    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in _ALLOWED_EXTS:
            skipped += 1
            continue
        label = parse_label(path)
        if label is None:
            skipped += 1
            continue
        try:
            xs.append(load_image(path))
            ys.append(label)
        except PreprocessError:
            skipped += 1

    if not xs:
        raise PreprocessError(
            f"No labelled images found under {directory} "
            f"(skipped {skipped}). Use 0_/1_ filename prefixes or "
            f"benign/ and malignant/ subfolders."
        )

    X = np.stack(xs, axis=0).astype(np.uint8)
    y = np.asarray(ys, dtype=np.int64)
    return X, y


def parse_label(path: Union[str, os.PathLike]) -> int | None:
    """Return 0 (benign) or 1 (malignant) for a file, or None if unresolvable."""
    path = Path(path)

    # 1. filename prefix: "0_xxx.jpg" / "1_xxx.jpg"
    stem = path.name
    if len(stem) >= 2 and stem[1] == "_" and stem[0] in ("0", "1"):
        return int(stem[0])

    # 2. any ancestor directory whose name encodes the class
    for part in reversed(path.parts[:-1]):
        name = part.strip().lower()
        if name in _MALIGNANT_NAMES:
            return 1
        if name in _BENIGN_NAMES:
            return 0
    return None


def count_by_class(y: np.ndarray) -> dict[str, int]:
    """Human-readable per-class counts for a label array."""
    y = np.asarray(y)
    return {
        "benign": int((y == 0).sum()),
        "malignant": int((y == 1).sum()),
        "total": int(y.size),
    }


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------
def _read_bytes(source: ImageSource, *, max_bytes: int) -> bytes:
    """Coerce any supported source into a size-checked ``bytes`` blob."""
    if isinstance(source, (bytes, bytearray)):
        raw = bytes(source)
    elif isinstance(source, (str, os.PathLike)):
        p = Path(source)
        if not p.is_file():
            raise PreprocessError(f"File not found: {p}")
        size = p.stat().st_size
        if size > max_bytes:
            raise PreprocessError(
                f"File too large: {size} bytes > {max_bytes} byte limit"
            )
        raw = p.read_bytes()
    elif hasattr(source, "read"):
        raw = source.read()
        if not isinstance(raw, (bytes, bytearray)):
            raise PreprocessError("File-like object did not yield bytes")
        raw = bytes(raw)
    else:
        raise PreprocessError(f"Unsupported image source type: {type(source)!r}")

    if len(raw) == 0:
        raise PreprocessError("Empty file / no bytes")
    if len(raw) > max_bytes:
        raise PreprocessError(
            f"File too large: {len(raw)} bytes > {max_bytes} byte limit"
        )
    return raw
