"""
Single-image inference for the HAM10000 skin-lesion classifier.

The model outputs a single sigmoid = P(malignant). Class 0 = benign,
1 = malignant. The decision threshold is read from model_meta.json
(currently 0.39) and is NEVER hardcoded here.

Remember: the model normalises pixels internally, so we hand it the raw uint8
array produced by preprocessing.load_image untouched.
"""

from __future__ import annotations

import time
from typing import Union

import numpy as np

from . import model as model_module
from .preprocessing import ImageSource, PreprocessError, load_image


def predict(source: Union[ImageSource, np.ndarray]) -> dict:
    """Classify one image.

    Parameters
    ----------
    source : a file path, raw bytes, a binary file-like object, OR an
        already-decoded raw uint8 (H, W, 3) array.

    Returns
    -------
    dict with keys:
        label                 : "benign" | "malignant"
        probability_malignant : float in [0, 1]
        threshold             : the operating threshold applied
        latency_ms            : inference wall-time (model.predict only)
    """
    # Accept a pre-decoded array (used by batch paths) or decode from source.
    if isinstance(source, np.ndarray):
        arr = np.asarray(source, dtype=np.uint8)
        if arr.ndim == 3:
            arr = arr[None, ...]
        if arr.ndim != 4:
            raise PreprocessError(
                f"Array input must be (H,W,3) or (N,H,W,3); got {arr.shape}"
            )
    else:
        arr = load_image(source)[None, ...]

    model = model_module.load()
    threshold = model_module.get_threshold()

    t0 = time.perf_counter()
    probs = model.predict(arr, verbose=0).ravel()
    latency_ms = (time.perf_counter() - t0) * 1000.0

    prob = float(probs[0])
    label = "malignant" if prob >= threshold else "benign"

    return {
        "label": label,
        "probability_malignant": round(prob, 6),
        "threshold": round(threshold, 6),
        "latency_ms": round(latency_ms, 2),
    }
