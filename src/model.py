"""
Model loading, metadata access, and the retrain / promotion guardrail.

Design notes
------------
* The trained Keras model already contains its own `Rescaling` and augmentation
  layers, so we always feed it RAW uint8 arrays (0-255). Nothing in this module
  normalises pixels.
* `load()` caches the model at module level so the ~21 MB file is deserialised
  once per process (a web worker imports this once, not per request).
* `retrain()` is self-contained: it takes the new data as explicit arguments and
  loads the held-out test cache from disk. It does NOT rely on any notebook
  globals (the notebook version read X_train/val_ds/test_ds/metrics from the
  global namespace, which does not exist in a web process).
* Promotion guardrail: a freshly fine-tuned candidate is only written over the
  live model file if its test ROC-AUC has not regressed by more than
  MAX_AUC_REGRESSION (default 0.005) versus the current live model.
* The guardrail is fail-closed. The candidate and the live model are always
  scored on the SAME rows, and those rows are always held out of the fit. If
  the evaluation cannot be computed at all, the candidate is REJECTED -- an
  unmeasurable candidate is not a passing one.
* Evaluation set: the frozen test cache when it exists, otherwise a stratified
  holdout carved out of the uploaded batch *before* training.
"""

from __future__ import annotations

import json
import os
import shutil
import threading
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Paths & config (relative to repo root, overridable by env var)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = Path(os.environ.get("MODELS_DIR", REPO_ROOT / "models"))
DATA_DIR = Path(os.environ.get("DATA_DIR", REPO_ROOT / "data"))
CACHE_DIR = Path(os.environ.get("CACHE_DIR", DATA_DIR / "cache"))

MODEL_KERAS_PATH = Path(
    os.environ.get("MODEL_PATH", MODELS_DIR / "skin_lesion_model.keras")
)
MODEL_META_PATH = Path(
    os.environ.get("MODEL_META_PATH", MODELS_DIR / "model_meta.json")
)
TEST_CACHE_PATH = Path(
    os.environ.get("TEST_CACHE_PATH", CACHE_DIR / "test_cache.npz")
)
METADATA_CSV = Path(
    os.environ.get("METADATA_CSV", DATA_DIR / "metadata_split.csv")
)
# Where the raw HAM10000 .jpg images live locally. The `path` column in the CSV
# stores Colab paths (/content/data/raw/images/...) that do not exist off-Colab,
# so the test-cache builder resolves images by `image_id` under this directory.
RAW_IMAGES_DIR = Path(
    os.environ.get("RAW_IMAGES_DIR", DATA_DIR / "raw" / "images")
)

# Retrain knobs (demo-friendly defaults for a CPU-only container)
RETRAIN_EPOCHS = int(os.environ.get("RETRAIN_EPOCHS", "3"))
RETRAIN_SUBSAMPLE = int(os.environ.get("RETRAIN_SUBSAMPLE", "1500"))
RETRAIN_LR = float(os.environ.get("RETRAIN_LR", "1e-4"))
MAX_AUC_REGRESSION = float(os.environ.get("MAX_AUC_REGRESSION", "0.005"))
# Fraction of the uploaded batch held out for scoring when there is no frozen
# test cache on disk.
HOLDOUT_FRACTION = float(os.environ.get("RETRAIN_HOLDOUT_FRACTION", "0.2"))
# Below this many evaluation rows, a ROC-AUC comparison is mostly sampling
# noise: a handful of rows can swing it by tenths. The decision is still made
# (refusing outright would block every small demo batch), but the report says so
# out loud rather than presenting a coin flip as a verdict.
MIN_CONFIDENT_EVAL_ROWS = int(os.environ.get("MIN_CONFIDENT_EVAL_ROWS", "30"))

DEFAULT_THRESHOLD = 0.5  # only used if meta is unreadable

# ---------------------------------------------------------------------------
# Module-level cache
# ---------------------------------------------------------------------------
_MODEL = None
_MODEL_MTIME: Optional[float] = None
_LOAD_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
def get_meta() -> dict:
    """Load models/model_meta.json. Falls back to sane defaults if absent."""
    try:
        with open(MODEL_META_PATH, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        return {
            "img_size": 128,
            "classes": {"0": "benign", "1": "malignant"},
            "operating_threshold": DEFAULT_THRESHOLD,
            "test_metrics": {},
        }


def get_threshold() -> float:
    """Operating decision threshold. NEVER hardcode this at call sites."""
    return float(get_meta().get("operating_threshold", DEFAULT_THRESHOLD))


def get_test_metrics() -> dict:
    return dict(get_meta().get("test_metrics", {}))


def model_file_mtime() -> Optional[float]:
    """POSIX mtime of the live model file, or None if it does not exist."""
    try:
        return MODEL_KERAS_PATH.stat().st_mtime
    except FileNotFoundError:
        return None


# ---------------------------------------------------------------------------
# Loading (cached)
# ---------------------------------------------------------------------------
def load(path: Optional[os.PathLike] = None, *, force: bool = False):
    """Return the trained Keras model, cached per process.

    The cache is invalidated automatically when the underlying file's mtime
    changes (e.g. after a successful retrain promotion), so long-lived workers
    pick up a promoted model without a restart.
    """
    global _MODEL, _MODEL_MTIME

    model_path = Path(path) if path is not None else MODEL_KERAS_PATH
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    current_mtime = model_path.stat().st_mtime

    with _LOAD_LOCK:
        if force or _MODEL is None or _MODEL_MTIME != current_mtime:
            _MODEL = _load_from_disk(model_path)
            _MODEL_MTIME = current_mtime
        return _MODEL


def _load_from_disk(model_path: Path):
    # Imported lazily so simply importing this module (e.g. in tests that only
    # touch metadata) does not pull in TensorFlow.
    import tensorflow as tf  # noqa: WPS433

    return tf.keras.models.load_model(model_path, compile=False)


def is_loaded() -> bool:
    """True if the model is already deserialised in this process's cache."""
    return _MODEL is not None


def clear_cache() -> None:
    """Drop the cached model (forces a reload on next `load()`)."""
    global _MODEL, _MODEL_MTIME
    with _LOAD_LOCK:
        _MODEL = None
        _MODEL_MTIME = None


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------
def _load_test_cache() -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Load the held-out test cache (raw uint8 X, int y), or None if absent."""
    if not TEST_CACHE_PATH.exists():
        return None
    with np.load(TEST_CACHE_PATH) as data:
        return data["X"].astype(np.uint8), data["y"].astype(np.int64)


def _resolve_image_path(csv_path: str, image_id: str) -> Optional[Path]:
    """Find a lesion image on the local disk.

    Tries, in order: the literal path from the CSV; RAW_IMAGES_DIR/<image_id>
    with a handful of common extensions; and the CSV path's basename under
    RAW_IMAGES_DIR. Returns None if nothing exists.
    """
    candidates: list[Path] = []
    if csv_path:
        candidates.append(Path(csv_path))
        candidates.append(RAW_IMAGES_DIR / Path(csv_path).name)
    if image_id:
        for ext in (".jpg", ".jpeg", ".png"):
            candidates.append(RAW_IMAGES_DIR / f"{image_id}{ext}")
    for cand in candidates:
        if cand.is_file():
            return cand
    return None


def build_test_cache(
    *,
    metadata_csv: Optional[os.PathLike] = None,
    out_path: Optional[os.PathLike] = None,
    limit: Optional[int] = None,
) -> dict:
    """Build data/cache/test_cache.npz from the held-out test split.

    Reads the `split == "test"` rows of metadata_split.csv, loads each image as
    a raw uint8 (128,128,3) array, and saves a compressed npz with arrays
    ``X`` (uint8, N x 128 x 128 x 3) and ``y`` (int, from the `label` column).
    This is the frozen test set the retrain promotion guardrail evaluates on.

    Images are resolved by `image_id` under RAW_IMAGES_DIR (the CSV's own
    `path` column holds Colab paths). Rows whose image is missing are skipped.

    Returns a small report; raises if no test images could be loaded.
    """
    import csv  # stdlib, lazy

    from .preprocessing import PreprocessError, load_image  # avoid import cycle

    csv_path = Path(metadata_csv) if metadata_csv else METADATA_CSV
    dest = Path(out_path) if out_path else TEST_CACHE_PATH

    if not csv_path.is_file():
        raise FileNotFoundError(f"Metadata CSV not found: {csv_path}")

    xs: list[np.ndarray] = []
    ys: list[int] = []
    missing = 0
    unreadable = 0
    seen = 0

    with open(csv_path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if (row.get("split") or "").strip() != "test":
                continue
            seen += 1
            img_path = _resolve_image_path(
                (row.get("path") or "").strip(), (row.get("image_id") or "").strip()
            )
            if img_path is None:
                missing += 1
                continue
            try:
                xs.append(load_image(img_path))
                ys.append(int(row["label"]))
            except (PreprocessError, KeyError, ValueError):
                unreadable += 1
            if limit and len(xs) >= limit:
                break

    if not xs:
        raise FileNotFoundError(
            f"No test images could be loaded. Looked for {seen} test rows under "
            f"RAW_IMAGES_DIR={RAW_IMAGES_DIR} (missing={missing}). "
            f"Set RAW_IMAGES_DIR to the folder containing the HAM10000 .jpg files."
        )

    X = np.stack(xs, axis=0).astype(np.uint8)
    y = np.asarray(ys, dtype=np.int64)

    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, X=X, y=y)

    report = {
        "out_path": str(dest),
        "test_rows_seen": seen,
        "images_loaded": int(X.shape[0]),
        "missing": missing,
        "unreadable": unreadable,
        "benign": int((y == 0).sum()),
        "malignant": int((y == 1).sum()),
    }
    return report


def _stratified_holdout(
    X: np.ndarray, y: np.ndarray, frac: float, seed: int
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Split (X, y) into (X_fit, y_fit, X_hold, y_hold), stratified by class.

    Sampling is per-class so BOTH sides of the split always contain both
    classes: a holdout with one class has an undefined ROC-AUC, and a
    single-class training set teaches the model to answer a constant.

    Callers must not assume the incoming order is random. `build_array_from_dir`
    walks the batch in sorted filename order, so every `0_*` file precedes every
    `1_*` -- slicing the first 20% off the front would hand back an all-benign
    holdout every single time.

    Returns None if any class has fewer than 2 examples, i.e. the batch cannot
    give up a row for scoring and still have one left to train on.
    """
    rng = np.random.default_rng(seed)
    fit_idx: list[int] = []
    hold_idx: list[int] = []

    for cls in (0, 1):
        idx = np.flatnonzero(y == cls)
        if idx.size < 2:
            return None
        rng.shuffle(idx)
        # At least 1 held out, but never so many that training loses the class.
        n_hold = min(idx.size - 1, max(1, int(round(frac * idx.size))))
        hold_idx.extend(idx[:n_hold].tolist())
        fit_idx.extend(idx[n_hold:].tolist())

    fit = np.asarray(fit_idx, dtype=np.int64)
    hold = np.asarray(hold_idx, dtype=np.int64)
    rng.shuffle(fit)  # de-interleave the per-class blocks before batching
    return X[fit], y[fit], X[hold], y[hold]


def evaluate_roc_auc(model, X: np.ndarray, y: np.ndarray, *, batch_size: int = 64) -> float:
    """ROC-AUC of `model` on raw uint8 X against labels y."""
    from sklearn.metrics import roc_auc_score  # lazy import

    probs = model.predict(X, batch_size=batch_size, verbose=0).ravel()
    # A degenerate label set (all one class) has undefined ROC-AUC.
    if len(np.unique(y)) < 2:
        raise ValueError("Test set has a single class; ROC-AUC is undefined.")
    return float(roc_auc_score(y, probs))


def full_test_metrics(model, X: np.ndarray, y: np.ndarray, threshold: float) -> dict:
    """Compute the metric bundle stored in model_meta.json for a given model."""
    from sklearn.metrics import (  # lazy import
        accuracy_score,
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    probs = model.predict(X, batch_size=64, verbose=0).ravel()
    preds = (probs >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y, preds)),
        "precision": float(precision_score(y, preds, zero_division=0)),
        "recall": float(recall_score(y, preds, zero_division=0)),
        "f1": float(f1_score(y, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probs)),
        "pr_auc": float(average_precision_score(y, probs)),
    }


# ---------------------------------------------------------------------------
# Retrain + promotion guardrail
# ---------------------------------------------------------------------------
def retrain(
    new_X: np.ndarray,
    new_y: np.ndarray,
    *,
    epochs: int = RETRAIN_EPOCHS,
    subsample: int = RETRAIN_SUBSAMPLE,
    base_model_path: Optional[os.PathLike] = None,
    progress_cb: Optional[Callable[[dict], None]] = None,
) -> dict:
    """Fine-tune the EXISTING custom model on new data and gate its promotion.

    The rubric requires the custom model itself to be the pre-trained starting
    point, so we load `base_model_path` (the live .keras file) rather than fresh
    ImageNet weights.

    Parameters
    ----------
    new_X : uint8 (N, H, W, 3), raw pixels 0-255.
    new_y : int (N,), 0/1 labels.
    epochs, subsample : capped for a CPU-only demo; both env-configurable.
    progress_cb : optional callback receiving dicts like
        {"stage": "train", "epoch": 1, "epochs": 3, "logs": {...}}.

    Returns a JSON-serialisable report describing whether the candidate was
    promoted, and the before/after test ROC-AUC.
    """
    import tensorflow as tf  # noqa: WPS433

    base_path = Path(base_model_path) if base_model_path else MODEL_KERAS_PATH
    new_X = np.asarray(new_X, dtype=np.uint8)
    new_y = np.asarray(new_y, dtype=np.int64)

    if new_X.ndim != 4 or new_X.shape[0] == 0:
        raise ValueError(f"new_X must be (N,H,W,3); got shape {new_X.shape}")
    if new_X.shape[0] != new_y.shape[0]:
        raise ValueError("new_X and new_y length mismatch")
    if len(np.unique(new_y)) < 2:
        raise ValueError(
            "Retrain data has a single class; provide both benign and "
            "malignant examples."
        )

    def emit(payload: dict) -> None:
        if progress_cb is not None:
            progress_cb(payload)

    meta = get_meta()
    seed = int(meta.get("seed", 42))

    # --- subsample so a demo retrain finishes quickly -----------------------
    n = new_X.shape[0]
    if subsample and n > subsample:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=subsample, replace=False)
        new_X, new_y = new_X[idx], new_y[idx]

    # --- choose the evaluation set BEFORE training --------------------------
    # With a frozen test cache we fine-tune on the whole batch and score against
    # that. Without one we must carve a holdout out of the batch and keep it out
    # of the fit: scoring a candidate on rows it just trained on measures
    # memorisation, not generalisation, and would wave through any model at all.
    test_cache = _load_test_cache()
    fit_X, fit_y = new_X, new_y
    hold_X = hold_y = None

    if test_cache is None:
        split = _stratified_holdout(new_X, new_y, HOLDOUT_FRACTION, seed)
        if split is None:
            raise ValueError(
                "No frozen test set on disk, and this batch is too small to "
                "carve an evaluation holdout from (need at least 2 benign and 2 "
                "malignant images). Upload more data, or build the frozen test "
                "set once with `python -m src.model build-test-cache`."
            )
        fit_X, fit_y, hold_X, hold_y = split

    emit({
        "stage": "prepared",
        "n_train": int(fit_X.shape[0]),
        "n_holdout": 0 if hold_y is None else int(hold_y.size),
        "eval_source": "test_cache" if test_cache is not None else "new_data_holdout",
    })

    # --- load the existing custom model as the starting point ---------------
    emit({"stage": "loading_base", "base_model_path": str(base_path)})
    candidate = tf.keras.models.load_model(base_path, compile=False)

    class_weight = {int(k): float(v) for k, v in meta.get("class_weight", {}).items()} or None

    candidate.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=RETRAIN_LR),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc")],
    )

    class _Progress(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):  # noqa: D401
            emit(
                {
                    "stage": "train",
                    "epoch": int(epoch) + 1,
                    "epochs": int(epochs),
                    "logs": {k: float(v) for k, v in (logs or {}).items()},
                }
            )

    emit({"stage": "train", "epoch": 0, "epochs": int(epochs), "logs": {}})
    candidate.fit(
        fit_X,
        fit_y.astype("float32"),
        epochs=int(epochs),
        batch_size=32,
        class_weight=class_weight,
        verbose=0,
        callbacks=[_Progress()],
    )

    # --- score candidate AND live model on the same held-out rows -----------
    threshold = get_threshold()

    if test_cache is not None:
        X_test, y_test = test_cache
        eval_source = "test_cache"
        emit({"stage": "evaluating", "n_test": int(len(y_test)),
              "eval_source": eval_source})
    else:
        X_test, y_test = hold_X, hold_y
        eval_source = "new_data_holdout"
        emit({"stage": "evaluating", "n_test": int(len(y_test)),
              "eval_source": eval_source,
              "note": "no test cache; scoring on a stratified holdout that was "
                      "kept out of training"})

    report = {
        "epochs": int(epochs),
        "n_train": int(fit_X.shape[0]),
        "n_eval": int(len(y_test)),
        "eval_source": eval_source,
        "threshold": threshold,
        "max_allowed_regression": MAX_AUC_REGRESSION,
    }

    # The base figure is the CURRENT LIVE model measured on these same rows,
    # never a number copied out of meta -- otherwise the comparison is against a
    # score computed on different data by a different model.
    try:
        live_model = load()
        base_auc = evaluate_roc_auc(live_model, X_test, y_test)
        new_auc = evaluate_roc_auc(candidate, X_test, y_test)
        candidate_metrics = full_test_metrics(candidate, X_test, y_test, threshold)
    except (ValueError, FileNotFoundError) as exc:
        # Fail closed. "Could not be measured" is not "did not regress"; treating
        # the two as equivalent is exactly how a bad model silently replaces a
        # good one. The roc_auc keys are omitted rather than set to None so that
        # consumers formatting them as floats still get their own default.
        eval_error = f"{type(exc).__name__}: {exc}"
        report.update({
            "promoted": False,
            "candidate_metrics": {},
            "eval_error": eval_error,
            "message": (
                f"Rejected: the candidate could not be scored ({eval_error}). "
                f"Kept previous model."
            ),
        })
        emit({"stage": "done", "report": report})
        return report

    regression = base_auc - new_auc
    promoted = regression <= MAX_AUC_REGRESSION

    report.update({
        "promoted": bool(promoted),
        "base_roc_auc": round(base_auc, 4),
        "new_roc_auc": round(new_auc, 4),
        "regression": round(regression, 4),
        "candidate_metrics": candidate_metrics,
    })

    if len(y_test) < MIN_CONFIDENT_EVAL_ROWS:
        report["low_confidence"] = True
        report["eval_warning"] = (
            f"Only {len(y_test)} evaluation rows (< {MIN_CONFIDENT_EVAL_ROWS}); "
            f"this ROC-AUC comparison is dominated by sampling noise. Build the "
            f"frozen test set with `python -m src.model build-test-cache` for a "
            f"decision you can actually rely on."
        )

    if promoted:
        emit({"stage": "promoting"})
        _promote(candidate, base_path, candidate_metrics, eval_source)
        report["message"] = (
            f"Promoted: ROC-AUC {new_auc:.4f} vs live model's {base_auc:.4f} on "
            f"{eval_source} (regression {regression:+.4f} within "
            f"{MAX_AUC_REGRESSION})."
        )
    else:
        report["message"] = (
            f"Rejected: ROC-AUC regressed {regression:.4f} on {eval_source} "
            f"(> {MAX_AUC_REGRESSION}). Kept previous model."
        )

    emit({"stage": "done", "report": report})
    return report


def _promote(
    candidate, base_path: Path, candidate_metrics: dict, eval_source: str = "test_cache"
) -> None:
    """Write the promoted model over the live file (with backup) and update meta."""
    base_path = Path(base_path)
    backup = base_path.with_suffix(base_path.suffix + ".bak")

    # Back up the previous live model so a bad promotion is recoverable.
    if base_path.exists():
        shutil.copy2(base_path, backup)

    # Save atomically: write to a temp file, then replace. The temp path MUST
    # keep the .keras suffix — Keras 3 validates the extension on save().
    tmp = base_path.with_name(f"{base_path.stem}.promote_tmp{base_path.suffix}")
    candidate.save(tmp)
    os.replace(tmp, base_path)

    # Best-effort refresh of the legacy .h5. Full-model HDF5 save can fail for
    # this graph on Keras 3 ("cannot pickle module"), so save to a temp file and
    # only replace the existing .h5 on SUCCESS — never leave it corrupted. The
    # .keras file is the authoritative artefact regardless.
    h5_path = base_path.with_suffix(".h5")
    h5_tmp = base_path.with_name(f"{base_path.stem}.promote_tmp.h5")
    try:
        candidate.save(h5_tmp)
        os.replace(h5_tmp, h5_path)
    except Exception as exc:  # noqa: BLE001
        try:
            h5_tmp.unlink(missing_ok=True)
        except OSError:
            pass
        print(f"[promote] .h5 refresh skipped ({type(exc).__name__}: {exc}); "
              f"kept existing .h5; .keras is authoritative.")

    # Update model_meta.json. The timestamp and the provenance of the decision
    # are recorded on every promotion, so /metrics can never show a stale
    # "retrained_at: null" for a model that has in fact been replaced.
    meta = get_meta()
    meta["retrained_at"] = datetime.now().isoformat(timespec="seconds")
    meta["last_eval_source"] = eval_source
    if candidate_metrics:
        if eval_source == "test_cache":
            meta["test_metrics"] = candidate_metrics
        else:
            # Holdout scores come from a handful of rows of one uploaded batch.
            # They are not comparable to the frozen test-set numbers, so they go
            # in their own key instead of quietly overwriting the headline
            # metrics with something measured on different, much smaller data.
            meta["holdout_metrics"] = candidate_metrics
    with open(MODEL_META_PATH, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    clear_cache()  # next load() picks up the new mtime


# ---------------------------------------------------------------------------
# CLI:  python -m src.model build-test-cache [--limit N]
# ---------------------------------------------------------------------------
def _main(argv: Optional[list[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="HAM10000 model utilities")
    sub = parser.add_subparsers(dest="command", required=True)

    p_cache = sub.add_parser(
        "build-test-cache",
        help="Build data/cache/test_cache.npz from the test split for the "
        "retrain promotion guardrail.",
    )
    p_cache.add_argument("--metadata-csv", default=None)
    p_cache.add_argument("--out", default=None)
    p_cache.add_argument("--limit", type=int, default=None,
                         help="Cap number of images (for a quick smoke build).")

    args = parser.parse_args(argv)

    if args.command == "build-test-cache":
        report = build_test_cache(
            metadata_csv=args.metadata_csv, out_path=args.out, limit=args.limit
        )
        print(json.dumps(report, indent=2))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(_main())
