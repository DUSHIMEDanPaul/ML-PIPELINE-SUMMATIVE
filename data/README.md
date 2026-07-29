# Data

The HAM10000 dermatoscopic dataset (10,015 lesion images, 7 diagnostic classes)
collapsed to a **binary** target: `label = 1` for malignant/pre-malignant
(`akiec`, `bcc`, `mel`) and `0` otherwise.

## Layout

```
data/
├── metadata_split.csv        # master manifest — 10,015 rows, the source of truth
├── train/
│   ├── train_metadata.csv    # 5,988 rows (4,822 benign / 1,166 malignant)
│   └── images/               # raw .jpg files (not committed — see below)
├── test/
│   ├── test_metadata.csv     # 2,023 rows (1,631 benign / 392 malignant)
│   └── images/               # raw .jpg files (not committed — see below)
└── cache/
    └── test_cache.npz        # decoded test split for the retrain guardrail
```

The per-split CSVs are slices of `metadata_split.csv` by its `split` column and
carry the identical schema. The master file is kept in place because
`src/model.py` and `app/ui.py` both resolve it at `data/metadata_split.csv`.

The **validation** split (2,004 rows) is not broken out into its own folder; it
lives in `metadata_split.csv` under `split == "val"` and is recoverable with a
one-line filter.

## The images are not in this repository

All 10,015 jpgs total ~2.6 GB, well past what a git repository should carry, so
`images/` holds only a `.gitkeep`. To populate them, download the dataset from
either:

* **Harvard Dataverse** (original release, no account required) —
  <https://doi.org/10.7910/DVN/DBW86T>
* **Kaggle** — `kmader/skin-cancer-mnist-ham10000`

Both ship the images as `HAM10000_images_part_1.zip` and `..._part_2.zip`.

## Building the retrain guardrail's test set

The promotion guardrail scores a fine-tuned candidate against the current live
model on a frozen test set. That set is built from decoded pixels, so it needs
the actual jpgs — the CSVs alone are not sufficient.

Extract the archives so the files sit **flat** under `data/raw/images/`:

```
data/raw/images/ISIC_0027419.jpg
data/raw/images/ISIC_0025030.jpg
...
```

Then build the cache:

```bash
python -m src.model build-test-cache --limit 400
```

`--limit` caps how many images are decoded. 400 keeps a retrain to roughly a
minute on a CPU-only container; omit it to use all 2,023 test images, which is
more trustworthy but adds a couple of minutes per retrain since both the live
model and the candidate are scored across the whole set.

Images are resolved by `image_id`, and rows whose image is absent are skipped
and reported as `missing` — so `part_1.zip` on its own is usually enough for a
few-hundred-image cache.

> **Do not substitute `hmnist_28_28_RGB.csv`** from the Kaggle mirror. It is real
> pixel data, but at 28×28. The model expects 128×128 produced by a PIL BILINEAR
> resize of the original ~600×450 jpg (see `src/preprocessing.py`). Upscaled
> thumbnails do not match training preprocessing, and the resulting metrics would
> look authoritative while being meaningless.

## Without a test cache

Retrain still works. With no `test_cache.npz` on disk, `src/model.py` carves a
**stratified holdout out of the uploaded batch before training** and scores the
candidate and the live model on those held-out rows. Below 30 evaluation rows
the report is flagged `low_confidence`, because a ROC-AUC computed on a handful
of images is dominated by sampling noise. The frozen test set is what makes the
promotion decision meaningful.
