# HAM10000 Skin-Lesion Classifier — End-to-End ML Pipeline

A complete, deployable machine-learning pipeline that classifies dermatoscopic
skin-lesion images as **benign** or **malignant**, with a REST API, an interactive
UI, one-click **retraining**, containerised **horizontal scaling**, and a **Locust**
load test. Built on the [HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
dataset (10,015 images).

> **Non-tabular data requirement:** this project uses **images** (128×128 RGB
> dermatoscopy tiles), extending the tabular introductory summative to an image
> classification task.

---

## 🔗 Links

| Item | Link |
|------|------|
| 🎥 Video demo (YouTube) | _`<ADD YOUTUBE LINK>`_ |
| 🌐 Live deployment URL | _`<ADD CLOUD URL>`_ |
| 📦 GitHub repository | _`<ADD REPO URL>`_ |

---

## Project description

Given a dermatoscopic image, the model outputs a single sigmoid probability
**P(malignant)** and applies a tuned operating threshold to decide the label.
The system demonstrates the full ML lifecycle:

- **Offline** (in `notebook/ham10000_pipeline.ipynb`): data acquisition →
  preprocessing → model training (transfer learning on MobileNetV2) → evaluation
  with all classification metrics → threshold selection → artefact persistence.
- **Online** (this repo's `src/`, `app/`, Docker): a FastAPI service serving
  predictions, bulk upload, asynchronous retraining with a promotion guardrail,
  and live metrics; a Streamlit UI; nginx-load-balanced, scalable containers; and
  a Locust flood test.

### Model card

| Metric (held-out test set) | Value |
|---|---|
| ROC-AUC | **0.849** |
| PR-AUC | **0.600** |
| Recall (malignant) | 0.589 |
| Precision (malignant) | 0.551 |
| F1 | 0.570 |
| Accuracy | 0.827 |
| **Operating threshold** | **0.39** (not 0.5 — tuned for screening recall) |

- **Architecture:** MobileNetV2 (ImageNet, frozen base + fine-tuned top) → GAP →
  dropout → 1 sigmoid unit. Pixel rescaling `1/127.5, offset −1.0` and augmentation
  are **layers inside the model graph**.
- **Input contract:** raw **uint8 RGB (128,128,3), 0–255**. Do **not** pre-normalise —
  the model does it internally.
- **Classes:** malignant = {`akiec`, `bcc`, `mel`}; everything else benign.

---

## Repository structure

```
ML-PIPELINE-SUMMATIVE/
├── README.md
├── notebook/
│   └── ham10000_pipeline.ipynb      # full offline ML cycle (with outputs)
├── src/
│   ├── preprocessing.py             # image -> uint8 array; bulk/label parsing
│   ├── model.py                     # load (cached), retrain, promotion guardrail
│   └── prediction.py                # single-image predict
├── app/
│   ├── main.py                      # FastAPI service
│   └── ui.py                        # Streamlit UI (4 tabs)
├── models/                          # committed model artefacts
│   ├── skin_lesion_model.keras
│   ├── skin_lesion_model.h5
│   └── model_meta.json              # threshold, class map, test metrics
├── data/
│   ├── README.md                    # dataset provenance + how to get the jpgs
│   ├── metadata_split.csv           # 10,015 rows w/ train/val/test split + labels
│   ├── train/
│   │   ├── train_metadata.csv       # 5,988 rows (4,822 benign / 1,166 malignant)
│   │   └── images/                  # raw jpgs — downloaded, not committed
│   └── test/
│       ├── test_metadata.csv        # 2,023 rows (1,631 benign / 392 malignant)
│       └── images/                  # raw jpgs — downloaded, not committed
├── figures/                         # fig1..fig8 EDA & evaluation plots
├── docker/nginx.conf                # load balancer config
├── Dockerfile                       # single image for api + ui
├── docker-compose.yml               # api (scalable) + nginx + ui
├── locustfile.py                    # flood test (128×128 JPEG payload)
├── requirements.txt
├── jobs/                            # retrain job state (gitignored)
└── uploads/                         # uploaded batches (gitignored)
```

---

## Feature interpretations (the story the data tells)

Computed live in the UI's **Insights** tab and in the notebook (§4):

1. **Malignancy rate by anatomical site** — highly non-uniform. Sun-exposed sites
   (face ≈ 43% malignant) dominate; low-exposure sites (e.g. genital) are near 0%.
   *Localization carries real predictive signal (UV-driven carcinomas).*
2. **Malignancy rate by age band** — rises **monotonically** from ~6% (<30) to ~43%
   (70+), roughly a 7× increase. *Age is a clinically expected risk factor.*
3. **Class distribution** — the dataset is **imbalanced**: only **19.5%** of lesions
   are malignant. *This is why the model uses class weights, is evaluated with
   PR-AUC/recall, and operates at threshold 0.39 — a naive "always benign" model
   would score 80% accuracy while catching zero cancers.*

---

## Setup

### Option A — Docker (recommended)

Requires Docker + Docker Compose.

```bash
# Build and start 1 API replica + nginx + UI
docker compose up --build

# API / Swagger docs : http://localhost:8000/docs
# Streamlit UI       : http://localhost:8501
```

Scale the API horizontally (for the load test) behind the nginx balancer:

```bash
docker compose up --scale api=4          # 4 API containers on one port (8000)
```

### Option B — Local (no Docker)

Use a fresh virtual environment (Python 3.11 recommended).

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

# Terminal 1 — API
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Terminal 2 — UI
set API_URL=http://localhost:8000        # Windows (PowerShell: $env:API_URL=...)
export API_URL=http://localhost:8000     # macOS/Linux
streamlit run app/ui.py
```

### One-command demo

```bash
./run_demo.sh        # or: make demo   (starts everything via Docker Compose)
```

---

## API endpoints

Base URL: `http://localhost:8000`

| Method | Path | Description |
|---|---|---|
| GET | `/` | Redirect to `/docs` |
| GET | `/health` | Status, model-loaded flag, uptime, model file mtime |
| POST | `/predict` | Multipart image → `{label, probability_malignant, threshold, latency_ms}` |
| POST | `/upload` | Bulk images (multipart list **or** `.zip`) → `{batch_id, counts}` |
| POST | `/retrain` | Body `{batch_id, epochs?, subsample?}` → `{job_id}` (returns immediately) |
| GET | `/status/{job_id}` | `queued`/`running`/`done`/`failed` + progress + final report |
| GET | `/metrics` | Test metrics + request count / mean & p95 latency since boot |

Full interactive schema at `/docs`. A copy-paste `curl` walkthrough of every
endpoint is in [`test_api.sh`](./test_api.sh).

---

## UI (Streamlit) — four tabs

1. **Status** — model up-time, model version (file mtime), current test metrics,
   request count and p95 latency (from `/health` + `/metrics`).
2. **Insights** — the three feature interpretations above (live from
   `metadata_split.csv`) plus the eight evaluation figures.
3. **Predict** — upload one image → label, P(malignant), threshold, latency.
4. **Retrain** — bulk upload (per-class counts) → **Start retraining** button →
   live progress → before/after test ROC-AUC and promotion decision.

The UI communicates with the API **over HTTP only** (`API_URL`) and never imports
the model — the two run as independent containers.

---

## Retraining & the promotion guardrail

1. Upload a labelled batch (`/upload`): images named `0_*`/`1_*` **or** placed in
   `benign/` and `malignant/` subfolders; a `.zip` is accepted.
2. Trigger retraining (`/retrain` or the UI button). The request returns a
   `job_id` **immediately**; the heavy work runs in a background thread (so a proxy
   timeout can't kill it), guarded by a lock file against concurrent retrains.
3. The **existing custom model** is loaded as the starting point (per the rubric),
   fine-tuned on the new data, and evaluated on the held-out test cache.
4. **Guardrail:** the candidate is promoted only if test ROC-AUC has not regressed
   by more than **0.005**; otherwise the previous model is kept (a `.bak` backup is
   written before any overwrite).

Retrain is capped for CPU containers (default 3 epochs, ≤1,500 images) — all
configurable via `RETRAIN_EPOCHS`, `RETRAIN_SUBSAMPLE`, `MAX_AUC_REGRESSION`.

> **Optional:** to enable the full test-set guardrail (instead of a holdout
> fallback), place the raw HAM10000 `.jpg` files under `data/raw/images/` and run
> `python -m src.model build-test-cache` to generate `data/cache/test_cache.npz`.

---

## Load testing (Locust) — flood simulation

The Locust payload is a **pre-resized 128×128 JPEG (~12 KB)**, so the test measures
model inference latency, not upload bandwidth.

```bash
# 1) Start N API containers
docker compose up --scale api=1     # then re-run with =2, =4

# 2) Run Locust against the load balancer (port 8000)
locust -f locustfile.py --host http://localhost:8000
#    → open http://localhost:8089

# ...or headless:
locust -f locustfile.py --host http://localhost:8000 \
       --users 50 --spawn-rate 10 --run-time 2m --headless
```

### Results

Measured on the reference machine (Docker Desktop, **8 logical CPUs / 4 GB RAM**),
each API replica capped at **1.5 CPU**, `POST /predict` with a 12 KB 128×128 JPEG,
40 users / 90 s per run:

| Containers | Users | RPS (/predict) | Median latency (ms) | p95 latency (ms) | Failure rate |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 40 | 10.2 | 3300 | 4300 | 0% |
| 2 | 40 | 11.3 | 2700 | 5800 | 0% |
| 4 | 40 | 9.0  | 1600 | 13000 | 0% |

**Observation.** **Median (typical-request) latency scales down cleanly with more
replicas — 3300 → 2700 → 1600 ms — as the nginx balancer spreads work across
containers, with 0% failures throughout.** Aggregate throughput does *not* scale on
this host: the model is **CPU-bound** and a single 1.5-CPU replica already sits near
the machine's usable capacity, so extra replicas mainly reduce per-request latency
rather than raise RPS. The rising **p95 tail** is an artifact of two things specific
to this small single-host setup: (1) nginx's dynamic-DNS load balancing
(`resolver valid=10s`) periodically re-resolves the `api` service, adding occasional
stalls, and (2) only 8 cores shared between 4 replicas + the balancer + the load
generator + the OS. A definitive throughput-scaling test should be run on the **cloud
deployment**, where more vCPUs and separate hosts remove the single-machine ceiling —
there RPS is expected to scale roughly linearly with replica count.

> Reproduce: `docker compose up -d --scale api=N`, wait for all replicas healthy,
> then run the Locust command above. Numbers will vary with host CPU/RAM.

---

## Cloud deployment

The stack is portable to any Docker host (AWS ECS / EC2, Azure Container Apps,
GCP Cloud Run, Render, Railway, etc.). General steps:

1. Push the image (built from the `Dockerfile`) to a registry, **or** deploy the
   `docker-compose.yml` on a VM with Docker.
2. Expose port **8000** (API, via nginx) and **8501** (UI).
3. Set `API_URL` on the UI service to the public API URL.
4. Record the public URL in the **Links** table above and demonstrate the
   `/metrics` endpoint to show evaluation in production.

---

## Notebook

`notebook/ham10000_pipeline.ipynb` contains the complete offline cycle **with saved
outputs**: data acquisition (Kaggle), preprocessing & caching, lesion-aware
stratified split, MobileNetV2 model, training + fine-tuning, full evaluation
(accuracy, precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix, threshold
sweep), single-image prediction, and the retraining entry point. It saves the
model as `skin_lesion_model.keras` / `.h5` plus `model_meta.json`.

---

## Tech stack

TensorFlow-CPU 2.20 · FastAPI · Streamlit · nginx · Docker Compose · Locust ·
scikit-learn · Pillow · pandas.
