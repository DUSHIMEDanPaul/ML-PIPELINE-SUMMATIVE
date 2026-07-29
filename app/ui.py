"""
Streamlit UI for the HAM10000 skin-lesion classifier.

Four tabs: Status, Insights, Predict, Retrain.

The UI talks to the FastAPI service over HTTP only (API_URL env var, default
http://localhost:8000). It NEVER imports the model — API and UI run as separate
containers. The Insights tab reads data/metadata_split.csv directly, but that is
plain data analysis (no model), and it surfaces the pre-computed figures/.

Run:  streamlit run app/ui.py
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")
METADATA_CSV = Path(os.environ.get("METADATA_CSV", REPO_ROOT / "data" / "metadata_split.csv"))
FIGURES_DIR = Path(os.environ.get("FIGURES_DIR", REPO_ROOT / "figures"))

AGE_BAND_ORDER = ["<30", "30-39", "40-49", "50-59", "60-69", "70+"]
REQUEST_TIMEOUT = 30  # seconds for normal calls

st.set_page_config(page_title="HAM10000 Skin-Lesion Classifier", page_icon="🔬", layout="wide")


# ---------------------------------------------------------------------------
# API helpers (all return (ok, payload_or_error))
# ---------------------------------------------------------------------------
def api_get(path: str, timeout: int = REQUEST_TIMEOUT):
    try:
        r = requests.get(f"{API_URL}{path}", timeout=timeout)
        if r.status_code >= 400:
            return False, _err_text(r)
        return True, r.json()
    except requests.RequestException as exc:
        return False, f"Cannot reach API at {API_URL}{path}: {exc}"


def api_post(path: str, *, files=None, json=None, timeout: int = REQUEST_TIMEOUT):
    try:
        r = requests.post(f"{API_URL}{path}", files=files, json=json, timeout=timeout)
        if r.status_code >= 400:
            return False, _err_text(r)
        return True, r.json()
    except requests.RequestException as exc:
        return False, f"Cannot reach API at {API_URL}{path}: {exc}"


def _err_text(resp) -> str:
    try:
        body = resp.json()
        return body.get("detail") or body.get("error") or str(body)
    except ValueError:
        return f"HTTP {resp.status_code}: {resp.text[:300]}"


@st.cache_data(show_spinner=False)
def load_metadata() -> pd.DataFrame:
    df = pd.read_csv(METADATA_CSV)
    # `label` is already 0/1 (1 = malignant) in the split file.
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    return df.dropna(subset=["label"])


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🔬 HAM10000 Skin-Lesion Classifier")
st.caption(f"Binary benign / malignant classifier · API: `{API_URL}`")

tab_status, tab_insights, tab_predict, tab_retrain = st.tabs(
    ["📊 Status", "🔎 Insights", "🩺 Predict", "🔁 Retrain"]
)


# ===========================================================================
# TAB 1 — STATUS
# ===========================================================================
with tab_status:
    st.subheader("Service & model status")
    if st.button("↻ Refresh", key="refresh_status"):
        st.rerun()

    ok_h, health = api_get("/health")
    ok_m, metrics = api_get("/metrics")

    if not ok_h:
        st.error(f"Health check failed: {health}")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Status", health.get("status", "?"))
        c2.metric("Model loaded", "yes" if health.get("model_loaded") else "no")
        up = health.get("uptime_seconds", 0)
        c3.metric("Uptime", f"{up/60:.1f} min" if up >= 60 else f"{up:.0f} s")
        c4.metric("Model file present", "yes" if health.get("model_file_present") else "no")
        st.caption(f"Model file last modified (version): **{health.get('model_file_mtime_iso') or 'n/a'}**")

    if ok_m:
        st.markdown("#### Current test metrics")
        tm = metrics.get("model", {}).get("test_metrics", {})
        if tm:
            mc = st.columns(len(tm))
            for col, (name, val) in zip(mc, tm.items()):
                col.metric(name.upper(), f"{val:.3f}" if isinstance(val, (int, float)) else val)
        thr = metrics.get("model", {}).get("operating_threshold")
        st.caption(f"Operating threshold: **{thr}** (decisions use this, not 0.5)")

        st.markdown("#### Request telemetry (since boot)")
        req = metrics.get("requests", {})
        r1, r2, r3 = st.columns(3)
        r1.metric("Requests served", req.get("request_count", 0))
        mean_l = req.get("mean_latency_ms")
        p95_l = req.get("p95_latency_ms")
        r2.metric("Mean latency", f"{mean_l:.1f} ms" if mean_l is not None else "n/a")
        r3.metric("p95 latency", f"{p95_l:.1f} ms" if p95_l is not None else "n/a")
    elif ok_h:
        st.warning(f"Metrics unavailable: {metrics}")


# ===========================================================================
# TAB 2 — INSIGHTS (computed live from metadata_split.csv)
# ===========================================================================
with tab_insights:
    st.subheader("Dataset insights")
    st.caption("Three feature interpretations computed live from `data/metadata_split.csv`.")

    try:
        df = load_metadata()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not load metadata: {exc}")
        df = None

    if df is not None:
        n = len(df)
        mal_rate = df["label"].mean()

        # --- Insight 1: malignancy rate by anatomical site ------------------
        st.markdown("### 1 · Malignancy rate by anatomical site")
        site = (
            df.groupby("localization")["label"]
            .agg(["mean", "count"])
            .rename(columns={"mean": "malignancy_rate", "count": "n"})
            .sort_values("malignancy_rate", ascending=False)
        )
        st.bar_chart(site["malignancy_rate"])
        top_site = site.index[0]
        low_site = site.index[-1]
        st.markdown(
            f"**Interpretation.** Anatomical site is strongly associated with malignancy. "
            f"Lesions on the **{top_site}** have the highest malignant fraction "
            f"({site.loc[top_site, 'malignancy_rate']:.0%}, n={int(site.loc[top_site, 'n'])}), "
            f"while the **{low_site}** is among the lowest "
            f"({site.loc[low_site, 'malignancy_rate']:.0%}). "
            f"Sun-exposed regions (face, ear, neck, scalp) skew malignant, consistent with "
            f"UV-driven carcinomas — so localization carries real predictive signal."
        )

        # --- Insight 2: malignancy rate by age band -------------------------
        st.markdown("### 2 · Malignancy rate by age band")
        age = (
            df.dropna(subset=["age_band"])
            .groupby("age_band")["label"]
            .agg(["mean", "count"])
            .rename(columns={"mean": "malignancy_rate", "count": "n"})
        )
        age = age.reindex([b for b in AGE_BAND_ORDER if b in age.index])
        st.bar_chart(age["malignancy_rate"])
        if len(age) >= 2:
            youngest = age.index[0]
            oldest = age.index[-1]
            st.markdown(
                f"**Interpretation.** Malignancy risk rises monotonically with age. "
                f"The **{youngest}** band sits at {age.loc[youngest, 'malignancy_rate']:.0%} malignant, "
                f"climbing to {age.loc[oldest, 'malignancy_rate']:.0%} in the **{oldest}** band — "
                f"roughly a {age.loc[oldest, 'malignancy_rate'] / max(age.loc[youngest, 'malignancy_rate'], 1e-6):.1f}× increase. "
                f"Age is a clinically expected risk factor and a useful covariate for triage."
            )

        # --- Insight 3: class distribution ----------------------------------
        st.markdown("### 3 · Class distribution")
        dist = df["label"].map({0: "benign", 1: "malignant"}).value_counts()
        st.bar_chart(dist)
        st.markdown(
            f"**Interpretation.** The dataset is **imbalanced**: of {n:,} lesions, "
            f"only **{mal_rate:.1%} are malignant**. This is why the model is trained with "
            f"class weights and evaluated at an operating threshold of ~0.39 (not 0.5), and why "
            f"**PR-AUC and recall matter more than raw accuracy** — a naive 'always benign' "
            f"classifier would score {1 - mal_rate:.0%} accuracy while catching zero cancers."
        )

    # --- Evaluation figures from the notebook -------------------------------
    st.markdown("---")
    st.markdown("### Evaluation figures")
    st.caption("Pre-computed EDA and evaluation plots from the training notebook.")
    fig_titles = {
        "fig1_class_distribution.png": "Class distribution",
        "fig2_site.png": "Malignancy by anatomical site",
        "fig3_age.png": "Malignancy by age",
        "fig4_image_features.png": "Image feature analysis",
        "fig5_samples.png": "Sample lesions",
        "fig6_training_history.png": "Training history",
        "fig7_evaluation.png": "Evaluation (ROC / PR / confusion)",
        "fig8_threshold.png": "Threshold selection",
    }
    figs = sorted(FIGURES_DIR.glob("*.png")) if FIGURES_DIR.is_dir() else []
    if not figs:
        st.info(f"No figures found in {FIGURES_DIR}")
    else:
        cols = st.columns(2)
        for i, fig in enumerate(figs):
            with cols[i % 2]:
                st.image(str(fig), caption=fig_titles.get(fig.name, fig.stem), use_container_width=True)


# ===========================================================================
# TAB 3 — PREDICT
# ===========================================================================
with tab_predict:
    st.subheader("Classify a single lesion image")
    uploaded = st.file_uploader(
        "Upload a dermatoscopic image (jpg / png)",
        type=["jpg", "jpeg", "png"],
        key="predict_uploader",
    )

    col_img, col_res = st.columns([1, 1])
    if uploaded is not None:
        with col_img:
            st.image(uploaded, caption=uploaded.name, use_container_width=True)

        with col_res:
            if st.button("Predict", type="primary", key="predict_btn"):
                with st.spinner("Scoring image..."):
                    files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type or "image/jpeg")}
                    ok, res = api_post("/predict", files=files)
                if not ok:
                    st.error(f"Prediction failed: {res}")
                else:
                    label = res["label"]
                    prob = res["probability_malignant"]
                    if label == "malignant":
                        st.error(f"### 🔴 {label.upper()}")
                    else:
                        st.success(f"### 🟢 {label.upper()}")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("P(malignant)", f"{prob:.3f}")
                    m2.metric("Threshold", f"{res['threshold']:.2f}")
                    m3.metric("Latency", f"{res['latency_ms']:.0f} ms")
                    st.progress(min(1.0, float(prob)))
                    st.caption(
                        "Decision: malignant when P(malignant) ≥ threshold. "
                        "This is a decision-support demo, not a medical device."
                    )
    else:
        st.info("Upload an image to get a prediction.")


# ===========================================================================
# TAB 4 — RETRAIN
# ===========================================================================
with tab_retrain:
    st.subheader("Retrain on new labelled images")
    st.caption(
        "Upload labelled images (name them `0_*`/`1_*` or place in `benign/`/`malignant/` "
        "folders; a `.zip` is accepted). The custom model is fine-tuned and only promoted "
        "if test ROC-AUC does not regress by more than 0.005."
    )

    files = st.file_uploader(
        "Upload images or a .zip",
        type=["jpg", "jpeg", "png", "zip"],
        accept_multiple_files=True,
        key="retrain_uploader",
    )

    # --- Step 1: upload batch ----------------------------------------------
    if files and st.button("1 · Upload batch", key="upload_batch_btn"):
        multipart = [("files", (f.name, f.getvalue(), f.type or "application/octet-stream")) for f in files]
        with st.spinner("Uploading..."):
            ok, res = api_post("/upload", files=multipart, timeout=120)
        if not ok:
            st.error(f"Upload failed: {res}")
        else:
            st.session_state["batch_id"] = res["batch_id"]
            st.session_state["batch_counts"] = res["counts"]
            st.session_state.pop("job_id", None)

    if "batch_id" in st.session_state:
        counts = st.session_state.get("batch_counts", {})
        st.success(f"Batch `{st.session_state['batch_id']}` ready.")
        c1, c2, c3 = st.columns(3)
        c1.metric("Benign", counts.get("benign", 0))
        c2.metric("Malignant", counts.get("malignant", 0))
        c3.metric("Unlabeled (ignored)", counts.get("unlabeled", 0))
        if counts.get("benign", 0) == 0 or counts.get("malignant", 0) == 0:
            st.warning("Retraining needs BOTH classes present. Add the missing class before starting.")

        # --- Step 2: start retraining --------------------------------------
        cole1, cole2 = st.columns(2)
        epochs = cole1.number_input("Epochs", 1, 10, 3, key="retrain_epochs")
        subsample = cole2.number_input("Max training images", 100, 10000, 1500, step=100, key="retrain_subsample")

        if st.button("2 · Start retraining", type="primary", key="start_retrain_btn"):
            ok, res = api_post(
                "/retrain",
                json={"batch_id": st.session_state["batch_id"],
                      "epochs": int(epochs), "subsample": int(subsample)},
            )
            if not ok:
                st.error(f"Could not start retraining: {res}")
            else:
                st.session_state["job_id"] = res["job_id"]
                st.info(f"Job `{res['job_id']}` queued. Polling status...")

    # --- Step 3: poll status -----------------------------------------------
    if "job_id" in st.session_state:
        job_id = st.session_state["job_id"]
        st.markdown(f"#### Retrain job `{job_id}`")
        progress_bar = st.progress(0)
        status_line = st.empty()
        detail_box = st.empty()

        terminal = {"done", "failed"}
        job = None
        for _ in range(600):  # ~20 min ceiling at 2s cadence
            ok, job = api_get(f"/status/{job_id}")
            if not ok:
                status_line.error(f"Status check failed: {job}")
                break
            pct = int(job.get("progress", 0) or 0)
            progress_bar.progress(min(100, pct) / 100.0)
            status_line.write(f"**{job.get('status', '?').upper()}** — stage: `{job.get('stage', '...')}` ({pct}%)")
            if job.get("status") in terminal:
                break
            time.sleep(2)

        if job and job.get("status") == "done":
            report = job.get("report", {})
            if report.get("promoted"):
                st.success(f"✅ New model PROMOTED. {report.get('message', '')}")
            else:
                st.warning(f"⛔ New model rejected. {report.get('message', '')}")
            b1, b2, b3 = st.columns(3)
            b1.metric("Base ROC-AUC", f"{report.get('base_roc_auc', float('nan')):.3f}")
            b2.metric("New ROC-AUC", f"{report.get('new_roc_auc', float('nan')):.3f}",
                      delta=f"{-report.get('regression', 0):+.3f}")
            b3.metric("Images trained", report.get("n_train", "?"))
            st.caption(f"Evaluation source: `{report.get('eval_source', '?')}` · "
                       f"scored on {report.get('n_eval', '?')} held-out rows · "
                       f"max allowed regression: {report.get('max_allowed_regression', '?')}")
            if report.get("eval_warning"):
                st.info(f"ℹ️ {report['eval_warning']}")
            with st.expander("Full retrain report"):
                st.json(report)
        elif job and job.get("status") == "failed":
            st.error(f"❌ Retrain failed: {job.get('error', 'unknown error')}")

        if st.button("Clear job", key="clear_job_btn"):
            st.session_state.pop("job_id", None)
            st.rerun()
