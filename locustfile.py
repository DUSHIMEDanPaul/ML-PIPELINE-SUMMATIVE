"""
Locust load test for the HAM10000 classifier API.

Run against nginx (the load balancer), e.g.:

    locust -f locustfile.py --host http://localhost:8000

then open http://localhost:8089, or headless:

    locust -f locustfile.py --host http://localhost:8000 \
           --users 50 --spawn-rate 10 --run-time 2m --headless

IMPORTANT: the request body is a PRE-RESIZED 128x128 JPEG (~tens of KB), not a
full-size dermatoscopy photo. Sending a multi-MB image would make this measure
network bandwidth and JPEG decode time instead of the model's inference latency.
Point LOCUST_IMAGE at a real 128x128 lesion JPEG to use it; otherwise a synthetic
128x128 JPEG is generated once at startup.
"""

from __future__ import annotations

import io
import os

from locust import HttpUser, between, task

IMG_SIZE = 128


def _build_payload() -> bytes:
    """Return the JPEG bytes posted to /predict (built once, reused by all users)."""
    path = os.environ.get("LOCUST_IMAGE")
    if path and os.path.isfile(path):
        with open(path, "rb") as fh:
            return fh.read()

    # Synthesize a 128x128 RGB JPEG. Structured noise compresses to a realistic
    # ~10-40 KB, close to a real resized lesion tile.
    from PIL import Image
    import numpy as np

    rng = np.random.default_rng(42)
    base = rng.integers(60, 200, size=(IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    # add a soft blob so it isn't pure noise (and compresses a touch better)
    yy, xx = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE]
    blob = (80 * np.exp(-((xx - 64) ** 2 + (yy - 64) ** 2) / (2 * 30.0 ** 2))).astype(np.uint8)
    base[..., 0] = np.clip(base[..., 0].astype(int) + blob, 0, 255).astype(np.uint8)

    buf = io.BytesIO()
    Image.fromarray(base, "RGB").save(buf, format="JPEG", quality=90)
    return buf.getvalue()


# Built once at import time; shared read-only by every simulated user.
IMAGE_BYTES = _build_payload()
IMAGE_KB = len(IMAGE_BYTES) / 1024.0
print(f"[locustfile] /predict payload = {IMAGE_KB:.1f} KB "
      f"({'LOCUST_IMAGE' if os.environ.get('LOCUST_IMAGE') else 'synthetic'} 128x128 JPEG)")


class PredictUser(HttpUser):
    """Simulates a client hammering /predict, with occasional /health checks."""

    # Think-time between requests keeps this a throughput test, not a tight loop.
    wait_time = between(0.1, 0.5)

    @task(9)
    def predict(self):
        files = {"file": ("lesion_128.jpg", IMAGE_BYTES, "image/jpeg")}
        with self.client.post("/predict", files=files, name="/predict", catch_response=True) as resp:
            if resp.status_code != 200:
                resp.failure(f"HTTP {resp.status_code}: {resp.text[:120]}")
                return
            try:
                body = resp.json()
            except ValueError:
                resp.failure("response was not JSON")
                return
            if body.get("label") not in ("benign", "malignant"):
                resp.failure(f"unexpected body: {body}")
            else:
                resp.success()

    @task(1)
    def health(self):
        with self.client.get("/health", name="/health", catch_response=True) as resp:
            if resp.status_code != 200:
                resp.failure(f"HTTP {resp.status_code}")
            else:
                resp.success()
