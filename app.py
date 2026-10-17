"""NeuroScreen: EEG depression screening dashboard.

Run with:   python app.py        (then open http://localhost:8000)

Everything lives in this one folder:
  app.py                     this server (Python / FastAPI)
  index.html style.css script.js    the web dashboard (plain HTML, CSS, JavaScript)
  common.py train.py gen_data.py    EEG processing, model training, synthetic data
  dataset/                   training recordings + labels.csv
  models/                    trained model and its metrics
  sample_recordings/         files to upload in the dashboard

API:
  GET  /api/health          liveness check
  GET  /api/metrics         model quality + reference band profiles
  GET  /api/demo/{kind}     demo recording ("healthy" or "mdd")
  POST /api/predict         analyse an uploaded .npy / .csv recording
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
import webbrowser
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from starlette.middleware.base import BaseHTTPMiddleware

from common import (BANDS, CHANNELS, FS, band_power, extract_features,
                    load_recording, make_epochs, preprocess)

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "models" / "model.joblib"
METRICS_PATH = BASE / "models" / "metrics.json"
DEMO_FILES = {
    "healthy": BASE / "sample_recordings" / "healthy_sample_1.npy",
    "mdd": BASE / "sample_recordings" / "mdd_sample_1.npy",
}
MAX_UPLOAD_MB = 200


def train_model():
    """(Re)build the model from dataset/. Generates synthetic data first if there is none."""
    if not (BASE / "dataset" / "labels.csv").exists():
        print("No dataset found. Generating synthetic demo data...")
        subprocess.run([sys.executable, str(BASE / "gen_data.py")], cwd=BASE, check=True)
    print("Training the model from dataset/ (about a minute)...")
    subprocess.run([sys.executable, str(BASE / "train.py")], cwd=BASE, check=True)


def load_model():
    """Load the saved model; if it is missing or was saved by an incompatible
    scikit-learn version, train a fresh one from the dataset."""
    if MODEL_PATH.exists() and METRICS_PATH.exists():
        try:
            return joblib.load(MODEL_PATH), json.loads(METRICS_PATH.read_text())
        except Exception as e:  # noqa: BLE001
            print(f"Saved model could not be loaded ({e}). Retraining...")
    train_model()
    return joblib.load(MODEL_PATH), json.loads(METRICS_PATH.read_text())


model, metrics = load_model()
app = FastAPI(title="NeuroScreen EEG")


class NoCache(BaseHTTPMiddleware):
    """Always serve the latest page so an old cached copy never shows."""
    async def dispatch(self, request, call_next):
        resp = await call_next(request)
        resp.headers["Cache-Control"] = "no-store, max-age=0"
        return resp


app.add_middleware(NoCache)


# ---- the web page: only these three files are served --------------------------
def _page(name):
    def handler():
        return FileResponse(BASE / name)
    return handler


for route, filename in {"/": "index.html", "/style.css": "style.css", "/script.js": "script.js"}.items():
    app.add_api_route(route, _page(filename), methods=["GET"], include_in_schema=False)


# ---- API -----------------------------------------------------------------------
@app.get("/api/health")
def health():
    return {"status": "ok", "demo": bool(metrics.get("demo"))}


@app.get("/api/metrics")
def get_metrics():
    return metrics


@app.get("/api/demo/{kind}")
def demo(kind: str):
    path = DEMO_FILES.get(kind)
    if path is None or not path.exists():
        raise HTTPException(404, "Demo recording not found.")
    return FileResponse(path, media_type="application/octet-stream", filename=path.name)


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in (".npy", ".csv"):
        raise HTTPException(400, "Unsupported file type. Upload a .npy (channels x samples) or .csv (samples x channels) file.")
    data = await file.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File is larger than {MAX_UPLOAD_MB} MB.")
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(data)
    try:
        raw = load_recording(tmp.name)
    except Exception as e:  # noqa: BLE001 - surface any parse problem to the user
        raise HTTPException(400, str(e)) from e
    finally:
        os.unlink(tmp.name)

    if not np.isfinite(raw).all():
        raise HTTPException(400, "The recording contains missing or invalid values (NaN or infinity).")
    if raw.shape[1] < 8 * FS:
        raise HTTPException(
            400, f"Recording is too short ({raw.shape[1] / FS:.1f} s). At least 8 seconds at {FS} Hz are needed."
        )

    x = preprocess(raw)
    epochs = make_epochs(x)
    if len(epochs) < 3:
        raise HTTPException(400, "Recording is too short. At least 8 seconds are needed.")
    probs = model.predict_proba(extract_features(epochs))[:, 1]
    p = float(probs.mean())

    bp = band_power(epochs).mean(0)  # (channels, bands)
    rel = bp / bp.sum(1, keepdims=True)
    return {
        "filename": file.filename,
        "duration_sec": round(raw.shape[1] / FS, 1),
        "n_epochs": int(len(epochs)),
        "probability_mdd": round(p, 4),
        "label": "MDD-like pattern" if p >= 0.5 else "Healthy-like pattern",
        "confidence": round(max(p, 1 - p), 4),
        "epoch_probs": [round(float(v), 3) for v in probs],
        "epoch_agreement": round(float(np.mean((probs >= 0.5) == (p >= 0.5))), 3),
        "bands": list(BANDS),
        "band_mean": rel.mean(0).round(4).tolist(),
        "channels": CHANNELS,
        "band_by_channel": rel.round(4).tolist(),
        "trace": {"fs": FS, "channels": CHANNELS[:4],
                  "data": x[:4, : 5 * FS : 2].round(2).tolist()},
        "demo": bool(metrics.get("demo")),
    }


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="Start the NeuroScreen dashboard")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true", help="do not open the browser automatically")
    args = parser.parse_args()
    url = f"http://localhost:{args.port}"
    print(f"\nNeuroScreen is running at {url}  (press Ctrl+C to stop)\n")
    if not args.no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
