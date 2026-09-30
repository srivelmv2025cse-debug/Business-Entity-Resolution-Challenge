"""
app.py – FastAPI backend for Business Entity Resolution.

Endpoints:
  POST /api/model/train    – train the ML model
  POST /api/model/predict  – run prediction on candidate_pairs.tsv
  GET  /api/model/metrics  – return last evaluation metrics
  GET  /api/model/status   – check model/threshold file status

Member 2 owns all /api/model/* routes.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ---- Ensure project root on path ----------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s – %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App instance
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Business Entity Resolution API",
    version="1.0.0",
    description="ML-powered entity matching with F0.5 evaluation",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Persistent state (in-memory; reset on restart)
# ---------------------------------------------------------------------------
_last_metrics: Dict[str, Any] = {}
_training_status: str = "idle"   # idle | running | done | error
_training_error: str  = ""


# ---------------------------------------------------------------------------
# Helper paths
# ---------------------------------------------------------------------------
MODEL_PATH  = PROJECT_ROOT / "models" / "entity_match_model.pkl"
THRESH_PATH = PROJECT_ROOT / "models" / "threshold.json"


# ---------------------------------------------------------------------------
# Background training task
# ---------------------------------------------------------------------------

def _run_training() -> None:
    global _last_metrics, _training_status, _training_error
    _training_status = "running"
    try:
        from backend.models.trainer import train
        metrics = train()
        _last_metrics    = metrics
        _training_status = "done"
        logger.info("Training complete. F0.5=%.4f", metrics.get("f05", 0))
    except Exception as exc:
        _training_status = "error"
        _training_error  = str(exc)
        logger.exception("Training failed: %s", exc)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/api/model/train", summary="Train the entity-matching ML model")
async def train_model(background_tasks: BackgroundTasks):
    """
    Trigger model training in the background.

    - Loads training data (or synthetic fallback)
    - Builds positive + hard-negative examples
    - Trains LightGBM with F0.5-optimised threshold
    - Saves model and threshold to models/
    """
    global _training_status, _training_error
    if _training_status == "running":
        return JSONResponse(
            status_code=202,
            content={"message": "Training already in progress", "status": "running"},
        )
    _training_error = ""
    background_tasks.add_task(_run_training)
    return {"message": "Training started", "status": "running"}


@app.post("/api/model/predict", summary="Generate matching_results.tsv")
async def predict_matches():
    """
    Run prediction on candidate_pairs.tsv produced by Member 1.

    Returns summary of matched entities.
    """
    if not MODEL_PATH.exists():
        raise HTTPException(
            status_code=400,
            detail="Model not trained yet. POST /api/model/train first.",
        )
    try:
        from backend.models.predictor import predict
        summary = predict()
        return summary
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/model/metrics", summary="Retrieve latest evaluation metrics")
async def get_metrics():
    """
    Return the last computed F0.5, precision, recall, threshold etc.

    Returns 404 if no training has been completed yet.
    """
    if not _last_metrics:
        # Try to read from threshold file
        if THRESH_PATH.exists():
            with open(THRESH_PATH) as f:
                thr_data = json.load(f)
            return {
                "f05":         thr_data.get("val_f05",       None),
                "precision":   thr_data.get("val_precision", None),
                "recall":      thr_data.get("val_recall",    None),
                "threshold":   thr_data.get("threshold",     None),
                "match_count":     None,
                "singleton_count": None,
                "source": "threshold_file",
            }
        raise HTTPException(
            status_code=404,
            detail="No metrics available. Train the model first.",
        )
    return {
        "f05":             _last_metrics.get("f05"),
        "precision":       _last_metrics.get("precision"),
        "recall":          _last_metrics.get("recall"),
        "threshold":       _last_metrics.get("threshold"),
        "match_count":     _last_metrics.get("match_count"),
        "singleton_count": _last_metrics.get("singleton_count"),
        "false_merges":    _last_metrics.get("false_merges"),
        "n_train_pairs":   _last_metrics.get("n_train_pairs"),
        "source": "in_memory",
    }


@app.get("/api/model/status", summary="Check model and threshold file status")
async def model_status():
    """Return the current training status and file availability."""
    model_exists  = MODEL_PATH.exists()
    thresh_exists = THRESH_PATH.exists()

    threshold = None
    val_f05   = None
    if thresh_exists:
        with open(THRESH_PATH) as f:
            thr_data = json.load(f)
        threshold = thr_data.get("threshold")
        val_f05   = thr_data.get("val_f05")

    return {
        "training_status": _training_status,
        "training_error":  _training_error or None,
        "model_exists":    model_exists,
        "threshold_exists": thresh_exists,
        "threshold":       threshold,
        "val_f05":         val_f05,
        "model_path":      str(MODEL_PATH),
        "threshold_path":  str(THRESH_PATH),
    }


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "service": "entity-resolution-api"}


# ---------------------------------------------------------------------------
# Run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
