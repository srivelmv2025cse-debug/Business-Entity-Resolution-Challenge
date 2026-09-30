"""
app.py – Full FastAPI backend for Business Entity Resolution.

Routes (Member 2 owns /api/model/*):
  POST /api/model/train
  POST /api/model/predict
  GET  /api/model/metrics
  GET  /api/model/status

Routes (Member 3 adds):
  GET  /api/data/stats
  POST /api/data/upload
  POST /api/pipeline/preprocess
  POST /api/pipeline/block
  GET  /api/pipeline/status
  GET  /api/candidates/stats
  GET  /api/candidates/search
  GET  /api/entity/lookup
  GET  /api/evaluation/threshold-curve
  GET  /api/output/preview
  POST /api/output/download-request
  POST /api/validate
  GET  /api/validate/results
  POST /api/submission/package
  GET  /health
"""

from __future__ import annotations

import csv
import io
import json
import logging
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, BackgroundTasks, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

# ---- Ensure project root on path ----------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App instance
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Business Entity Resolution API",
    version="2.0.0",
    description="ML-powered entity matching with F0.5 evaluation",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------
MODEL_DIR      = PROJECT_ROOT / "models"
MODEL_PATH     = MODEL_DIR / "entity_match_model.pkl"
THRESH_PATH    = MODEL_DIR / "threshold.json"
OUTPUT_DIR     = PROJECT_ROOT / "output"
DATASET_DIR    = PROJECT_ROOT / "dataset"
TRAIN_DIR      = DATASET_DIR / "train"
TEST_DIR       = DATASET_DIR / "test"
CAND_PATH      = OUTPUT_DIR / "candidate_pairs.tsv"
RESULTS_PATH   = OUTPUT_DIR / "matching_results.tsv"
FRONTEND_DIR   = PROJECT_ROOT / "frontend"

# ---------------------------------------------------------------------------
# Serve frontend static files
# ---------------------------------------------------------------------------
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR / "static"), html=False), name="static")

# ---------------------------------------------------------------------------
# Global state (in-memory; reset on restart)
# ---------------------------------------------------------------------------
_last_metrics: Dict[str, Any] = {}
_training_status: str = "idle"   # idle | running | done | error
_training_error: str  = ""
_pipeline_status: Dict[str, Any] = {
    "preprocess": "idle",
    "blocking":   "idle",
    "last_run":   None,
}
_validation_results: Dict[str, Any] = {}


# ===========================================================================
# HEALTH
# ===========================================================================

@app.get("/health")
async def health():
    return {"status": "ok", "service": "entity-resolution-api", "version": "2.0.0"}


# ===========================================================================
# DATA STATS & UPLOAD
# ===========================================================================

@app.get("/api/data/stats")
async def data_stats():
    """Return row/column counts for all available source files."""
    import pandas as pd

    def file_stats(path: Path) -> Dict:
        if not path.exists():
            return {"exists": False, "rows": 0, "cols": 0, "size_bytes": 0}
        try:
            df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
            return {
                "exists": True,
                "rows": len(df),
                "cols": len(df.columns),
                "size_bytes": path.stat().st_size,
                "columns": list(df.columns),
            }
        except Exception as e:
            return {"exists": True, "error": str(e), "rows": 0, "cols": 0}

    def out_stats(path: Path) -> Dict:
        if not path.exists():
            return {"exists": False, "rows": 0}
        try:
            df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
            return {"exists": True, "rows": len(df), "size_bytes": path.stat().st_size}
        except Exception:
            return {"exists": True, "rows": 0}

    train = {
        "source1": file_stats(TRAIN_DIR / "train_source1.tsv"),
        "source2": file_stats(TRAIN_DIR / "train_source2.tsv"),
        "source3": file_stats(TRAIN_DIR / "train_source3.tsv"),
        "ground_truth": file_stats(TRAIN_DIR / "train_ground_truth.tsv"),
    }
    test = {
        "source1": file_stats(TEST_DIR / "test_source1.tsv"),
        "source2": file_stats(TEST_DIR / "test_source2.tsv"),
        "source3": file_stats(TEST_DIR / "test_source3.tsv"),
    }
    output = {
        "candidate_pairs":   out_stats(CAND_PATH),
        "matching_results":  out_stats(RESULTS_PATH),
    }
    model = {
        "model_exists":     MODEL_PATH.exists(),
        "threshold_exists": THRESH_PATH.exists(),
        "threshold":        None,
        "val_f05":          None,
    }
    if THRESH_PATH.exists():
        try:
            with open(THRESH_PATH) as f:
                td = json.load(f)
            model["threshold"] = td.get("threshold")
            model["val_f05"]   = td.get("val_f05")
        except Exception:
            pass

    return {"train": train, "test": test, "output": output, "model": model}


@app.post("/api/data/upload")
async def upload_file(
    file: UploadFile = File(...),
    target: str = Query(..., description="train_source1 | train_source2 | train_source3 | train_ground_truth | test_source1 | test_source2 | test_source3"),
):
    """Upload a TSV file to the correct dataset directory."""
    import pandas as pd

    VALID_TARGETS = {
        "train_source1":      TRAIN_DIR / "train_source1.tsv",
        "train_source2":      TRAIN_DIR / "train_source2.tsv",
        "train_source3":      TRAIN_DIR / "train_source3.tsv",
        "train_ground_truth": TRAIN_DIR / "train_ground_truth.tsv",
        "test_source1":       TEST_DIR  / "test_source1.tsv",
        "test_source2":       TEST_DIR  / "test_source2.tsv",
        "test_source3":       TEST_DIR  / "test_source3.tsv",
    }
    if target not in VALID_TARGETS:
        raise HTTPException(status_code=400, detail=f"Invalid target. Choose from: {list(VALID_TARGETS)}")

    contents = await file.read()
    # Quick TSV validation
    try:
        text = contents.decode("utf-8", errors="replace")
        df = pd.read_csv(io.StringIO(text), sep="\t", dtype=str, keep_default_na=False)
        rows, cols = len(df), len(df.columns)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid TSV file: {e}")

    out_path = VALID_TARGETS[target]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(contents)

    return {
        "message": f"File uploaded to {out_path.name}",
        "rows": rows,
        "cols": cols,
        "size_bytes": len(contents),
        "columns": list(df.columns),
    }


# ===========================================================================
# PIPELINE – PREPROCESSING + BLOCKING
# ===========================================================================

def _run_preprocess_and_block():
    global _pipeline_status
    import time
    _pipeline_status["preprocess"] = "running"
    _pipeline_status["blocking"]   = "pending"
    _pipeline_status["last_run"]   = None
    try:
        from backend.blocking.candidate_generator import run_pipeline
        run_pipeline(use_train=False)
        _pipeline_status["preprocess"] = "done"
        _pipeline_status["blocking"]   = "done"
        _pipeline_status["last_run"]   = time.strftime("%H:%M:%S")
        logger.info("Preprocessing + blocking complete")
    except Exception as exc:
        _pipeline_status["preprocess"] = "error"
        _pipeline_status["blocking"]   = "error"
        _pipeline_status["error"]      = str(exc)
        logger.exception("Pipeline failed: %s", exc)


@app.post("/api/pipeline/preprocess")
async def run_preprocessing(background_tasks: BackgroundTasks):
    """Trigger preprocessing and candidate blocking in the background."""
    if _pipeline_status.get("preprocess") == "running":
        return {"message": "Pipeline already running", "status": _pipeline_status}
    background_tasks.add_task(_run_preprocess_and_block)
    return {"message": "Preprocessing + blocking started", "status": "running"}


@app.get("/api/pipeline/status")
async def pipeline_status():
    return _pipeline_status


# ===========================================================================
# CANDIDATE STATS & SEARCH
# ===========================================================================

@app.get("/api/candidates/stats")
async def candidate_stats():
    """Statistics about the generated candidate_pairs.tsv."""
    import pandas as pd

    if not CAND_PATH.exists():
        return {"exists": False, "total_s1": 0, "total_candidates": 0, "avg_per_s1": 0}

    df = pd.read_csv(CAND_PATH, sep="\t", dtype=str, keep_default_na=False)
    total_s1 = len(df)
    def count_cands(x):
        if not x or str(x).lower() in ("nan", ""):
            return 0
        return len([c for c in str(x).split(",") if c.strip()])

    df["_cnt"] = df["candidate_entity_ids"].apply(count_cands)
    total_cands = int(df["_cnt"].sum())
    avg_per_s1  = round(total_cands / total_s1, 2) if total_s1 else 0

    # Blocking strategy labels (always shown since they're fixed)
    strategies = [
        {"name": "Exact normalized name",          "status": "active"},
        {"name": "Token-based blocking",            "status": "active"},
        {"name": "Country-aware blocking",          "status": "active"},
        {"name": "Address token blocking",          "status": "active"},
        {"name": "Postal-code-like token blocking", "status": "active"},
        {"name": "TF-IDF retrieval",                "status": "active"},
    ]

    # Rough reduction ratio estimate
    s2_rows = 0
    s3_rows = 0
    for p in [TEST_DIR/"test_source2.tsv", TRAIN_DIR/"train_source2.tsv"]:
        if p.exists():
            try:
                s2_rows = len(pd.read_csv(p, sep="\t"))
                break
            except Exception:
                pass
    for p in [TEST_DIR/"test_source3.tsv", TRAIN_DIR/"train_source3.tsv"]:
        if p.exists():
            try:
                s3_rows = len(pd.read_csv(p, sep="\t"))
                break
            except Exception:
                pass

    naive_pairs = total_s1 * (s2_rows + s3_rows) if (s2_rows + s3_rows) > 0 else 0
    reduction   = round(1.0 - (total_cands / naive_pairs), 4) if naive_pairs > 0 else None

    return {
        "exists":          True,
        "total_s1":        total_s1,
        "total_candidates":total_cands,
        "avg_per_s1":      avg_per_s1,
        "reduction_ratio": reduction,
        "strategies":      strategies,
    }


@app.get("/api/candidates/search")
async def search_candidates(s1_id: str = Query(..., description="Source 1 entity ID")):
    """Return candidate records for a given S1 entity ID."""
    import pandas as pd

    if not CAND_PATH.exists():
        raise HTTPException(status_code=404, detail="candidate_pairs.tsv not found")

    cp_df = pd.read_csv(CAND_PATH, sep="\t", dtype=str, keep_default_na=False)
    row = cp_df[cp_df["source1_entity_id"] == s1_id]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"No entry for S1 ID: {s1_id}")

    cand_str = str(row.iloc[0]["candidate_entity_ids"])
    cand_ids = [c.strip() for c in cand_str.split(",") if c.strip()] if cand_str and cand_str.lower() != "nan" else []

    # Load entity details
    s1_entity = None
    cand_details = []
    for src_dir, prefix in [(TEST_DIR, "test"), (TRAIN_DIR, "train")]:
        p1 = src_dir / f"{prefix}_source1.tsv"
        if p1.exists():
            try:
                df1 = pd.read_csv(p1, sep="\t", dtype=str, keep_default_na=False)
                match = df1[df1["entity_id"] == s1_id]
                if not match.empty:
                    s1_entity = match.iloc[0].to_dict()
                    break
            except Exception:
                pass

    for cid in cand_ids:
        for src_dir, prefix in [(TEST_DIR, "test"), (TRAIN_DIR, "train")]:
            for num in [2, 3]:
                p = src_dir / f"{prefix}_source{num}.tsv"
                if p.exists():
                    try:
                        df = pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False)
                        match = df[df["entity_id"] == cid]
                        if not match.empty:
                            cand_details.append({"source": f"source{num}", **match.iloc[0].to_dict()})
                            break
                    except Exception:
                        pass
            else:
                continue
            break

    return {
        "s1_id":      s1_id,
        "s1_entity":  s1_entity,
        "candidate_ids": cand_ids,
        "candidates": cand_details,
    }


# ===========================================================================
# ENTITY LOOKUP (for matching page)
# ===========================================================================

@app.get("/api/entity/lookup")
async def entity_lookup(s1_id: str = Query(...)):
    """Return full entity details + model predictions for a given S1 ID."""
    import pandas as pd

    if not MODEL_PATH.exists():
        raise HTTPException(status_code=400, detail="Model not trained yet")
    if not CAND_PATH.exists():
        raise HTTPException(status_code=404, detail="candidate_pairs.tsv not found")

    # Get candidates
    cp_df = pd.read_csv(CAND_PATH, sep="\t", dtype=str, keep_default_na=False)
    row   = cp_df[cp_df["source1_entity_id"] == s1_id]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"S1 ID not found: {s1_id}")

    cand_str  = str(row.iloc[0]["candidate_entity_ids"])
    cand_ids  = [c.strip() for c in cand_str.split(",") if c.strip()] if cand_str and cand_str.lower() != "nan" else []

    try:
        import joblib
        from backend.preprocessing.normalizer import normalize_dataframe
        from backend.models.features import fit_tfidf, build_feature_matrix, make_lookups, FEATURE_COLS, _tfidf_cache

        clf       = joblib.load(MODEL_PATH)
        with open(THRESH_PATH) as f:
            thr_data = json.load(f)
        threshold = float(thr_data["threshold"])

        # Load entity data
        from backend.models.predictor import _load_test_data
        df_s1, df_s2, df_s3 = _load_test_data()
        normalize_dataframe(df_s1)
        normalize_dataframe(df_s2)
        normalize_dataframe(df_s3)

        if _tfidf_cache._name_vec is None:
            fit_tfidf(df_s1, df_s2, df_s3)

        s1_lookup, cand_lookup = make_lookups(df_s1, df_s2, df_s3)
        s1_row = s1_lookup.get(s1_id, {})

        results = []
        if cand_ids:
            pairs_to_score = [(s1_id, cid) for cid in cand_ids]
            X      = build_feature_matrix(pairs_to_score, s1_lookup, cand_lookup)
            scores = clf.predict_proba(X[FEATURE_COLS])[:, 1]

            for cid, score in zip(cand_ids, scores):
                cand_row = cand_lookup.get(cid, {})
                from backend.models.features import compute_pair_features
                feats = compute_pair_features(s1_row, cand_row) if s1_row and cand_row else {}
                results.append({
                    "entity_id":       cid,
                    "business_name":   cand_row.get("business_name", ""),
                    "business_address":cand_row.get("business_address", ""),
                    "country":         cand_row.get("country", ""),
                    "probability":     round(float(score), 4),
                    "decision":        "MATCH" if score >= threshold else "NO MATCH",
                    "name_similarity": round(feats.get("rapidfuzz_ratio", 0), 3),
                    "address_similarity": round(feats.get("address_sim", 0), 3),
                    "country_match":   bool(feats.get("country_match", 0)),
                })

        return {
            "s1_id":    s1_id,
            "s1_entity": {
                "entity_id":        s1_id,
                "business_name":    s1_row.get("business_name", ""),
                "business_address": s1_row.get("business_address", ""),
                "country":          s1_row.get("country", ""),
            } if s1_row else None,
            "threshold":    threshold,
            "candidates":   results,
            "has_match":    any(r["decision"] == "MATCH" for r in results),
        }
    except Exception as exc:
        logger.exception("Entity lookup failed")
        raise HTTPException(status_code=500, detail=str(exc))


# ===========================================================================
# EVALUATION – THRESHOLD CURVE
# ===========================================================================

@app.get("/api/evaluation/threshold-curve")
async def threshold_curve():
    """Compute F0.5 / precision / recall at multiple thresholds (requires ground truth)."""
    import pandas as pd

    gt_path = TRAIN_DIR / "train_ground_truth.tsv"
    mr_path = RESULTS_PATH

    if not gt_path.exists():
        return {"available": False, "reason": "Ground truth not found"}
    if not MODEL_PATH.exists():
        return {"available": False, "reason": "Model not trained"}

    try:
        from backend.evaluation.f05 import (
            parse_ground_truth, macro_f05, find_best_threshold
        )
        from backend.preprocessing.normalizer import normalize_dataframe
        from backend.models.features import fit_tfidf, build_feature_matrix, make_lookups, FEATURE_COLS, _tfidf_cache
        from backend.data.loader import load_ground_truth, get_train_paths, load_all_sources
        import joblib

        # Load training data
        train_paths = get_train_paths()
        df_s1, df_s2, df_s3, df_gt = load_all_sources(
            train_paths["source1"], train_paths["source2"],
            train_paths["source3"], gt_path
        )
        normalize_dataframe(df_s1)
        normalize_dataframe(df_s2)
        normalize_dataframe(df_s3)

        if _tfidf_cache._name_vec is None:
            fit_tfidf(df_s1, df_s2, df_s3)

        s1_lookup, cand_lookup = make_lookups(df_s1, df_s2, df_s3)
        gt_dict = parse_ground_truth(df_gt)

        # Load candidate pairs
        if not CAND_PATH.exists():
            return {"available": False, "reason": "candidate_pairs.tsv not found"}
        cp_df = pd.read_csv(CAND_PATH, sep="\t", dtype=str, keep_default_na=False)

        # Score all candidate pairs
        clf = joblib.load(MODEL_PATH)
        pair_scores = {}
        for _, row in cp_df.iterrows():
            s1_id    = str(row["source1_entity_id"]).strip()
            cand_str = str(row.get("candidate_entity_ids", "")).strip()
            if not cand_str or cand_str.lower() in ("nan", ""):
                continue
            cids = [c.strip() for c in cand_str.split(",") if c.strip()]
            if not cids:
                continue
            pairs = [(s1_id, cid) for cid in cids]
            X     = build_feature_matrix(pairs, s1_lookup, cand_lookup)
            scores = clf.predict_proba(X[FEATURE_COLS])[:, 1]
            for cid, sc in zip(cids, scores):
                pair_scores[(s1_id, cid)] = float(sc)

        s1_ids = list(df_s1["entity_id"])
        thresholds = [round(t * 0.05, 2) for t in range(1, 21)]

        curve = []
        for thr in thresholds:
            preds = {s1_id: set() for s1_id in s1_ids}
            for (s1_id, cid), sc in pair_scores.items():
                if sc >= thr:
                    preds[s1_id].add(cid)
            m = macro_f05(gt_dict, preds)
            curve.append({
                "threshold": thr,
                "f05":       round(m["f05"], 4),
                "precision": round(m["precision"], 4),
                "recall":    round(m["recall"], 4),
            })

        best_thr, _ = find_best_threshold(s1_ids, pair_scores, gt_dict, thresholds)

        return {
            "available": True,
            "curve":     curve,
            "best_threshold": best_thr,
        }
    except Exception as exc:
        logger.exception("Threshold curve failed")
        return {"available": False, "reason": str(exc)}


# ===========================================================================
# MODEL TRAINING + PREDICTION (Member 2 routes kept intact)
# ===========================================================================

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


@app.post("/api/model/train", summary="Train the entity-matching ML model")
async def train_model(background_tasks: BackgroundTasks):
    global _training_status, _training_error
    if _training_status == "running":
        return JSONResponse(status_code=202, content={"message": "Training already in progress", "status": "running"})
    _training_error = ""
    background_tasks.add_task(_run_training)
    return {"message": "Training started", "status": "running"}


@app.post("/api/model/predict", summary="Generate matching_results.tsv")
async def predict_matches():
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=400, detail="Model not trained yet. POST /api/model/train first.")
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
    if not _last_metrics:
        if THRESH_PATH.exists():
            with open(THRESH_PATH) as f:
                thr_data = json.load(f)
            return {
                "f05":         thr_data.get("val_f05"),
                "precision":   thr_data.get("val_precision"),
                "recall":      thr_data.get("val_recall"),
                "threshold":   thr_data.get("threshold"),
                "match_count":     None,
                "singleton_count": None,
                "source": "threshold_file",
            }
        raise HTTPException(status_code=404, detail="No metrics available. Train the model first.")
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
        "training_status":  _training_status,
        "training_error":   _training_error or None,
        "model_exists":     model_exists,
        "threshold_exists": thresh_exists,
        "threshold":        threshold,
        "val_f05":          val_f05,
        "model_path":       str(MODEL_PATH),
        "threshold_path":   str(THRESH_PATH),
    }


# ===========================================================================
# OUTPUT PREVIEW & DOWNLOAD
# ===========================================================================

@app.get("/api/output/preview")
async def output_preview(file: str = Query(..., description="candidate_pairs | matching_results")):
    """Return first 100 rows of an output file as JSON."""
    import pandas as pd

    paths = {
        "candidate_pairs":  CAND_PATH,
        "matching_results": RESULTS_PATH,
    }
    if file not in paths:
        raise HTTPException(status_code=400, detail="file must be candidate_pairs or matching_results")
    p = paths[file]
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"{file} not found")
    try:
        df = pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False)
        total_rows = len(df)
        preview    = df.head(100).to_dict(orient="records")
        return {
            "file":       p.name,
            "total_rows": total_rows,
            "preview":    preview,
            "columns":    list(df.columns),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/output/download")
async def download_output(file: str = Query(..., description="candidate_pairs | matching_results")):
    """Download an output TSV file."""
    paths = {
        "candidate_pairs":  CAND_PATH,
        "matching_results": RESULTS_PATH,
    }
    if file not in paths:
        raise HTTPException(status_code=400, detail="Invalid file name")
    p = paths[file]
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"{file} not found")
    return FileResponse(str(p), media_type="text/tab-separated-values", filename=p.name)


# ===========================================================================
# VALIDATION
# ===========================================================================

@app.post("/api/validate")
async def validate_submission():
    """Run all submission validation checks."""
    import pandas as pd

    checks = []

    def add(name: str, passed: bool, detail: str = ""):
        checks.append({"name": name, "passed": passed, "detail": detail})

    # Check 1: candidate_pairs.tsv exists
    cp_exists = CAND_PATH.exists()
    add("candidate_pairs.tsv exists", cp_exists)
    if not cp_exists:
        _validation_results["checks"] = checks
        _validation_results["all_passed"] = False
        return {"all_passed": False, "checks": checks}

    # Check 2: matching_results.tsv exists
    mr_exists = RESULTS_PATH.exists()
    add("matching_results.tsv exists", mr_exists)

    # Load candidate_pairs
    try:
        cp_df = pd.read_csv(CAND_PATH, sep="\t", dtype=str, keep_default_na=False)
    except Exception as e:
        add("candidate_pairs.tsv is valid TSV", False, str(e))
        _validation_results["checks"] = checks
        _validation_results["all_passed"] = False
        return {"all_passed": False, "checks": checks}

    add("candidate_pairs.tsv is valid TSV", True)

    # Check columns
    req_cols = {"source1_entity_id", "candidate_entity_ids"}
    has_cols = req_cols.issubset(set(cp_df.columns))
    add("candidate_pairs.tsv has correct columns", has_cols,
        f"Found: {list(cp_df.columns)}")

    # Check: every S1 entity appears exactly once
    s1_ids_in_cp = cp_df["source1_entity_id"].tolist()
    dup_s1 = len(s1_ids_in_cp) != len(set(s1_ids_in_cp))
    add("No duplicate Source1 IDs in candidate_pairs", not dup_s1,
        f"Duplicates found: {dup_s1}")

    # Load valid S2/S3 IDs
    valid_cand_ids = set()
    s1_entity_ids  = set(s1_ids_in_cp)
    for src_dir, prefix in [(TEST_DIR, "test"), (TRAIN_DIR, "train")]:
        for num in [2, 3]:
            p = src_dir / f"{prefix}_source{num}.tsv"
            if p.exists():
                try:
                    df = pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False)
                    valid_cand_ids.update(df["entity_id"].tolist())
                except Exception:
                    pass

    # Check: no S1 self-matches in candidate_pairs
    self_matches = []
    invalid_ids  = []
    dup_cands    = []
    for _, row in cp_df.iterrows():
        s1_id    = str(row["source1_entity_id"]).strip()
        cand_str = str(row.get("candidate_entity_ids", "")).strip()
        if not cand_str or cand_str.lower() == "nan":
            continue
        cids = [c.strip() for c in cand_str.split(",") if c.strip()]
        if len(cids) != len(set(cids)):
            dup_cands.append(s1_id)
        for cid in cids:
            if cid in s1_entity_ids:
                self_matches.append(f"{s1_id}→{cid}")
            if valid_cand_ids and cid not in valid_cand_ids:
                invalid_ids.append(f"{s1_id}→{cid}")

    add("No S1 self-matches in candidate_pairs", len(self_matches) == 0,
        "; ".join(self_matches[:5]))
    add("No invalid candidate IDs", len(invalid_ids) == 0,
        "; ".join(invalid_ids[:5]))
    add("No duplicate candidate IDs per S1", len(dup_cands) == 0,
        "; ".join(dup_cands[:5]))

    # France check
    france_rows = []
    for src_dir, prefix in [(TEST_DIR, "test"), (TRAIN_DIR, "train")]:
        p = src_dir / f"{prefix}_source1.tsv"
        if p.exists():
            try:
                df = pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False)
                fr = df[df["country"].str.lower() == "france"]
                france_rows = fr["entity_id"].tolist()
            except Exception:
                pass
            break
    if france_rows:
        france_in_output = all(fid in set(s1_ids_in_cp) for fid in france_rows)
        add("France records present in output", france_in_output,
            f"France S1 IDs: {france_rows}")
    else:
        add("France records check", True, "No France records found in data")

    # matching_results checks
    if mr_exists:
        try:
            mr_df = pd.read_csv(RESULTS_PATH, sep="\t", dtype=str, keep_default_na=False)
            add("matching_results.tsv is valid TSV", True)
            req_mr = {"source1_entity_id", "matched_entity_ids"}
            add("matching_results.tsv has correct columns",
                req_mr.issubset(set(mr_df.columns)))

            # Every match must be in candidate_pairs
            cp_cands_by_s1 = {}
            for _, row in cp_df.iterrows():
                s1_id    = str(row["source1_entity_id"]).strip()
                cand_str = str(row.get("candidate_entity_ids", "")).strip()
                cids = set()
                if cand_str and cand_str.lower() != "nan":
                    cids = {c.strip() for c in cand_str.split(",") if c.strip()}
                cp_cands_by_s1[s1_id] = cids

            subset_violations = []
            for _, row in mr_df.iterrows():
                s1_id    = str(row["source1_entity_id"]).strip()
                match_str = str(row.get("matched_entity_ids", "")).strip()
                if not match_str or match_str.lower() == "nan":
                    continue
                mids = {c.strip() for c in match_str.split(",") if c.strip()}
                allowed = cp_cands_by_s1.get(s1_id, set())
                bad = mids - allowed
                if bad:
                    subset_violations.append(f"{s1_id}: {bad}")

            add("All matches are subset of candidates",
                len(subset_violations) == 0,
                "; ".join(subset_violations[:5]))

            # No S1 self-matches in matching results
            mr_self = []
            s1_ids_set = set(mr_df["source1_entity_id"].tolist())
            for _, row in mr_df.iterrows():
                s1_id    = str(row["source1_entity_id"]).strip()
                match_str = str(row.get("matched_entity_ids", "")).strip()
                if not match_str or match_str.lower() == "nan":
                    continue
                for mid in match_str.split(","):
                    mid = mid.strip()
                    if mid in s1_ids_set:
                        mr_self.append(f"{s1_id}→{mid}")
            add("No S1 self-matches in matching_results", len(mr_self) == 0,
                "; ".join(mr_self[:5]))

        except Exception as e:
            add("matching_results.tsv parsing", False, str(e))

    all_passed = all(c["passed"] for c in checks)
    _validation_results["checks"]     = checks
    _validation_results["all_passed"] = all_passed

    return {"all_passed": all_passed, "checks": checks}


@app.get("/api/validate/results")
async def get_validation_results():
    if not _validation_results:
        return {"available": False}
    return {"available": True, **_validation_results}


# ===========================================================================
# SUBMISSION PACKAGE
# ===========================================================================

@app.post("/api/submission/package")
async def create_submission_package():
    """Create a zip submission package."""
    try:
        zip_path = OUTPUT_DIR / "submission.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            # output files
            for fname in ["candidate_pairs.tsv", "matching_results.tsv"]:
                fp = OUTPUT_DIR / fname
                if fp.exists():
                    zf.write(fp, f"output/{fname}")

            # code files
            for src_file in PROJECT_ROOT.rglob("*.py"):
                rel = src_file.relative_to(PROJECT_ROOT)
                parts = rel.parts
                if any(p.startswith(".") or p in ("__pycache__", ".pytest_cache") for p in parts):
                    continue
                zf.write(src_file, f"code/business_entity_resolution/src/{rel}")

            # requirements + readme
            for fname in ["requirements.txt", "README.md", "RUN_PROJECT.md"]:
                fp = PROJECT_ROOT / fname
                if fp.exists():
                    zf.write(fp, f"code/business_entity_resolution/{fname}")

        return {
            "message":  "Submission package created",
            "path":     str(zip_path),
            "size_bytes": zip_path.stat().st_size,
        }
    except Exception as exc:
        logger.exception("Package creation failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/submission/download")
async def download_submission():
    """Download the submission zip."""
    zip_path = OUTPUT_DIR / "submission.zip"
    if not zip_path.exists():
        raise HTTPException(status_code=404, detail="No submission package yet. POST /api/submission/package first.")
    return FileResponse(str(zip_path), media_type="application/zip", filename="submission.zip")


# ---------------------------------------------------------------------------
# Serve frontend index.html at root
# ---------------------------------------------------------------------------
@app.get("/")
async def serve_frontend():
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "Business Entity Resolution API v2.0", "docs": "/docs"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
