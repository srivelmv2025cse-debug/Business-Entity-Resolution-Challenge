# RUN_PROJECT.md – Business Entity Resolution Challenge

Complete step-by-step guide to run the end-to-end system.

---

## 1. Installation

```powershell
# From project root
python -m pip install -r requirements.txt
```

**Requirements:** Python 3.9+, see requirements.txt for all packages.

---

## 2. Dataset Placement

Place official TSV files in:

```
dataset/
├── train/
│   ├── train_source1.tsv
│   ├── train_source2.tsv
│   ├── train_source3.tsv
│   └── train_ground_truth.tsv
└── test/
    ├── test_source1.tsv
    ├── test_source2.tsv
    └── test_source3.tsv
```

> **Note:** If official data is absent, a synthetic development dataset is auto-created for testing.

Alternatively, upload files via the web dashboard → **Data Upload** page.

---

## 3. Training Command

```powershell
# Train the LightGBM entity matching model
python -m backend.models.trainer
```

Outputs:
- `models/entity_match_model.pkl` — trained classifier
- `models/threshold.json`         — F0.5-optimized threshold

---

## 4. Prediction Command

```powershell
# Generate candidate pairs (Member 1)
python -m backend.blocking.candidate_generator

# Generate matching results (Member 2)
python -m backend.models.predictor
```

---

## 5. Evaluation Command

```powershell
# Run with training ground truth
python -m backend.blocking.candidate_generator --train
python -m backend.models.predictor
```

Evaluation metrics are shown in the web dashboard → **Evaluation** page.

---

## 6. Backend Command

```powershell
# Start the FastAPI backend (port 8000)
python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```

API docs available at: http://localhost:8000/docs

---

## 7. Frontend Command

The frontend is served by FastAPI at:

```
http://localhost:8000/
```

Open this URL in your browser after starting the backend.

---

## 8. Validation Command

**Via API:**
```powershell
curl -X POST http://localhost:8000/api/validate
```

**Via Dashboard:**
Navigate to http://localhost:8000/ → **Validation** page → Click "Run Validation"

**Via pytest:**
```powershell
python -m pytest tests/ -v
```

---

## 9. Output Locations

| File | Path |
|------|------|
| Candidate pairs | `output/candidate_pairs.tsv` |
| Matching results | `output/matching_results.tsv` |
| Trained model | `models/entity_match_model.pkl` |
| Threshold config | `models/threshold.json` |
| Submission zip | `output/submission.zip` |

---

## 10. Full End-to-End Workflow

```powershell
# Step 1: Install
python -m pip install -r requirements.txt

# Step 2: Start backend
python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload

# Step 3: Open browser
# Navigate to http://localhost:8000/

# Step 4: (In another terminal or via dashboard)
# Upload data, or place TSV files in dataset/

# Step 5: Run preprocessing + blocking
python -m backend.blocking.candidate_generator

# Step 6: Train model
python -m backend.models.trainer

# Step 7: Run test prediction
python -m backend.models.predictor

# Step 8: Validate
curl -X POST http://localhost:8000/api/validate

# Step 9: Download outputs
# Via dashboard Output page or:
# output/candidate_pairs.tsv
# output/matching_results.tsv
```

---

## Fair Play Notice

This system **never** calls:
- Google Maps or geocoding APIs
- External business databases
- Government business registries
- External entity resolution APIs
- Any external internet data

All ML processing uses **only** the supplied challenge dataset.
