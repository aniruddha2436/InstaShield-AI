# Combined v4 + v5 candidate: review and rollout

The new candidate has been trained and evaluated. It has NOT replaced the active model, and this work has not been pushed to GitHub or deployed to Render.

## Review first

Read reports/combined_v4_v5/EVALUATION.md. The untouched 2,001-row test set gives precision 73.91%, recall 62.33%, F1 67.63%, ROC-AUC 0.8448, and PR-AUC 0.7300. The threshold is 0.698607861995697. The two dataset sources have different results, and this is not an unbiased improvement comparison with the previous v4-trained model. Dataset provenance, real-world performance, and calibration are unverified.

The candidate preserves eight raw inputs and the same 11 engineered features. The frontend still accepts pasted bio text and sends only its character count. No API request or response fields changed. Predictions and range warnings can change when this model is selected. Existing browser-history entries retain their original scores; they are not recomputed by a model update.

## Files

- Original CSVs: reference/instagram_fake_profile_v4.csv and reference/instagram_fake_profile_v5.csv.
- Combined CSV: reference/instagram_fake_profile_v4_v5_combined.csv, with row_id and source tracking.
- Candidate: backend/models/candidate_v4_v5/fake_profile_model.json and its paired metadata.json.
- Current active model: backend/models/fake_profile_model.json and paired metadata.
- Rollback model: backend/models/rollback_v4/fake_profile_model.json and paired metadata. Original pre-update metadata is also preserved as original_metadata.json.
- Full metrics, folds, OOF predictions, held-out predictions, and all search trials: reports/combined_v4_v5/.

## Reproduce training

Run from the project root using the root virtual environment:

~~~powershell
.\.venv\Scripts\python.exe -m pip install -r tools/training-requirements.txt
.\.venv\Scripts\python.exe tools/train_combined.py
Push-Location backend
..\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
Pop-Location
~~~

Training writes only the candidate, combined CSV and reports. It never selects the candidate, commits, pushes or deploys. The seed is 42, username groups are case-insensitive, and each shared username stays in one test/train partition and one inner validation fold. The final candidate is fitted on 7,999 training rows only; the held-out 2,001 rows are never used in search or threshold selection. CSV probabilities are exported with full float32 round-trip precision.

If your virtual environment is inside backend instead, use backend\.venv\Scripts\python.exe from the root, and .\.venv\Scripts\python.exe while inside backend.

## Activate locally after review

Stop the local FastAPI server before selecting an artifact. From the project root:

~~~powershell
.\.venv\Scripts\python.exe tools/select_model.py candidate
Push-Location backend
..\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
..\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
~~~

The selection command validates native loading, metadata, feature ordering and the candidate model checksum before copying the pair into the active paths. Restarting loads the newly selected model. This command does not push or deploy anything.

## Publish the selected candidate

Only after selecting and testing locally, open a separate terminal at the project root:

~~~powershell
git status --short
git add .gitattributes backend/app/service.py backend/models backend/tests tools reference reports README.md MODEL_AUDIT.md DEPLOYMENT.md UPDATE_V4_V5.md
git diff --cached --stat
git commit -m "Update model using combined v4 and v5 datasets"
git push origin main
~~~

Review staged files before committing. The existing Dockerfile already copies the active native model and its metadata, so no Dockerfile change is required. Candidate, rollback and reference data are not copied into the runtime image. If Render auto-deploy is enabled, the main-branch push triggers it; otherwise use Manual Deploy for the latest commit.

Wait for Render to report Live. On the deployed hostname, confirm /api/health reports threshold 0.698607861995697, then analyze both examples and a manually entered profile. Examples demonstrate inputs, not guaranteed fixed verdicts. Confirm the bio count, range warnings and optional history work. Do not claim the candidate is deployed solely because the local files changed.

## Roll back

From the root, with the local backend stopped:

~~~powershell
.\.venv\Scripts\python.exe tools/select_model.py rollback
git add backend/models/fake_profile_model.json backend/models/fake_profile_model.metadata.json
git commit -m "Restore previous v4 model"
git push origin main
~~~

Restart locally if needed and verify Render after its deploy. Rollback metadata includes the previous service's full-v4 range bounds for compatibility; the original v4 training-partition bounds were unavailable. Candidate bounds are calculated strictly from its training partition.

The .gitattributes rules preserve checksummed CSV and model bytes across Windows and Linux checkouts. Include this file in the update commit.
