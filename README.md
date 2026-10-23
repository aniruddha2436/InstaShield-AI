# InstaShield AI

Local Instagram profile risk analysis using the supplied trained XGBoost artifact. Manual inputs only: no Instagram API, scraping, image analysis, identity verification, or account login. A prediction is a model estimate, not proof of fraud.

## Windows setup

Install **64-bit Python 3.11 or 3.12** and **Node.js 22.12+**. Internet access is needed for the initial dependency installation. Extract this project to a normal local folder, then open PowerShell in that folder.

### Terminal 1 — backend

```powershell
cd backend
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Use `py -3.11` instead if Python 3.11 is installed. Activation is unnecessary, so these commands work without changing PowerShell execution policy. Keep this terminal open. The backend loads the included portable native model once at startup; startup fails explicitly if its contract or dependencies are incompatible.

### Terminal 2 — frontend

Open another PowerShell terminal in the project folder:

```powershell
cd frontend
npm.cmd install
npm.cmd run dev
```

Open **http://127.0.0.1:5173**. The Vite development server forwards `/api` requests to FastAPI at port 8000. API documentation is at **http://127.0.0.1:8000/docs**; readiness is at **http://127.0.0.1:8000/api/health**.

For a local production-build preview, keep FastAPI running and use:

```powershell
cd frontend
npm.cmd run build
npm.cmd run preview
```

Open the address printed by Vite (normally http://127.0.0.1:4173). The preview also proxies `/api`. Deploying beyond localhost requires an appropriate web server/reverse proxy, HTTPS, access controls, and resource limits; this project is configured for local academic use.

## Using the app

Enter the username without `@`, picture presence, pasted bio text, followers, following, posts, account age in days, and known username-change count. All fields are required; do not substitute guesses for unavailable information. The two example buttons reproduce notebook examples using the real backend. Results include the fake-class probability, saved threshold, strongest per-profile model contributions, and warnings when inputs lie outside the dataset range.

History is off by default. When enabled, up to ten username/score/verdict/time summaries are stored in this browser's local storage. Use **Clear history** to delete them. The backend does not persist submitted profiles.

## Model contract

- Dataset: 5,000 rows, nine columns, 4,340 label `0` (genuine), 660 label `1` (fake); no missing values or duplicate rows.
- Model: supplied `XGBClassifier`, 150 trees, serialized XGBoost version **3.4.1**.
- Threshold: **0.6835914254188538**, inclusive (`probability >= threshold`).
- Feature engineering is outside the saved model; the backend reproduces the notebook exactly. No extra scaling or log transform is applied to this tuned XGBoost model. Those transformations belong to the notebook's separate logistic-regression candidate.
- Four derived features: `followers / (following + 1)`, `followers / (account_age_days + 1)`, `posts / (account_age_days + 1)`, and the number of digit characters in the username.
- Ordered features: `profile_picture`, `bio_length`, `followers`, `following`, `posts`, `account_age_days`, `username_changed`, `followers_following_ratio`, `followers_per_day`, `posts_per_day`, `username_digits`.
- Startup checks compare both bundle metadata and booster feature names. The backend loads fake_profile_model.json and its metadata.json companion. The original joblib is preserved for reference; it is not deserialized during startup. There is no model upload endpoint.
- Explanations use the model's native TreeSHAP contributions in **log-odds**, not percentage points. Positive values push toward the fake class. They describe model behavior, not causal findings or evidence against a person.

The notebook uses a stratified 80/20 split with seed 42, five-fold cross-validation, 25-search tuning configurations, class weighting, and an out-of-fold F1-selected threshold. The supplied dataset's provenance and real-world representativeness are not established by these files. No live accuracy, calibrated probability, or real-world detection guarantee is claimed.

## Troubleshooting

- **Cannot reach backend:** confirm Terminal 1 reports successful startup and `/api/health` returns ready. Keep port 8000 unchanged or edit both proxy targets in `frontend/vite.config.js`.
- **Module not found / incompatible artifact:** use the backend virtual environment's Python exactly as shown; install all requirements. XGBoost is pinned to the artifact's version. Do not silently replace or retrain the model.
- **XGBoost DLL load error:** use 64-bit Python; install the Microsoft Visual C++ Redistributable for x64 if required by the XGBoost wheel.
- **Port in use:** stop the older local server or change the corresponding server/proxy port together.
- **Validation error:** counts must be nonnegative whole numbers. Username supports letters, digits, dots, and underscores, up to 30 characters. Unknown fields, booleans for counts, fractional values, and numeric strings in raw API JSON are rejected.
- Browser history can be unavailable in restricted browsing modes. The current result still appears if storage cannot be used.

If you already created .venv at the project root, run `..\.venv\Scripts\python.exe` from backend instead of creating another environment.

See `MODEL_AUDIT.md` for inspection and verification details. Original notebook and CSV are included in `reference/`; they are not loaded by the web request handler.

Reference documentation: [FastAPI request validation](https://fastapi.tiangolo.com/tutorial/body/), [Vite setup](https://vite.dev/guide/), [XGBoost Python API](https://xgboost.readthedocs.io/en/stable/python/python_api.html).

## Fixed: input stream corrupted

The original joblib contains a memory snapshot that fails to deserialize in the installed Windows XGBoost runtime. The backend now uses the same recovered trees in XGBoost’s native JSON format with a separate metadata file. No retraining or feature changes were made. All nine backend tests passed, including real predictions, API validation, and native TreeSHAP additivity. Native predictions matched the original saved trees on all 5,000 rows within 1e-6.

If your terminal is already in backend and .venv is at the project root, restart with:

```powershell
..\.venv\Scripts\python.exe -m pytest -q
..\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

### Automatic bio length

Paste the bio into the text area. An empty bio counts as zero. The frontend counts Unicode code points (including spaces, line breaks and emoji components), matching Python string length. It sends only bio_length to the backend; it does not send or store the bio text. The existing model uses length only and does not interpret the wording. The limit is an operational 10,000 characters, not a claim about Instagram limits.

## Hosting

Render deployment files are included. See [DEPLOYMENT.md](DEPLOYMENT.md). In production the FastAPI service also serves the built React frontend at its root URL.

## Combined v4 + v5 candidate

A separate 10,000-row candidate has been trained with username-grouped evaluation. The default deployed artifact remains the preserved v4 model until explicitly selected. See [UPDATE_V4_V5.md](UPDATE_V4_V5.md) and [the evaluation report](reports/combined_v4_v5/EVALUATION.md). Range warnings now come from each model’s paired metadata; the v4 metadata retains its previous legacy bounds.
