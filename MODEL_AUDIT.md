# Model inspection and Windows loading repair

The original joblib is preserved unchanged. Its memory snapshot fails during native XGBoost deserialization on Windows with "input stream corrupted", even with installed version 3.4.1 matching its embedded version. The exact cross-platform snapshot failure cause has not been established. This does not imply that its saved trees are damaged.

## Repair

A restricted pickle reader recovered the artifact's model state without invoking XGBoost snapshot deserialization. The embedded UBJSON Model document was extracted, loaded through the native XGBoost model loader, and re-saved by XGBoost as backend/models/fake_profile_model.json. Threshold, ordered features and raw columns are stored in fake_profile_model.metadata.json. The backend uses these portable files directly, with startup contract checks. No training occurred; tree splits, leaf weights, feature engineering, labels and threshold are unchanged.

XGBoost documents native model IO separately from pickle memory snapshots: [Model IO and compatibility](https://xgboost.readthedocs.io/en/stable/tutorials/saving_model.html).

## Verified after repair

- Python 3.12 / XGBoost 3.4.1 from the user's root .venv.
- Complete backend suite: **9 passed**, including both real prediction examples, API readiness and validation, training-range warnings, and native TreeSHAP additivity.
- **11 invalid-input subtests passed**.
- Preprocessing exactly matches the notebook's make_features on all **5,000 rows**.
- Native predictions agree with independent traversal of all original saved trees on every CSV row: maximum absolute probability difference **1.767582124001521e-7**, below 1e-6. Inputs were converted to float32 for native split-comparison parity.
- Notebook reference probabilities reproduced: user48291 approximately **0.9940333**, priya.sharma approximately **0.09731775**.
- Frontend production build previously passed; the model-loading repair does not change the frontend. Browser visual inspection previously timed out.

## Dataset and training

5,000 rows, nine columns, 4,340 genuine (0), 660 fake (1), no missing values or duplicate rows. Raw inputs: profile_picture, bio_length, username, followers, following, posts, account_age_days, username_changed. The target is is_fake.

The notebook compares logistic regression, random forest, weighted XGBoost, and SMOTE plus XGBoost. It explicitly tunes and saves weighted XGBoost: 150 trees, 11 features, saved threshold **0.6835914254188538**. It uses a stratified 80/20 split with seed 42, five-fold cross-validation, 25 randomized search trials, and out-of-fold F1 threshold selection. No logistic-regression log transform or scaler is applied to the saved XGBoost model. SMOTE is not applied at inference.

Derived signals are followers/(following+1), followers/(account_age_days+1), posts/(account_age_days+1), and username digit count. Ordered features are checked against the portable model's native feature names.

Numeric dataset ranges: profile_picture 0–1; bio_length 0–147; followers 2–490299; following 15–7500; posts 1–5000; account_age_days 10–5800; username_changed 0–5. Range warnings describe dataset coverage; operational input caps do not claim platform limits.

## Interpretation

The model estimates association with the supplied fake-profile labels. It does not prove identity or fraud. Native TreeSHAP contributions explain this particular model's prediction in log-odds, not percentage points or causal evidence. Dataset provenance, representativeness and real-world calibration remain unestablished. No live Instagram access, scraping, image analysis or private-data access is implemented. Optional history stores summaries only in the browser.
