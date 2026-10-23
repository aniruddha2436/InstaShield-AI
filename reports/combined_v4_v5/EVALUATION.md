# Combined v4 + v5 candidate evaluation

Candidate only: the active model and live Render predictions have not been replaced.

## Data and evaluation design

10,000 rows: 8,501 genuine (0), 1,499 fake (1). Both originals are preserved, and the combined CSV includes row_id and source. Neither source has missing values or duplicates, there are no identical combined input rows or conflicting input labels, and 289 usernames occur in both sources. Case-insensitive username groups are kept in a single partition and CV fold.

The first seeded five-fold StratifiedGroupKFold split reserves 2,001 test rows, with 7,999 rows available for training. Inner tuning uses five group-stratified folds and 25 randomized XGBoost configurations. Test labels are excluded from parameter and threshold selection. The decision threshold, **0.698607861995697**, maximizes training-only out-of-fold F1. The final candidate is fitted on the training partition only, not refitted on the held-out test data.

## Held-out results

| Test subset | Rows | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Combined | 2001 | 73.91% | 62.33% | 67.63% | 0.8448 | 0.7300 |
| v4 | 989 | 60.34% | 58.82% | 59.57% | 0.8326 | 0.6384 |
| v5 | 1012 | 85.40% | 64.64% | 73.58% | 0.8546 | 0.7825 |

PR-AUC is average precision, as in the original notebook. Positive class is fake. The combined random-ranking PR baseline is 14.99% fake prevalence. The candidate misses 113 of 300 labeled fake profiles and flags 66 of 1,701 genuine profiles in this test set. Precision and recall should both be considered before promotion.

The v4 and v5 results differ materially. Adding data does not by itself prove improvement. The original deployed model was trained on v4, so evaluating that existing model on this combined test partition would not be an unbiased comparison: it may have already seen v4 test rows. No such comparison is claimed. Test results describe these supplied datasets, not confirmed real-world accuracy or calibrated fraud probabilities.

## Confusion matrices

Rows are actual labels; columns are predicted labels. Threshold is the saved training-only threshold.

### Combined

| Actual / Predicted | Genuine | Fake |
|---|---:|---:|
| Genuine | 1635 | 66 |
| Fake | 113 | 187 |

### v4

| Actual / Predicted | Genuine | Fake |
|---|---:|---:|
| Genuine | 824 | 46 |
| Fake | 49 | 70 |

### v5

| Actual / Predicted | Genuine | Fake |
|---|---:|---:|
| Genuine | 811 | 20 |
| Fake | 64 | 117 |

## Model and reproducibility

The feature interface remains eight manual inputs mapped to the same 11 features. No additional scaler, text-language analysis or Instagram capability was introduced. Bio text still produces a character count in the frontend.

Best parameters:

~~~json
{
  "subsample": 0.7,
  "reg_lambda": 0.5,
  "n_estimators": 150,
  "min_child_weight": 5,
  "max_depth": 3,
  "learning_rate": 0.05,
  "gamma": 0,
  "colsample_bytree": 0.85
}
~~~

Best mean CV PR-AUC: 0.692683. Native JSON save/load prediction difference across all 10,000 rows: **0**. The model metadata records original/combined CSV checksums, model checksum, split assignment checksum, feature order, training-partition ranges, runtime versions and evaluation metrics.

Inspect search_results.csv for all 25 trials; split_assignments.csv for partition and inner-fold membership; training_oof_predictions.csv for threshold selection; heldout_predictions.csv for final test predictions; and evaluation.json for complete machine-readable results.

## Deployment status

Candidate file: backend/models/candidate_v4_v5/fake_profile_model.json, with its paired metadata file. Existing active model: backend/models/fake_profile_model.json. Rollback copy: backend/models/rollback_v4/. No Git push or Render deployment has been performed for this update. See UPDATE_V4_V5.md for activation and rollback instructions.

## Completed verification

The full backend suite passed 19 tests and 14 subtests. An additional isolated candidate-selection and rollback test also passed (20 tests covered in total). Both active and candidate models passed API input validation, metadata range checks, native explanation additivity and prediction checks. Native save/load parity was exact on all 10,000 rows. The active model file still matches its preserved rollback copy byte-for-byte. The v5 source file is unchanged. A dependency deprecation warning from Starlette/httpx remains non-failing.
