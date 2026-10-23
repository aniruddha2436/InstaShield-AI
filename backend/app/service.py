from pathlib import Path
import math
import hashlib
import numpy as np
import pandas as pd
import json
import xgboost as xgb
from .features import RAW_COLS, FEATURES, make_features

MODEL_PATH = Path(__file__).resolve().parents[1] / 'models' / 'fake_profile_model.json'

class Detector:
    def __init__(self, path=MODEL_PATH):
        # Load the selected native model and its paired metadata.
        path = Path(path)
        bundle = json.loads(path.with_suffix('.metadata.json').read_text(encoding='utf-8'))
        self.training_ranges = bundle.get('training_ranges')
        expected_ranges = set(RAW_COLS) - {'username'}
        if not isinstance(self.training_ranges, dict) or set(self.training_ranges) != expected_ranges:
            raise ValueError('Missing or unexpected model training ranges')
        for name, bounds in self.training_ranges.items():
            if not isinstance(bounds, list) or len(bounds) != 2 or any(isinstance(x, bool) or not isinstance(x, (int,float)) or not math.isfinite(x) for x in bounds) or bounds[0] < 0 or bounds[0] > bounds[1]:
                raise ValueError('Invalid training range: ' + name)
        if 'model_sha256' in bundle and hashlib.sha256(path.read_bytes()).hexdigest() != bundle['model_sha256']:
            raise ValueError('Model checksum does not match metadata')
        self.model = xgb.XGBClassifier(n_jobs=1)
        self.model.load_model(path)
        self.threshold = float(bundle['threshold'])
        if bundle['features'] != FEATURES or bundle['raw_cols'] != RAW_COLS:
            raise ValueError('Model feature contract differs from notebook preprocessing')
        if not math.isfinite(self.threshold) or not 0 <= self.threshold <= 1:
            raise ValueError('Invalid saved threshold')
        if self.model.get_booster().feature_names != FEATURES:
            raise ValueError('Booster feature order differs from bundle')
        if list(self.model.classes_) != [0,1]:
            raise ValueError('Unexpected class labels')
        self.model.set_params(n_jobs=1)

    def predict(self, profile):
        raw = pd.DataFrame([profile.model_dump()])[RAW_COLS]
        features = make_features(raw)
        probability = float(self.model.predict_proba(features)[0,1])
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise RuntimeError('Invalid model output')
        # Native TreeSHAP explains this exact prediction in log-odds, not percent.
        contrib = self.model.get_booster().predict(xgb.DMatrix(features), pred_contribs=True)[0]
        explanation = sorted([
            {'feature': name, 'value': float(features.iloc[0][name]),
             'contribution': float(contrib[i]),
             'direction': 'toward suspicious' if contrib[i] > 0 else 'toward genuine' if contrib[i] < 0 else 'neutral'}
            for i,name in enumerate(FEATURES)
        ], key=lambda r: abs(r['contribution']), reverse=True)[:5]
        warnings = [f'{name.replace("_", " ")}: entered value is outside the dataset range {lo}–{hi}.' for name,(lo,hi) in self.training_ranges.items() if not lo <= getattr(profile,name) <= hi]
        return {'username': profile.username, 'fake_probability': probability,
                'threshold': self.threshold, 'is_suspicious': probability >= self.threshold,
                'verdict': 'Potentially suspicious' if probability >= self.threshold else 'Lower model risk',
                'explanation': explanation, 'warnings': warnings,
                'explanation_units': 'log-odds',
                'notice': 'This model estimate is not proof of identity or fraud. Inputs are manually supplied; no Instagram lookup is performed.'}
