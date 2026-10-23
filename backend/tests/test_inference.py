import math
from pathlib import Path
import pytest
pytest.importorskip('xgboost')
pytest.importorskip('pandas')
from fastapi.testclient import TestClient
from app.main import app
from app.schema import Profile
from app.service import Detector
from app.features import FEATURES,make_features
import pandas as pd

SAMPLES=[dict(profile_picture=0,bio_length=0,username='user48291',followers=1200,following=3500,posts=2,account_age_days=20,username_changed=4),dict(profile_picture=1,bio_length=90,username='priya.sharma',followers=850,following=600,posts=210,account_age_days=1800,username_changed=0)]
def test_real_model_predictions_and_explanations():
    detector=Detector()
    for sample,expected,suspicious in zip(SAMPLES,[.9940333,.09731775],[True,False]):
        result=detector.predict(Profile(**sample))
        assert result['fake_probability']==pytest.approx(expected,abs=1e-5)
        assert result['is_suspicious']==suspicious
        assert result['threshold']==pytest.approx(.6835914254188538)
        assert len(result['explanation'])==5
        assert all(math.isfinite(c['contribution']) for c in result['explanation'])

def test_api_and_validation():
    with TestClient(app) as client:
        assert client.get('/api/health').json()['status']=='ready'
        assert client.post('/api/analyze',json=SAMPLES[0]).status_code==200
        for change in [dict(followers=-1),dict(posts=1.5),dict(followers=True),dict(followers='4'),dict(username='bad user'),dict(is_fake=1)]:
            assert client.post('/api/analyze',json={**SAMPLES[0],**change}).status_code==422
        assert client.post('/api/analyze',json={}).status_code==422
        assert client.post('/api/analyze',content='broken',headers={'Content-Type':'application/json'}).status_code==422

def test_range_warning():
    result=Detector().predict(Profile(**{**SAMPLES[0],'followers':2000000}))
    assert any('followers:' in w for w in result['warnings'])

def test_shap_additivity():
    import xgboost as xgb
    detector=Detector()
    features=make_features(pd.DataFrame(SAMPLES))
    matrix=xgb.DMatrix(features)
    booster=detector.model.get_booster()
    contributions=booster.predict(matrix,pred_contribs=True)
    margins=booster.predict(matrix,output_margin=True)
    for c,m in zip(contributions,margins): assert float(c.sum())==pytest.approx(float(m),abs=1e-4)
