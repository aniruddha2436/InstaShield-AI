import math,json,hashlib
from pathlib import Path
import pytest
import numpy as np
from fastapi.testclient import TestClient
import app.main as main_module
from app.schema import Profile
from app.service import Detector,MODEL_PATH
from app.features import FEATURES,make_features
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
CANDIDATE=ROOT/'backend/models/candidate_v4_v5/fake_profile_model.json'
SAMPLES=[dict(profile_picture=0,bio_length=0,username='user48291',followers=1200,following=3500,posts=2,account_age_days=20,username_changed=4),dict(profile_picture=1,bio_length=90,username='priya.sharma',followers=850,following=600,posts=210,account_age_days=1800,username_changed=0)]

@pytest.fixture(params=[MODEL_PATH,CANDIDATE],ids=['active-v4','candidate-combined'])
def artifact(request):return request.param

@pytest.fixture
def detector(artifact):return Detector(artifact)

def test_real_model_predictions_and_explanations(detector,artifact):
    metadata=json.loads(artifact.with_suffix('.metadata.json').read_text())
    for sample in SAMPLES:
        result=detector.predict(Profile(**sample))
        assert 0<=result['fake_probability']<=1
        assert result['is_suspicious']==(result['fake_probability']>=metadata['threshold'])
        assert result['threshold']==metadata['threshold']
        assert len(result['explanation'])==5
        assert all(math.isfinite(c['contribution']) for c in result['explanation'])
    assert detector.model.get_booster().feature_names==FEATURES

def test_api_and_validation(artifact,monkeypatch):
    monkeypatch.setattr(main_module,'Detector',lambda:Detector(artifact))
    with TestClient(main_module.app) as client:
        assert client.get('/api/health').json()['status']=='ready'
        response=client.post('/api/analyze',json=SAMPLES[0]);assert response.status_code==200
        assert response.json()['threshold']==Detector(artifact).threshold
        for change in [dict(followers=-1),dict(posts=1.5),dict(followers=True),dict(followers='4'),dict(username='bad user'),dict(is_fake=1)]:
            assert client.post('/api/analyze',json={**SAMPLES[0],**change}).status_code==422
        assert client.post('/api/analyze',json={}).status_code==422
        assert client.post('/api/analyze',content='broken',headers={'Content-Type':'application/json'}).status_code==422

def test_range_warning(detector):
    result=detector.predict(Profile(**{**SAMPLES[0],'followers':2000000}))
    assert any('followers:' in w for w in result['warnings'])

def test_shap_additivity(detector):
    import xgboost as xgb
    matrix=xgb.DMatrix(make_features(pd.DataFrame(SAMPLES)))
    booster=detector.model.get_booster()
    contributions=booster.predict(matrix,pred_contribs=True)
    margins=booster.predict(matrix,output_margin=True)
    np.testing.assert_allclose(contributions.sum(axis=1),margins,atol=1e-4,rtol=0)

def test_metadata_ranges_control_warnings(tmp_path):
    path=tmp_path/'fake_profile_model.json';path.write_bytes(CANDIDATE.read_bytes())
    meta=json.loads(CANDIDATE.with_suffix('.metadata.json').read_text());meta['training_ranges']['followers']=[1000,2000]
    path.with_suffix('.metadata.json').write_text(json.dumps(meta))
    result=Detector(path).predict(Profile(**SAMPLES[1]))
    assert any('range 1000–2000' in w for w in result['warnings'])

def test_reject_mismatched_metadata(tmp_path):
    path=tmp_path/'fake_profile_model.json';path.write_bytes(CANDIDATE.read_bytes())
    source=json.loads(CANDIDATE.with_suffix('.metadata.json').read_text())
    for change in [{'features':list(reversed(FEATURES))},{'threshold':float('nan')},{'training_ranges':{}},{'model_sha256':'wrong'}]:
        path.with_suffix('.metadata.json').write_text(json.dumps({**source,**change}))
        with pytest.raises(ValueError):Detector(path)

def test_candidate_matches_saved_heldout_predictions():
    rows=pd.read_csv(ROOT/'reference/instagram_fake_profile_v4_v5_combined.csv',index_col='row_id')
    heldout=pd.read_csv(ROOT/'reports/combined_v4_v5/heldout_predictions.csv',index_col='row_id')
    actual=Detector(CANDIDATE).model.predict_proba(make_features(rows.loc[heldout.index]))[:,1]
    np.testing.assert_allclose(actual,heldout.fake_probability,rtol=0,atol=1e-7)
