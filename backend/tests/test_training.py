import importlib.util,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import precision_recall_curve,f1_score,confusion_matrix
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('combined_training',ROOT/'tools/train_combined.py')
training=importlib.util.module_from_spec(spec);spec.loader.exec_module(training)
REPORT=ROOT/'reports/combined_v4_v5'

def test_dataset_validation():
    valid=pd.read_csv(ROOT/'reference/instagram_fake_profile_v5.csv')
    training.validate_dataset(valid)
    for col,value in [('followers',-1),('following',1.5),('profile_picture',2),('is_fake',3),('username','bad user'),('bio_length',np.nan)]:
        broken=valid.copy();broken[col]=broken[col].astype(object);broken.loc[0,col]=value
        with pytest.raises(ValueError):training.validate_dataset(broken)
    with pytest.raises(ValueError):training.validate_dataset(valid.drop(columns=['posts']))
    duplicate=valid.iloc[[0]].copy();duplicate['is_fake']=1-duplicate['is_fake']
    with pytest.raises(ValueError):training.validate_dataset(pd.concat([valid,duplicate],ignore_index=True))

def test_partition_and_cv_group_isolation():
    split=pd.read_csv(REPORT/'split_assignments.csv')
    split['group']=split.username.str.lower()
    assert split.groupby('group').partition.nunique().max()==1
    train=split[split.partition=='train'];test=split[split.partition=='test']
    assert train.groupby('group').cv_fold.nunique().max()==1
    assert set(train.cv_fold)==set(range(5))
    assert set(test.cv_fold)=={-1}
    assert not set(train.group)&set(test.group)
    assert .18<len(test)/len(split)<.22
    assert len(split)==10000
    assert split.is_fake.value_counts().to_dict()=={0:8501,1:1499}
    for fold in range(5):
        validation=train[train.cv_fold==fold];fit=train[train.cv_fold!=fold]
        assert set(validation.is_fake)=={0,1}
        assert not set(validation.group)&set(fit.group)

def test_training_only_threshold_and_report():
    meta=json.loads((REPORT/'evaluation.json').read_text())
    oof=pd.read_csv(REPORT/'training_oof_predictions.csv')
    p,r,t=precision_recall_curve(oof.is_fake,oof.fake_probability)
    expected=t[np.argmax(2*p[:-1]*r[:-1]/(p[:-1]+r[:-1]+1e-9))]
    assert meta['threshold']==pytest.approx(expected,abs=1e-12)
    heldout=pd.read_csv(REPORT/'heldout_predictions.csv')
    for source,report in [('overall',meta['evaluation']['overall']),*meta['evaluation']['by_source'].items()]:
        subset=heldout if source=='overall' else heldout[heldout.source==source]
        score=training.metrics(subset.is_fake.to_numpy(),subset.fake_probability.to_numpy(),meta['threshold'])
        for key in ['precision','recall','f1','roc_auc','pr_auc']:assert score[key]==pytest.approx(report[key])
        assert score['confusion_matrix']==report['confusion_matrix']
    train_rows=pd.read_csv(ROOT/'reference/instagram_fake_profile_v4_v5_combined.csv',index_col='row_id').loc[oof.row_id]
    for col,bounds in meta['training_ranges'].items():assert bounds==[int(train_rows[col].min()),int(train_rows[col].max())]
    for filename,digest in meta['dataset_checksums'].items():assert hashlib.sha256((ROOT/'reference'/filename).read_bytes()).hexdigest()==digest
    assert len(pd.read_csv(REPORT/'search_results.csv'))==25


def test_local_selection_and_rollback(tmp_path,monkeypatch):
    import shutil,sys
    spec=importlib.util.spec_from_file_location('selection',ROOT/'tools/select_model.py')
    selection=importlib.util.module_from_spec(spec);spec.loader.exec_module(selection)
    modelroot=tmp_path/'backend/models';modelroot.mkdir(parents=True)
    for directory in ['candidate_v4_v5','rollback_v4']:
        (modelroot/directory).mkdir()
        for name in ['fake_profile_model.json','fake_profile_model.metadata.json']:
            shutil.copyfile(ROOT/'backend/models'/directory/name,modelroot/directory/name)
    original=(modelroot/'rollback_v4/fake_profile_model.json').read_bytes()
    candidate=(modelroot/'candidate_v4_v5/fake_profile_model.json').read_bytes()
    monkeypatch.setattr(selection,'ROOT',tmp_path)
    monkeypatch.setattr(sys,'argv',['select_model.py','candidate']);selection.main()
    assert (modelroot/'fake_profile_model.json').read_bytes()==candidate
    monkeypatch.setattr(sys,'argv',['select_model.py','rollback']);selection.main()
    assert (modelroot/'fake_profile_model.json').read_bytes()==original
    assert (modelroot/'candidate_v4_v5/fake_profile_model.json').read_bytes()==candidate
