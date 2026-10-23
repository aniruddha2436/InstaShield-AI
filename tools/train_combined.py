"""Train a candidate only. Never replace the active model or publish a deployment."""
from pathlib import Path
import sys, json, hashlib, platform
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from sklearn.model_selection import StratifiedGroupKFold, RandomizedSearchCV, cross_val_predict
from sklearn.metrics import precision_recall_curve, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'backend'))
from app.features import RAW_COLS, FEATURES, make_features
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'backend/models/candidate_v4_v5'
REPORT=ROOT/'reports/combined_v4_v5'
SEED=42

def checksum(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def validate_dataset(d):
    if list(d.columns)!=RAW_COLS+['is_fake']: raise ValueError('Unexpected dataset columns or ordering')
    if d.isna().any().any(): raise ValueError('Missing data')
    if not d['username'].map(lambda x:isinstance(x,str) and 0<len(x)<=30).all(): raise ValueError('Invalid username')
    if not d['username'].str.fullmatch(r'[A-Za-z0-9._]+').all(): raise ValueError('Invalid username characters')
    for c in [c for c in d if c!='username']:
        if not pd.api.types.is_numeric_dtype(d[c]) or not np.isfinite(d[c]).all(): raise ValueError('Invalid numeric column: '+c)
        if (d[c]<0).any() or (d[c]%1!=0).any(): raise ValueError('Counts must be nonnegative integers: '+c)
    for c in ['profile_picture','is_fake']:
        if not set(d[c].unique())<={0,1}: raise ValueError('Nonbinary '+c)
    if set(d.is_fake.unique())!={0,1}: raise ValueError('Both target classes are required')
    if (d.groupby(RAW_COLS,dropna=False).is_fake.nunique()>1).any(): raise ValueError('Conflicting labels for identical inputs')

def metrics(y,p,t):
    pred=p>=t
    return {'rows':len(y),'genuine':int((y==0).sum()),'fake':int((y==1).sum()),'precision':float(precision_score(y,pred,zero_division=0)), 'recall':float(recall_score(y,pred,zero_division=0)), 'f1':float(f1_score(y,pred,zero_division=0)), 'roc_auc':float(roc_auc_score(y,p)), 'pr_auc':float(average_precision_score(y,p)), 'confusion_matrix':confusion_matrix(y,pred,labels=[0,1]).tolist()}

def train():
    OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    paths=[ROOT/'reference/instagram_fake_profile_v4.csv',ROOT/'reference/instagram_fake_profile_v5.csv']
    pieces=[]
    for path in paths:
        d=pd.read_csv(path);validate_dataset(d);d['source']=path.stem;pieces.append(d)
    combined=pd.concat(pieces,ignore_index=True)
    if (combined.groupby(RAW_COLS).is_fake.nunique()>1).any():raise ValueError('Cross-source label conflict')
    duplicates=int(combined.duplicated(RAW_COLS+['is_fake']).sum())
    combined=combined.drop_duplicates(RAW_COLS+['is_fake']).reset_index(drop=True)
    combined.index.name='row_id'
    combined_path=ROOT/'reference/instagram_fake_profile_v4_v5_combined.csv'
    combined.to_csv(combined_path,index=True)
    X=make_features(combined[RAW_COLS]);y=combined.is_fake.to_numpy();groups=combined.username.str.lower().to_numpy()
    outer=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=SEED)
    tr,te=next(outer.split(X,y,groups))
    assert not set(groups[tr])&set(groups[te])
    Xt=X.iloc[tr];yt=y[tr];gt=groups[tr]
    cv=list(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=SEED).split(Xt,yt,gt))
    for a,b in cv:
        assert not set(gt[a])&set(gt[b])
        assert not (set(gt[a])|set(gt[b]))&set(groups[te])
    weight=float((yt==0).sum()/(yt==1).sum())
    model=xgb.XGBClassifier(scale_pos_weight=weight,eval_metric='logloss',n_jobs=2,random_state=SEED)
    params={'n_estimators':[150,250,400,600],'learning_rate':[.01,.03,.05,.1],'max_depth':[2,3,4,5,6],'min_child_weight':[1,3,5,10],'subsample':[.7,.85,1.],'colsample_bytree':[.7,.85,1.],'gamma':[0,.5,1,2],'reg_lambda':[.5,1,2,5]}
    print(f'Training rows {len(tr)}; untouched test rows {len(te)}; 25 configurations x 5 grouped folds',flush=True)
    search=RandomizedSearchCV(model,params,n_iter=25,scoring='average_precision',cv=cv,random_state=SEED,n_jobs=1,verbose=1,error_score='raise')
    search.fit(Xt,yt)
    pd.DataFrame(search.cv_results_).to_csv(REPORT/'search_results.csv',index=False)
    print('Computing training-only out-of-fold threshold',flush=True)
    oof=cross_val_predict(search.best_estimator_,Xt,yt,cv=cv,method='predict_proba',n_jobs=1)[:,1]
    prec,rec,thresholds=precision_recall_curve(yt,oof)
    f1=2*prec[:-1]*rec[:-1]/(prec[:-1]+rec[:-1]+1e-9)
    threshold=float(thresholds[np.argmax(f1)])
    best=search.best_estimator_ # refit by search on training rows only
    model_path=OUT/'fake_profile_model.json';best.save_model(model_path)
    restored=xgb.XGBClassifier(n_jobs=1);restored.load_model(model_path)
    original=best.predict_proba(X)[:,1];reloaded=restored.predict_proba(X)[:,1]
    np.testing.assert_allclose(original,reloaded,rtol=0,atol=1e-8)
    probability=reloaded[te]
    results={'overall':metrics(y[te],probability,threshold),'by_source':{}}
    for source in combined.source.unique():
        mask=combined.iloc[te].source.to_numpy()==source
        results['by_source'][source]=metrics(y[te][mask],probability[mask],threshold)
    ranges={c:[int(combined.iloc[tr][c].min()),int(combined.iloc[tr][c].max())] for c in RAW_COLS if c!='username'}
    assignments=combined[['username','source','is_fake']].copy();assignments['partition']='test';assignments.loc[tr,'partition']='train';assignments['cv_fold']=-1
    for fold,(_,v) in enumerate(cv):assignments.loc[tr[v],'cv_fold']=fold
    assignments.to_csv(REPORT/'split_assignments.csv',index=True)
    predictions=combined.iloc[te][['username','source','is_fake']].copy();predictions['fake_probability']=probability.astype(float);predictions['prediction']=(probability>=threshold).astype(int);predictions.to_csv(REPORT/'heldout_predictions.csv',index=True,float_format='%.17g')
    oof_frame=combined.iloc[tr][['username','source','is_fake']].copy();oof_frame['fake_probability']=oof.astype(float);oof_frame.to_csv(REPORT/'training_oof_predictions.csv',index=True,float_format='%.17g')
    metadata={'model_version':'combined-v4-v5-seed42','threshold':threshold,'features':FEATURES,'raw_cols':RAW_COLS,'labels':{'0':'genuine','1':'fake'},'training_ranges':ranges,'range_source':'training partition only','dataset_checksums':{p.name:checksum(p) for p in paths+[combined_path]},'model_sha256':checksum(model_path),'rows':len(combined),'removed_duplicates':duplicates,'class_counts':{'genuine':int((y==0).sum()),'fake':int((y==1).sum())},'split':{'seed':SEED,'method':'first fold of five-fold StratifiedGroupKFold','group':'case-insensitive username','train_rows':len(tr),'test_rows':len(te),'cv_folds':5,'username_overlap':0,'assignment_sha256':checksum(REPORT/'split_assignments.csv')},'best_params':search.best_params_,'best_cv_pr_auc':float(search.best_score_),'threshold_selection':'maximum training-only grouped out-of-fold F1; test excluded','evaluation':results,'serialization_max_abs_error':float(np.max(np.abs(original-reloaded))),'runtime':{'python':platform.python_version(),'xgboost':xgb.__version__,'sklearn':sklearn.__version__,'numpy':np.__version__,'pandas':pd.__version__},'provenance_notice':'Dataset provenance, real-world representativeness and probability calibration are unverified. Candidate only; not promoted or deployed.'}
    (OUT/'fake_profile_model.metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    (REPORT/'evaluation.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(json.dumps(results,indent=2),flush=True)
    print('CANDIDATE COMPLETE; active model unchanged',flush=True)

if __name__=='__main__':train()
