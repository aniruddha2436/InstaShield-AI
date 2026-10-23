import pickle,io,json,struct,math,csv
import numpy as np
from pathlib import Path
class State:
    pass
class Safe(pickle.Unpickler):
    def find_class(self,module,name):
        if (module,name) in [('xgboost.sklearn','XGBClassifier'),('xgboost.core','Booster')]: return State
        if (module,name)==('builtins','bytearray'): return bytearray
        if (module,name)==('numpy._core.multiarray','scalar'): return np._core.multiarray.scalar
        if (module,name)==('numpy','dtype'): return np.dtype
        raise ValueError((module,name))
b=Safe(io.BytesIO((Path(__file__).resolve().parents[1] / 'backend/models/fake_profile_model.joblib').read_bytes())).load()
class UB:
    def __init__(self,b): self.b=bytes(b); self.i=0
    def take(self,n): r=self.b[self.i:self.i+n]; self.i+=n; return r
    def value(self,t=None):
        t=t or self.take(1)
        formats={b'i':('>b',1),b'U':('>B',1),b'I':('>h',2),b'l':('>i',4),b'L':('>q',8),b'd':('>f',4),b'D':('>d',8)}
        if t in formats:
            fmt,n=formats[t]; return struct.unpack(fmt,self.take(n))[0]
        if t==b'S': return self.take(self.value()).decode()
        if t==b'T': return True
        if t==b'F': return False
        if t==b'Z': return None
        if t==b'[':
            typ=None; count=None
            if self.b[self.i:self.i+1]==b'$': self.i+=1; typ=self.take(1)
            if self.b[self.i:self.i+1]==b'#': self.i+=1; count=self.value()
            if count is not None: return [self.value(typ) for _ in range(count)]
            out=[]
            while self.b[self.i:self.i+1]!=b']': out.append(self.value())
            self.i+=1; return out
        if t==b'{':
            out={}
            while self.b[self.i:self.i+1]!=b'}':
                key=self.take(self.value()).decode(); out[key]=self.value()
            self.i+=1; return out
        raise ValueError((t,self.i))
obj=UB(b['model']._Booster.handle).value()
learner=obj['Model']['learner']
print(json.dumps({'threshold':b['threshold'],'features':b['features'],'raw_cols':b['raw_cols'],'version':obj['Model']['version'],'learner_params':learner['learner_model_param'],'trees':len(learner['gradient_booster']['model']['trees']),'booster_features':learner['feature_names']},indent=2))
base=float(str(learner['learner_model_param']['base_score']).strip('[]'))
def predict(x):
    score=math.log(base/(1-base))
    for t in learner['gradient_booster']['model']['trees']:
        n=0
        while t['left_children'][n]!=-1:
            n=t['left_children'][n] if x[t['split_indices'][n]]<t['split_conditions'][n] else t['right_children'][n]
        score+=t['split_conditions'][n]
    return 1/(1+math.exp(-score))
for raw in [(0,0,'user48291',1200,3500,2,20,4),(1,90,'priya.sharma',850,600,210,1800,0)]:
    pic,bio,user,fol,fng,posts,age,changed=raw
    x=[pic,bio,fol,fng,posts,age,changed,fol/(fng+1),fol/(age+1),posts/(age+1),sum(c.isdigit() for c in user)]
    p=predict(x)
    expected=0.9940333178046427 if user=="user48291" else 0.09731775281495583
    assert abs(p-expected) < 1e-8
    print(user,p)

assert b["features"] == learner["feature_names"]
assert len(learner["gradient_booster"]["model"]["trees"]) == 150
assert abs(b["threshold"] - 0.6835914254188538) < 1e-12
print("Artifact metadata verified. This is independent tree evaluation, not a native XGBoost runtime test.")
