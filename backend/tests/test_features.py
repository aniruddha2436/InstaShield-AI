import ast
import json
import unittest
from pathlib import Path
import pandas as pd
from app.features import make_features, FEATURES
ROOT=Path(__file__).resolve().parents[2]

class FeatureTests(unittest.TestCase):
    def test_exact_notebook_parity_all_rows(self):
        nb=json.loads((ROOT/'reference/Fake_profile.ipynb').read_text(encoding='utf-8'))
        source=next(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code' and 'def make_features' in ''.join(c['source']))
        tree=ast.parse(source)
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='make_features')
        scope={'FEATURES':FEATURES}
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'notebook_features','exec'),scope)
        for filename in ['instagram_fake_profile_v4.csv','instagram_fake_profile_v5.csv','instagram_fake_profile_v4_v5_combined.csv']:
            with self.subTest(filename=filename):
                raw=pd.read_csv(ROOT/'reference'/filename).drop(columns=['is_fake'])
                pd.testing.assert_frame_equal(make_features(raw),scope['make_features'](raw))
    def test_zero_denominators_and_order(self):
        raw=pd.DataFrame([dict(profile_picture=0,bio_length=0,username='a12._3',followers=7,following=0,posts=2,account_age_days=0,username_changed=0)])
        result=make_features(raw)
        self.assertEqual(list(result.columns),FEATURES)
        self.assertEqual(result.iloc[0].tolist(),[0,0,7,0,2,0,0,7,7,2,3])

if __name__=='__main__': unittest.main()
