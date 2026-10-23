"""Explicit local promotion/rollback. Does not commit, push, or deploy."""
import argparse,json,hashlib,sys,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'backend'))
from app.service import Detector

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('selection',choices=['candidate','rollback'])
    args=parser.parse_args()
    directory=ROOT/'backend/models'/('candidate_v4_v5' if args.selection=='candidate' else 'rollback_v4')
    source=directory/'fake_profile_model.json'
    Detector(source) # validate native loading, checksums, feature order and metadata first
    target=ROOT/'backend/models'
    # Run with the local server stopped. A deployment replaces the entire image.
    for name in ['fake_profile_model.json','fake_profile_model.metadata.json']:
        temporary=target/(name+'.pending')
        temporary.write_bytes((directory/name).read_bytes())
    for name in ['fake_profile_model.json','fake_profile_model.metadata.json']:
        os.replace(target/(name+'.pending'),target/name)
    meta=json.loads((target/'fake_profile_model.metadata.json').read_text())
    print('Local active model:',meta.get('model_version','legacy-v4'))
    print('No Git push or Render deployment was performed. Restart the local backend.')

if __name__=='__main__':main()
