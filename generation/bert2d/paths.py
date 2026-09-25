"""Relocatable paths for the BERT-only delivery."""
import os,sys,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
# Keep generation imports relative to this component root.
if not (ROOT/"h20/assets.py").exists() and (ROOT/"delivery_h20_20260910/h20/assets.py").exists():
    sys.path.insert(0,str(ROOT/"delivery_h20_20260910"))
ASSETS=Path(os.environ.get("MOTAR_ASSETS",ROOT.parent/"assets/generation"))
RESULT_ROOT=Path(os.environ.get("MOTAR_RESULTS",ROOT.parent/"results/generation/bert2d"))
EVALUATOR=ROOT/"bert2d/vendor/adm_evaluator.py"
REFERENCE=ASSETS/"fid/VIRTUAL_imagenet256_labeled.npz"
GRAPH=ASSETS/"fid/classify_image_graph_def.pb"
ONE_D=dict(guidance_scale=4.5,guidance_decay="linear",randomize_temperature=9.5,
           softmax_temperature_annealing=False,num_sample_steps=8)
def enable_h20(): pass
def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(16*1024**2),b""):h.update(b)
    return h.hexdigest()
digest=sha256
def atomic_json(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(data,indent=2,sort_keys=True,allow_nan=False))
    tmp.replace(path)
def output_path(path):
    path=Path(path).resolve();root=RESULT_ROOT.resolve()
    if path==root or not path.is_relative_to(root):raise ValueError("output must be a named child of MOTAR_RESULTS")
    if path.is_relative_to(ROOT):raise ValueError("large outputs must be outside the code repository")
    return path
