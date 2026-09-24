"""Download/verify only the assets used by BERT 2D-only; no val cache required."""
import argparse,json,os
from pathlib import Path
from .paths import ASSETS,ROOT,sha256
from h20.assets import specifications,materialize
FID=[
 dict(local_path="fid/VIRTUAL_imagenet256_labeled.npz",size=2037122530,
      sha256="b32732719497e42660a9affb4a966068cba0855ac449b82015e34ec376d20758",
      url="https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz"),
 dict(local_path="fid/classify_image_graph_def.pb",size=95673916,
      sha256="009d6814d1bc560d4e7b236e170e9b2d5ca6f4b57bd8037f6db05776204415c6",
      url="https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/classify_image_graph_def.pb")]
def required():
    weights={"weights/generator_titok_l32.bin","weights/tokenizer_titok_l32.bin","weights/mot_latest.pt","router/e117.pt"}
    return [s for s in specifications() if s["local_path"].startswith(("codes/train/","routes/train/")) or s["local_path"] in weights]
def check(path,spec):
    if not path.is_file():raise FileNotFoundError(path)
    if path.stat().st_size!=spec["size"] or sha256(path)!=spec["sha256"]:
        raise ValueError("asset checksum mismatch (not overwritten): "+str(path))
def verify(root,include_fid=False):
    specs=required()+(FID if include_fid else [])
    for s in specs:check(root/s["local_path"],s)
    return len(specs)
def main(a):
    root=a.root.resolve()
    if root.is_relative_to(ROOT):raise ValueError("place large assets outside the code repo, e.g. ../assets/generation")
    if a.list:
        print(json.dumps(required()+(FID if a.fid else []),indent=2));return
    if not a.verify_only:
        from huggingface_hub import hf_hub_download
        root.mkdir(parents=True,exist_ok=True)
        def fetch(**kw):return hf_hub_download(**kw,cache_dir=str(root/".hf_download_cache"))
        for s in required():
            path=root/s["local_path"]
            if path.exists():check(path,s)
            else:materialize(s,path,fetch)
        if a.fid:
            import requests
            for s in FID:
                path=root/s["local_path"]
                if path.exists():check(path,s);continue
                path.parent.mkdir(parents=True,exist_ok=True)
                tmp=path.with_name(path.name+".download")
                if tmp.exists():raise FileExistsError("previous incomplete download; inspect before removing: "+str(tmp))
                with requests.get(s["url"],stream=True,timeout=(30,120)) as r:
                    r.raise_for_status()
                    with tmp.open("xb") as f:
                        for b in r.iter_content(8*1024**2):f.write(b)
                check(tmp,s);tmp.replace(path)
    print(json.dumps(dict(status="verified",files=verify(root,a.fid),root=str(root))))
if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=ASSETS)
    p.add_argument("--fid",action="store_true")
    p.add_argument("--verify-only",action="store_true")
    p.add_argument("--list",action="store_true")
    main(p.parse_args())
