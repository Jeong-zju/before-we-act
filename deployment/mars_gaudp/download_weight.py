import os
from pathlib import Path
from huggingface_hub import hf_hub_download

def main():
    token=Path(os.environ.get("HF_TOKEN_FILE","/workspace/.secrets/hf_token")).read_text().strip()
    out=Path(os.environ.get("MARS_GAUDP_WEIGHT","/workspace/repos/Policy-Lightning/weights/re10k.ckpt")); out.parent.mkdir(parents=True,exist_ok=True)
    if not out.exists(): hf_hub_download("botaoye/NoPoSplat","re10k.ckpt",repo_type="model",revision="main",local_dir=str(out.parent),token=token)
    print({"status":"complete","weight":str(out),"bytes":out.stat().st_size})
if __name__=="__main__": main()
