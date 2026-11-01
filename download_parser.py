"""Fetch only a public teacher's safetensors/config at one pinned HF revision."""
import hashlib
import json
import urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parent
REPO='fashn-ai/fashn-human-parser'


def main():
    opener=urllib.request.build_opener(urllib.request.ProxyHandler())
    with opener.open('https://huggingface.co/api/models/'+REPO,timeout=30) as r:info=json.load(r)
    revision=info['sha'];directory=ROOT/'.models/fashn-human-parser'/revision;directory.mkdir(parents=True,exist_ok=True)
    allowed={'config.json','model.safetensors','preprocessor_config.json'}
    available={x['rfilename'] for x in info['siblings']}
    assert allowed<=available,'Expected standard model files are missing'
    records=[]
    for name in sorted(allowed):
        path=directory/name
        if not path.exists():
            temp=path.with_suffix(path.suffix+'.part')
            with opener.open('https://huggingface.co/'+REPO+'/resolve/'+revision+'/'+name,timeout=60) as r,temp.open('wb') as f:
                while True:
                    block=r.read(1024*1024)
                    if not block:break
                    f.write(block)
            temp.replace(path)
        digest=hashlib.sha256(path.read_bytes()).hexdigest();records.append(dict(file=name,size=path.stat().st_size,sha256=digest))
        print(name,path.stat().st_size,flush=True)
    record=dict(repo=REPO,revision=revision,path=str(directory),files=records)
    (directory.parent/'provenance.json').write_text(json.dumps(record,indent=2))
    print(json.dumps(record),flush=True)


if __name__=='__main__':main()
