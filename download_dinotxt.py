"""Download official public dino.txt additions; reuse the existing DINOv2 backbone."""
import hashlib
import json
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEST = ROOT / '.models'
BASE = 'https://dl.fbaipublicfiles.com/dinov2/'
FILES = {
    'dinov2_vitl14_reg4_dinotxt_tet1280d20h24l_vision_head.pth': ('dinov2_vitl14/', 101317814),
    'dinov2_vitl14_reg4_dinotxt_tet1280d20h24l_text_encoder.pth': ('dinov2_vitl14/', 2152639923),
    'bpe_simple_vocab_16e6.txt.gz': ('thirdparty/', 1356917),
}


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def status(value):
    p = DEST / 'download_status.json'
    t = p.with_suffix('.tmp'); t.write_text(json.dumps(value, indent=2)); t.replace(p)


def main():
    DEST.mkdir(exist_ok=True)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler())
    records = []
    for name, (prefix, size) in FILES.items():
        target = DEST / name
        if target.exists():
            assert target.stat().st_size == size, name + ' existing size mismatch'
        else:
            temporary = target.with_suffix(target.suffix + '.part')
            # Stream bytes; .part is never treated as a complete checkpoint.
            with opener.open(BASE + prefix + name, timeout=60) as response, temporary.open('wb') as f:
                n = 0; last = 0
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    f.write(block); n += len(block)
                    if time.time() - last > 5:
                        status(dict(stage='download', file=name, bytes_received=n, expected_bytes=size))
                        last = time.time()
            assert n == size, name + ' incomplete download'
            temporary.replace(target)
        records.append(dict(file=name, bytes=size, sha256=sha(target), source=BASE + prefix + name))
        print(json.dumps(records[-1]), flush=True)
    (DEST / 'provenance.json').write_text(json.dumps(records, indent=2))
    status(dict(stage='complete', files=records))


if __name__ == '__main__':
    main()
