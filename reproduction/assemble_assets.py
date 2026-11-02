#!/usr/bin/env python3
"""Reassemble *unchanged* P92C experimental input from GitHub chunks (no external deps)."""
from pathlib import Path
import hashlib, sys
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'assets'/'p92c_verified.zip'
EXPECTED='c5927d8122b32a037544a3e3eb0da35f647a6231b3f0cb4eefba2f62cf819928'
EXPECTED_SIZE=50324332
EXPECTED_PARTS=3

def main():
    if OUT.is_file():
        if OUT.stat().st_size==EXPECTED_SIZE and hashlib.sha256(OUT.read_bytes()).hexdigest()==EXPECTED:
            print('INPUT_ORIGINAL_HASH_PASS',OUT)
            return 0
        raise SystemExit('STOP: existing output has different hash; no overwrite')
    parts=[ROOT/'assets'/f'p92c_verified.zip.part{i:02}' for i in range(1,EXPECTED_PARTS+1)]
    for p in parts:
        if not p.is_file(): raise SystemExit('STOP: missing '+str(p))
    tmp=OUT.with_suffix('.zip.tmp')
    h=hashlib.sha256();total=0
    with tmp.open('xb') as f:
        for part in parts:
            with part.open('rb') as inp:
                while True:
                    chunk=inp.read(1<<20)
                    if not chunk:break
                    f.write(chunk);h.update(chunk);total+=len(chunk)
    if total!=EXPECTED_SIZE or h.hexdigest()!=EXPECTED:
        tmp.unlink(missing_ok=True)
        raise SystemExit('STOP: original archive content mismatch')
    tmp.rename(OUT)
    print('INPUT_ORIGINAL_HASH_PASS',OUT,total,h.hexdigest())
    return 0
if __name__=='__main__':sys.exit(main())
