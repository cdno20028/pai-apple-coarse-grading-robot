#!/usr/bin/env python3
"""Check original archive contents, not a PyBullet run."""
from pathlib import Path
import hashlib,io,zipfile,json,sys
ROOT=Path(__file__).resolve().parent
P=ROOT/'assets'/'p92c_verified.zip'
def digest(b): return hashlib.sha256(b).hexdigest()
def main():
    if not P.is_file(): raise SystemExit('STOP: run python reproduction/assemble_assets.py first')
    raw=P.read_bytes()
    assert len(raw)==50324332 and digest(raw)=='c5927d8122b32a037544a3e3eb0da35f647a6231b3f0cb4eefba2f62cf819928'
    with zipfile.ZipFile(io.BytesIO(raw)) as out:
        assert out.testzip() is None
        raw_source=out.read('source/PAI92_source.zip')
    assert digest(raw_source)=='68b1a4c905d44e7d3addea8e567305846ba8601e3f1fef9020180b8e1894766e'
    with zipfile.ZipFile(io.BytesIO(raw_source)) as src:
        assert src.testzip() is None
        assert len([n for n in src.namelist() if n.endswith('_POLAR_COMPOSITE_UV.png')])==11
        assert all(f'stage88/textures/apple_{x}.png' in src.namelist() for x in 'ABCD')
        with zipfile.ZipFile(io.BytesIO(src.read('REAL17_USER_SOURCE_17.zip'))) as photos:
            assert photos.testzip() is None
            assert len([n for n in photos.namelist() if n.endswith('.jfif')])==17
    print('ORIGINAL_17_PHOTOS_11_UVS_4_TEXTURES_HASH_PASS')
    print('NOTE: asset presence only, not a new simulation')
if __name__=='__main__':sys.exit(main())
