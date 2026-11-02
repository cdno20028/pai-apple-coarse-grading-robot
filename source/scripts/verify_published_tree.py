#!/usr/bin/env python3
"""Check publication tree file hashes (not a new physics simulation)."""
from pathlib import Path
import hashlib,sys
root=Path(__file__).resolve().parent.parent
p=root/'PUBLICATION_MANIFEST.sha256'
seen=set()
for row in p.read_text(encoding='utf-8').splitlines():
    if not row.strip():continue
    expected,rel=row.split('  ',1)
    f=root/rel
    assert f.is_file(),f'FILE_MISSING: {rel}'
    got=hashlib.sha256(f.read_bytes()).hexdigest()
    assert got==expected,f'HASH_MISMATCH: {rel}'
    seen.add(rel)
for f in root.rglob('*'):
    if not f.is_file() or '.git' in f.parts or '__pycache__' in f.parts or f.suffix in ('.pyc','.pyo'):continue
    rel=f.relative_to(root).as_posix()
    if rel!='PUBLICATION_MANIFEST.sha256' and rel not in seen:
        raise AssertionError('UNLISTED_FILE: '+rel)
print('PUBLICATION_TREE_VERIFIED',len(seen),'sha256 files; physics not rerun')
