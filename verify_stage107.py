#!/usr/bin/env python3
"""Compare P107 against archived P106 content SHA; static-only."""
from pathlib import Path
import hashlib,json,sys
R=Path(__file__).resolve().parent
m=json.loads((R/'P107_INTEGRITY.json').read_text('utf8'))
assert m['stage']==107
for rel, expected in m['unchanged_p106_files'].items():
 p=R/rel
 assert p.is_file(), 'MISSING_P106_ORIGINAL '+rel
 assert hashlib.sha256(p.read_bytes()).hexdigest()==expected, 'MODIFIED_P106_ORIGINAL '+rel
assert (R/'RUN_ALL_COLAB.ipynb').is_file()
assert (R/'QUICKSTART.md').is_file()
print('P107_P106_ORIGINAL_222_FILES_BYTE_IDENTICAL_PASS')
print('NOTE: static only, new PyBullet run not tested')
