# -*- coding: utf-8 -*-
"""Verify all 17 original photos, 11 polar composites, and 8 runtime bindings."""
from pathlib import Path
import hashlib, json, zipfile, collections
ROOT=Path(__file__).resolve().parents[1]
sha=lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
raw=ROOT/'REAL17_USER_SOURCE_17.zip'
assert raw.is_file()
with zipfile.ZipFile(raw) as z:
    assert z.testzip() is None and len(z.namelist())==17
    for n in z.namelist():
        assert (ROOT/'REAL17_SOURCE_ORIGINAL_17'/n).is_file()
        assert z.read(n)==(ROOT/'REAL17_SOURCE_ORIGINAL_17'/n).read_bytes(), 'ORIGINAL_BYTES_MISMATCH '+n
m=json.loads((ROOT/'REAL17_SOURCE_COMPOSITE_MANIFEST.json').read_text('utf8'))
assert m['version']=='REAL17_POLAR_COMPOSITE_V1'
assert m['source_total']==17 and m['side_total']==11
assert m['top_bottom_same_individual_confirmed'] is False
assert len(m['textures'])==11 and len(m['runtime_bindings'])==8
top_counts=collections.Counter(x['top_cap_photo'] for x in m['textures'])
bottom_counts=collections.Counter(x['bottom_cap_photo'] for x in m['textures'])
assert set(top_counts)==set(m['top_cap_sources']) and min(top_counts.values())>=3
assert set(bottom_counts)==set(m['bottom_cap_sources']) and min(bottom_counts.values())>=3
for e in m['textures']:
    source=ROOT/'REAL17_SOURCE_ORIGINAL_17'/e['side_photo']
    top=ROOT/'REAL17_SOURCE_ORIGINAL_17'/e['top_cap_photo']
    bottom=ROOT/'REAL17_SOURCE_ORIGINAL_17'/e['bottom_cap_photo']
    uv=ROOT/e['uv']
    assert sha(source)==e['side_photo_sha256']
    assert sha(top)==e['top_cap_photo_sha256']
    assert sha(bottom)==e['bottom_cap_photo_sha256']
    assert uv.is_file() and sha(uv)==e['uv_sha256']
    assert e['same_individual_confirmed'] is False
prep=ROOT/'stage88/stage58_apple_photo_assets/prepared'
assets=json.loads((prep/'manifest.json').read_text('utf8'))
assert len(assets['assets'])==8
for a in assets['assets']:
    tex=prep/a['texture_path']
    assert tex.is_file() and sha(tex)==a['source_sha256']
    binding=next(x for x in m['runtime_bindings'] if x['asset_id']==a['asset_id'])
    assert tex.read_bytes()==(ROOT/binding['runtime_texture']).read_bytes()
    assert a['polar_cap_composite']['top_photo']==binding['top_cap_photo']
    assert a['polar_cap_composite']['bottom_photo']==binding['bottom_cap_photo']
surface=(ROOT/'stage88/stage_direct_photo_surface.py').read_text('utf8')
assert 'USER_ORIGINAL_17_SIDE_TOP_BOTTOM_POLAR_COMPOSITE' in surface
print('REAL17_ORIGINAL_17_BYTE_MATCH_PASS')
print('REAL17_SIDE_PLUS_TOP_BOTTOM_POLAR_COMPOSITE_11_PASS')
print('ALL_SIX_TOP_BOTTOM_SOURCES_USED',dict(top_counts),dict(bottom_counts))
print('REAL17_RUNTIME_TEXTURE_BINDINGS_8_PASS')
