from pathlib import Path
import json, hashlib, sys
root=Path(__file__).resolve().parents[1]
data=json.loads((root/'public_ai_apple_grade/active_asset_hashes.json').read_text(encoding='utf8'))
for item in data['files']:
 p=root/item['file']
 if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=item['sha256']:
  raise SystemExit('PUBLIC_AI_ASSET_INTEGRITY_FAIL: '+str(p))
print('PUBLIC_AI_ASSET_INTEGRITY_OK',len(data['files']))

for grade in 'ABCD':
 for view in ('head','side','tail'):
  assert (root/f'public_ai_apple_grade/multiview/{grade}_{view}_AI.png').is_file()
for aid in ['01_A','02_B','03_B','04_C','05_C','06_C','07_C','08_D']:
 for view in ('head','tail'):
  assert (root/f'stage88/stage79_normal_pole_reference/{view}-{aid}.jpg').is_file()
print('AI_MULTIVIEW_HEAD_SIDE_TAIL_INTEGRATION_STATIC_PASS')
