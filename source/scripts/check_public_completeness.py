from pathlib import Path
import zipfile,json,hashlib,sys,ast
ROOT=Path(__file__).resolve().parents[1];S=ROOT/'stage88'
manifest=json.loads((S/'stage58_apple_photo_assets/prepared/manifest.json').read_text())
assert len(manifest['assets'])==8
for item in manifest['assets']:
 p=S/'stage58_apple_photo_assets/prepared'/item['texture_path'];assert p.is_file() and p.stat().st_size>1000
 assert hashlib.sha256(p.read_bytes()).hexdigest()==item['source_sha256']
for end in ('head','tail'):
 for n in range(1,4):assert (S/f'stage79_normal_pole_reference/{end}-{n}.jpg').is_file()
assert (S/'stage85_scar_appearance_REFERENCE_NOT_AI_EVIDENCE.png').is_file()
with zipfile.ZipFile(S/'stage27_original_96files.zip') as z:
 assert z.testzip() is None
 for grade in 'ABCD':assert f'apple_stage13/assets/stage13/apple_{grade}.png' in z.namelist()
assert (ROOT/'notebooks/PAI_Public_Synthetic_Stage88_Colab.ipynb').is_file()
for p in sorted(S.glob('*.py')):ast.parse(p.read_text('utf8'))
assert len(list((S/'stage67_public_camera_calibration').glob('*.png')))==21
print('PUBLIC_COMPLETENESS_STATIC_PASS; PyBullet physics runtime NOT tested')
