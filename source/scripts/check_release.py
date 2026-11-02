#!/usr/bin/env python3
"""Static and camera mock checks; never presents as PyBullet execution."""
import json,hashlib,pathlib,subprocess,sys,ast
root=pathlib.Path(__file__).resolve().parent.parent
manifest=json.loads((root/'audit'/'PUBLIC_FILE_SHA256.json').read_text(encoding='utf8'))
for name,h in manifest['files'].items():
    p=root/name
    if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=h:
        raise AssertionError(f'FILE_MISSING_OR_MODIFIED: {name}')
print('VERIFIED_EXACT_SOURCE_FILES',len(manifest['files']),flush=True)
ps=sorted((root/'stage88').rglob('*.py'))+sorted((root/'stage27').rglob('*.py'))
for p in ps:ast.parse(p.read_text(encoding='utf8'),filename=str(p))
print('PYTHON_AST_OK',len(ps),flush=True)
for name in ['test_stage88_underplate_camera.py','test_stage87_gripper_camera_static.py',
             'test_stage70_nonapple_gate.py','test_stage80_registry.py']:
    p=subprocess.run([sys.executable,name],cwd=root/'stage88',stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    print(name,'PASS' if p.returncode==0 else 'FAIL',p.stdout[-1200:],flush=True)
    if p.returncode:raise RuntimeError(f'CHECK_FAILED {name}')
print('OFFLINE_STATIC_VERIFIED — NO NEW PYBULLET RUN EXECUTED',flush=True)
