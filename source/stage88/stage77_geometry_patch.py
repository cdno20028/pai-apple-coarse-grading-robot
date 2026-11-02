# -*- coding: utf-8 -*-
"""Stage77 v2 minimum visual-only patch.
The pre-existing Stage76 continuous physics, camera positions, controller,
AI grading, thresholds, photos, textures, occupancy remain unchanged.
"""
from pathlib import Path
import hashlib,json,shutil

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def apply_stage77_geometry_patch(engine,results):
    engine=Path(engine);results=Path(results)
    source=engine/'apple_stage13'/'src'/'cushion_texture_assets.py'
    helper=engine/'apple_stage13'/'src'/'apple_stage77_mesh.py'
    original=source.read_text(encoding='utf8')
    original_sha=sha(source)
    hook='\ndef ensure_assets():\n'
    old_build="    build_uv_sphere(ASSETS/'apple_uv.obj')"
    old_visual="fileName=str(ASSETS/'apple_uv.obj'),"
    assert original.count(hook)==1,'Stage77 v2 missing ensure_assets hook'
    assert original.count(old_build)==1,'Stage77 v2 unexpected apple asset generation'
    assert original.count(old_visual)==1,'Stage77 v2 unexpected apple visual mesh source'
    assert 'from apple_stage77_mesh import' not in original,'Stage77 v2 duplicate helper'
    import_hook='\n# Stage77 v2: variable small offsets of the *visual* fruit pole wells.\nfrom apple_stage77_mesh import build_uv_sphere, build_offset_variants, choose_visual_mesh\n'
    revised=original.replace(hook,import_hook+hook,1)
    revised=revised.replace(old_build,old_build+'\n    build_offset_variants(ASSETS)',1)
    revised=revised.replace(old_visual,'fileName=str(choose_visual_mesh(ASSETS)),',1)
    compile(revised,str(source),'exec')
    source.write_text(revised,encoding='utf8')
    shutil.copyfile(Path(__file__).resolve().parent/'apple_stage77_mesh.py',helper)
    data={'stage':'77v2','date_jst':'2026/10/22',
          'source_sha_before':original_sha,'source_sha_after':sha(source),
          'generator_sha256':sha(helper),'variable_visual_pole_offsets':True,
          'collision_GEOM_SPHERE_unchanged':True,
          'photos_uv_and_ai_not_changed':True,
          'grading_thresholds_and_controller_unchanged':True,
          'actual_PyBullet_test_unperformed_in_preflight':True}
    (results/'stage77_v2_geometry_patch.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf8')
    return data
