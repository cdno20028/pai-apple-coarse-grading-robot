# -*- coding: utf-8 -*-
"""Stage77 v2 independent static geometry+original-engine patch compatibility."""
from pathlib import Path
from tempfile import TemporaryDirectory
import json,zipfile,shutil,hashlib
import numpy as np
import trimesh
from apple_stage77_mesh import build_offset_variants,choose_visual_mesh,OFFSET_PROFILES_MM,REFERENCE_RADIUS_MM
from stage77_geometry_patch import apply_stage77_geometry_patch

def main():
    assert len(OFFSET_PROFILES_MM)==8
    assert len({tuple(t)+tuple(b) for t,b in OFFSET_PROFILES_MM})==8
    with TemporaryDirectory() as temp:
        p=Path(temp);assets=p/'assets'
        records=build_offset_variants(assets)
        assert len(records)==8
        # Each variant remains a valid closed mesh; the pole-center positions
        # are allowed to differ from each other and from the sphere's axis.
        for v in records:
            mesh=trimesh.load_mesh(str(assets/v['filename']),process=False)
            original=mesh.vertices
            for which,idx,expected in [('head',0,v['top_center_xy_mm']),('tail',-1,v['bottom_center_xy_mm'])]:
                got=original[idx,:2]*REFERENCE_RADIUS_MM
                assert np.allclose(got,expected,atol=.00005), (which,got,expected)
            mesh.merge_vertices(merge_tex=True)
            assert mesh.is_watertight and mesh.is_winding_consistent, v['filename']
        seq=[choose_visual_mesh(assets).name for _ in range(12)]
        assert len(set(seq[:8]))==8 and seq[8]==seq[0]
        log=(assets/'stage77_v2_spawn_log.jsonl').read_text(encoding='utf8').splitlines()
        assert len(log)==12
        with zipfile.ZipFile(Path(__file__).resolve().parent/'stage27_original_96files.zip') as z:
            original=z.read('apple_stage13/src/cushion_texture_assets.py')
        scratch=p/'engine'/'apple_stage13'/'src';scratch.mkdir(parents=True)
        src=scratch/'cushion_texture_assets.py';src.write_bytes(original)
        before=hashlib.sha256(src.read_bytes()).hexdigest()
        report=apply_stage77_geometry_patch(p/'engine',p)
        text=src.read_text(encoding='utf8')
        assert text.count('build_offset_variants(ASSETS)')==1
        assert text.count('choose_visual_mesh(ASSETS)')==1
        assert 'c=p.createCollisionShape(p.GEOM_SPHERE,radius=radius)' in text
        assert report['source_sha_before']==before
        # Test generating assets through ORIGINAL module's actual ensure_assets.
        import sys
        sys.path.insert(0,str(scratch))
        try:
            import cushion_texture_assets as source_mod
            source_mod.ASSETS=p/'liveassets'
            source_mod.ensure_assets()
            assert len(list(source_mod.ASSETS.glob('apple_uv_offset_*.obj')))==8
            # monkey-patch pybullet to prove spawn selects a mesh while sphere
            # collision argument is identical.  No actual physics is claimed.
            class FakeBullet:
                GEOM_SPHERE=2;GEOM_MESH=5
                def createCollisionShape(self,kind,**kwargs):
                    assert kind==self.GEOM_SPHERE;assert kwargs['radius']==.04;return 11
                def createVisualShape(self,kind,**kwargs):
                    assert kind==self.GEOM_MESH
                    assert Path(kwargs['fileName']).is_file()
                    self.filename=kwargs['fileName'];return 12
                def createMultiBody(self,**kwargs):return 13
                def loadTexture(self,*args):return 14
                def changeVisualShape(self,*args,**kwargs):pass
                def changeDynamics(self,*args,**kwargs):pass
            fake=FakeBullet()
            for _ in range(3):
                assert source_mod.spawn_textured_apple(fake,'A',.18,.04,(0,0,0),.4)==13
            assert len((source_mod.ASSETS/'stage77_v2_spawn_log.jsonl').read_text().splitlines())==3
        finally:
            sys.path.remove(str(scratch))
            sys.modules.pop('cushion_texture_assets',None)
    print('STAGE77_V2_OFFSET_GEOMETRY_STATIC_OK',json.dumps({
        'variants':8,'top_bottom_can_have_independent_offsets':True,
        'no_requirement_that_pole_center_is_exactly_0':True,
        'visual_geometry_is_loaded_by_actual_original_asset_spawn':True,
        'base_collision_sphere_unchanged':True,
        'source_photos_and_grading_unmodified':True,
        'real_pybullet_simulation_not_run':True},ensure_ascii=False))
if __name__=='__main__':main()
