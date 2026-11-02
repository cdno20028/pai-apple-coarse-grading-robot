# -*- coding: utf-8 -*-
"""Stage77 visual double-dimple geometry offline preflight (NOT PyBullet)."""
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import trimesh
from apple_stage77_mesh import build_uv_sphere,unit_profile

def main():
    with TemporaryDirectory() as d:
        path=build_uv_sphere(Path(d)/'double_recess_apple.obj')
        mesh=trimesh.load_mesh(str(path),process=False)
        mesh.merge_vertices(merge_tex=True)  # weld visual UV seam only for mesh topology audit
        assert mesh.is_watertight and mesh.is_winding_consistent, 'OBJ topology invalid'
        angles=np.linspace(0,np.pi,30001)
        h=np.array([unit_profile(t)[1] for t in angles])
        ztop=unit_profile(0.)[1];zbottom=unit_profile(np.pi)[1]
        topdepth=float(h[(angles>.08)&(angles<.75)].max()-ztop)
        bottomrim=float(h[(angles>2.3)&(angles<3.12)].min())
        botdepth=float(zbottom-bottomrim)
        assert topdepth>.075 and botdepth>.1 and bottomrim>=-1.005
        original=Path(__file__).resolve().parent/'stage27_original_96files.zip'
        import zipfile
        with zipfile.ZipFile(original) as z:
            text=z.read('apple_stage13/src/cushion_texture_assets.py').decode('utf8')
        assert 'createCollisionShape(p.GEOM_SPHERE' in text, 'baseline shape changed unexpectedly'
        print('STAGE77_DOUBLE_DIMPLE_MESH_PREFLIGHT_OK',{
            'UV_mesh_watertight_after_seam_weld':True,
            'mesh_vertices_after_seam_weld':len(mesh.vertices),
            'triangle_faces':len(mesh.faces),
            'assumed_radius_m':.04,
            'top_dimple_depth_mm_at_40mm_r':round(topdepth*40,3),
            'bottom_dimple_depth_mm_at_40mm_r':round(botdepth*40,3),
            'collision_sphere_still_used':True,
            'actual_PyBullet_camera_test_unperformed':True})
if __name__=='__main__':main()
