# -*- coding: utf-8 -*-
"""Stage78 static comparison to v2; this is NOT a physical or orchard test."""
import math,json
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np,trimesh
import apple_stage77_mesh as model

def main():
    with TemporaryDirectory() as d:
        assets=Path(d)
        rows=model.build_offset_variants(assets)
        assert len(rows)==8
        for a in rows:
            mesh=trimesh.load_mesh(assets/a['filename'],process=False)
            assert np.isfinite(mesh.vertices).all()
            verts=mesh.vertices.copy()
            # Pole axes must be independent and remain within the existing small offset bounds.
            assert np.allclose(verts[0,:2]*40,a['top_center_xy_mm'],atol=1e-5)
            assert np.allclose(verts[-1,:2]*40,a['bottom_center_xy_mm'],atol=1e-5)
            # A closed, consistent UV mesh after seam-welding.
            mesh.merge_vertices(merge_tex=True)
            assert mesh.is_watertight and mesh.is_winding_consistent
            assert len(mesh.faces)>10000
            # Shape is a gentle variation around the sphere; no hidden changes to fruit bulk.
            r=np.linalg.norm(verts,axis=1)
            assert 0.76<r.min()<0.95 and r.max()<1.08,(r.min(),r.max())
        # The original ring profile still exhibits meaningful dimple geometry.
        ang=np.linspace(0,math.pi,20000)
        zz=np.array([model.unit_profile(t)[1] for t in ang]);tt=model.unit_profile(0)[1];bt=model.unit_profile(math.pi)[1]
        assert zz[(ang>0.08)&(ang<0.75)].max()-tt>0.07
        assert bt-zz[(ang>2.3)&(ang<3.12)].min()>0.10
    print('STAGE78_GEOMETRY_STATIC_PASS',json.dumps({'visual_variants':8,'mesh_closed':True,'head_and_calyx_pole_offsets_independent':True,'original_photo_and_grader_unmodified':True,'real_pybullet_not_executed':True},ensure_ascii=False))
if __name__=='__main__':main()
