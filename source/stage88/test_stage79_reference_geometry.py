# -*- coding: utf-8 -*-
"""Stage79 reference-grounded shape audit; no PyBullet or real orchard validation."""
from pathlib import Path
from tempfile import TemporaryDirectory
import math, json, numpy as np, trimesh
import apple_stage77_mesh as m

def main():
    root=Path(__file__).resolve().parent
    refs=list((root/'stage79_normal_pole_reference').glob('*.jpg'))
    assert len(refs)==6
    assert len({p.name for p in refs})==6
    with TemporaryDirectory() as d:
        variants=m.build_offset_variants(Path(d))
        assert len(variants)==8
        for rec in variants:
            mesh=trimesh.load_mesh(str(Path(d)/rec['filename']),process=False)
            assert np.isfinite(mesh.vertices).all()
            v=mesh.vertices.copy()
            assert np.allclose(40*v[0,:2],rec['top_center_xy_mm'],atol=1e-4)
            assert np.allclose(40*v[-1,:2],rec['bottom_center_xy_mm'],atol=1e-4)
            mesh.merge_vertices(merge_tex=True)
            assert mesh.is_watertight and mesh.is_winding_consistent
        # A real smooth depression MUST have a horizontal tangent at the pole
        # and negligible five-lobed displacement at theta=0 and theta=pi.
        for alpha in (0,math.pi):
            z0=m.vertex_for_angles(alpha,0,(0,0),(0,0))[2]
            near=alpha+0.002 if alpha==0 else alpha-0.002
            zs=[m.vertex_for_angles(near,phi,(0,0),(0,0))[2] for phi in np.linspace(0,2*math.pi,32)]
            assert max(abs(z-z0) for z in zs)<0.001
            assert max(zs)-min(zs)<0.0002
        theta=np.linspace(0,math.pi,30001)
        values=np.array([m.unit_profile(float(t))[1] for t in theta])
        head=float(np.max(values[(theta>.10)&(theta<.75)])-values[0])*40
        calyx=float(values[-1]-np.min(values[(theta>2.40)&(theta<3.10)]))*40
        assert 2.5<head<7.5,(head,calyx)
        assert 3.0<calyx<8.0,(head,calyx)
        # Different surfaces have different profiles, but same semantics across grades
        assert abs(head-calyx)>0.1
        assert len(list(Path(d).glob('apple_uv_offset_*.obj')))==8
    print('STAGE79_REFERENCED_HEAD_TAIL_STATIC_PASS',json.dumps({'head_well_mm':round(head,2),
         'tail_well_mm':round(calyx,2),'variants':8,'photos_for_reference':6,
         'center_offset_not_absolute':True,'photos_used_as_AI_input':False,
         'collision_shape_modified':False,'pybullet_not_run':True},ensure_ascii=False))
if __name__=='__main__':main()
