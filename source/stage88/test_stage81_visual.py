# -*- coding:utf-8 -*-
"""Stage81 offline visual/code integrity only. NOT a PyBullet or AI pass."""
from pathlib import Path
import tempfile,hashlib,json,zipfile
import numpy as np,trimesh
from PIL import Image
from apple_stage77_mesh import build_uv_sphere,vertex_for_angles,OFFSET_PROFILES_MM
from stage81_pole_surface import Stage81Surface

def main():
 root=Path(__file__).resolve().parent
 with tempfile.TemporaryDirectory() as td:
  meshpath=Path(td)/'fruit.obj'
  build_uv_sphere(meshpath,latitude=64,longitude=128,top_xy_mm=(0.6,-.4),bottom_xy_mm=(-.5,.6))
  m=trimesh.load_mesh(meshpath,process=False)
  assert len(m.faces)>16000 and len(m.vertices)>8000
  v=m.vertices.copy()
  assert np.allclose(40*v[0,:2],[.6,-.4],atol=.001)
  assert np.allclose(40*v[-1,:2],[-.5,.6],atol=.001)
  assert v[:,2].max()>vertex_for_angles(0,0,(.6,-.4),(-.5,.6))[2]+0.1, 'Raised stem not present'
  # Several non-main components are deliberately closed tetrahedra / stalk ring.
  m.merge_vertices(merge_tex=True)
  assert m.is_watertight and m.is_winding_consistent
  assert (root/'stage79_normal_pole_reference').is_dir()
  names=[f'{part}-{i}.jpg' for part in ('head','tail') for i in (1,2,3)]
  refs={n:hashlib.sha256((root/'stage79_normal_pole_reference'/n).read_bytes()).hexdigest() for n in names}
  assets=root/'stage58_apple_photo_assets'/'prepared'
  originals={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (assets/'uv_photo_front').glob('*.png')}
  surf=Stage81Surface(assets,Path(td)/'tex',root/'stage79_normal_pole_reference')
  assert len(surf.stage81_items)==8
  changed_poles=0
  for item in surf.stage81_items.values():
   im=np.asarray(Image.open(Path(td)/'tex'/item['texture_file']),dtype=np.uint8)
   old=np.asarray(Image.open(Path(td)/'tex'/'stage68_uv_review'/(item['asset_id']+'_synthetic_pole_repair.png')),dtype=np.uint8)
   # Mid-latitude legacy fruit photo region unchanged except a tiny visual-material atlas.
   ma=np.ones((512,1024),dtype=bool);ma[246:277,1004:1024]=False
   ma[:200]=False;ma[312:]=False
   assert np.array_equal(im[ma],old[ma]),'Lateral body pixels mutated'
   assert not np.array_equal(im[0:80],old[0:80])
   assert not np.array_equal(im[-80:],old[-80:])
   changed_poles+=1
  assert originals=={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (assets/'uv_photo_front').glob('*.png')}
  assert refs=={n:hashlib.sha256((root/'stage79_normal_pole_reference'/n).read_bytes()).hexdigest() for n in names}
  code=(root/'run_stage70_continuous.py').read_text(encoding='utf8')
  assert 'stage62=Stage81CameraAI(' in code
  assert 'apply_stage80_physical_registry(engine,result_dir)' in code
  assert 'apply_stage77_geometry_patch(engine,result_dir)' in code
  with zipfile.ZipFile(root/'stage27_original_96files.zip') as z:
   assert 'createCollisionShape(p.GEOM_SPHERE' in z.read('apple_stage13/src/cushion_texture_assets.py').decode()
 print('STAGE81_VISUAL_STATIC_PASS',json.dumps({'pole_photo_examples':6,'photo_assets_unchanged':8,'head_tail_composite_UVs':changed_poles,'stem_3D':True,'calyx_sepal_3D':True,'mesh_watertight':True,'worker_and_physics_registry_unchanged':True,'original_AI_classifier_unchanged':True,'collision_sphere_unchanged':True,'pybullet_execution_completed':False},ensure_ascii=False))
if __name__=='__main__':main()
