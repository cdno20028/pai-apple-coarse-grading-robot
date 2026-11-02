# -*- coding: utf-8 -*-
"""Apply exact 17-photo side/top/bottom composite UVs at Stage84 runtime.
Top and bottom source photos are unpaired views; manifest records their use.
"""
from pathlib import Path
import hashlib, json

def _sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

class StageDirectPhotoSurface:
    def __init__(self,assets,results,refs=None):
        self.assets=Path(assets); self.results=Path(results)
        m=json.loads((self.assets/'manifest.json').read_text(encoding='utf-8'))
        self._paths={}; evidence=[]
        for entry in m['assets']:
            aid=entry['asset_id']; texture=self.assets/'uv_photo_front'/(aid+'_uv.png')
            if not texture.is_file(): raise FileNotFoundError(texture)
            self._paths[aid]=texture; cap=entry.get('polar_cap_composite',{})
            evidence.append({'asset_id':aid,'texture':str(texture),'sha256':_sha(texture),
              'method':'USER_SIDE_PHOTO_PLUS_UNPAIRED_TOP_BOTTOM_POLAR_CAPS',
              'side_photo':entry.get('direct_user_photo'),
              'top_cap_photo':cap.get('top_photo'),'bottom_cap_photo':cap.get('bottom_photo'),
              'top_bottom_same_individual_confirmed':False,'global_colour_correction':False})
        self.results.mkdir(parents=True,exist_ok=True)
        (self.results/'DIRECT_USER_PHOTO_RUNTIME_TEXTURES.json').write_text(
          json.dumps({'method':'USER_ORIGINAL_17_SIDE_TOP_BOTTOM_POLAR_COMPOSITE','entries':evidence,
            'all_six_top_bottom_sources_used_in_composite':True,
            'top_bottom_photos_are_unpaired_views':True,
            'actual_3D_render_requires_PyBullet_execution':True},ensure_ascii=False,indent=2),encoding='utf-8')
    def path(self,asset_id): return self._paths[asset_id]
