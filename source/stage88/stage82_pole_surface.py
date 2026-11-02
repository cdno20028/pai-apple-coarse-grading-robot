# -*- coding: utf-8 -*-
"""Stage82: retain per-apple original UV chroma and damage; add only very local
achromatic NORMAL-pole micro-detail. Six reference photos are independent
unpaired examples, never true per-fruit 360-degree evidence or classifier truth.
The head/tail geometry and original source photos are not changed here.
"""
import hashlib,json,math
from pathlib import Path
import numpy as np, cv2
from PIL import Image
from stage68_surface import Stage68Surface
from stage81_pole_surface import CENTERS,REFERENCE_RADIUS

INNER_ANGLE_RAD=.14  # ~23 UV rows
OUTER_ANGLE_RAD=.27  # ~44 UV rows, <16 degrees from the pole
DETAIL_GAIN=.26
MAX_DELTA_LUMA=11.0

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pole_detail_preserving_chroma(original,head,tail,head_name,tail_name):
    """No colour copy from another fruit; use only weak high-frequency relief.

    The source is Stage68's per-fruit repaired UV, not Stage81's wide 67-degree
    transplant. Outside the very narrow polar caps, RGB pixels are identical.
    There is deliberately NO reference grade, asset ID, or classification input.
    """
    src=np.asarray(original.convert('RGB'),dtype=np.uint8)
    if src.shape!=(512,1024,3):raise ValueError(f'Unexpected UV dimensions {src.shape}')
    h,w=src.shape[:2];out=src.astype(np.float32).copy()
    xangles=2*np.pi*(np.arange(w,dtype=np.float32)+.5)/w
    report={}
    for tag,ref_img,name,top in [('head',head,head_name,True),('tail',tail,tail_name,False)]:
        ref=np.asarray(ref_img.convert('RGB'),dtype=np.uint8)
        cx,cy=CENTERS[name]; radius=REFERENCE_RADIUS[name]
        # Exclusive cap angle; no sampling or modification of broad lateral regions
        nrows=int(math.ceil(OUTER_ANGLE_RAD*h/math.pi))+1
        rows=np.arange(nrows) if top else np.arange(h-nrows,h)
        theta=(rows+.5)*math.pi/h if top else (h-rows-.5)*math.pi/h
        rs=np.sin(theta)*radius
        mapx=(cx+rs[:,None]*np.cos(xangles)[None,:]).astype(np.float32)
        mapy=(cy+rs[:,None]*np.sin(xangles)[None,:]).astype(np.float32)
        sample=cv2.remap(ref,mapx,mapy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
        gray=cv2.cvtColor(sample,cv2.COLOR_RGB2GRAY).astype(np.float32)
        # Subtract the very-low-frequency reference shading and lighting.
        low=cv2.GaussianBlur(gray,(0,0),sigmaX=15,sigmaY=9)
        high=np.clip(gray-low,-MAX_DELTA_LUMA,MAX_DELTA_LUMA)
        # Polar singularity cannot be treated as a physical radial texture feature.
        ramp=np.clip(theta/.055,0,1)
        fade=np.clip((OUTER_ANGLE_RAD-theta)/(OUTER_ANGLE_RAD-INNER_ANGLE_RAD),0,1)
        fade=fade*fade*(3-2*fade)
        fade=np.where(theta<=INNER_ANGLE_RAD,1.0,fade)*ramp
        out[rows]=np.clip(out[rows]+(high*DETAIL_GAIN*fade[:,None])[:,:,None],0,255)
        report[tag]={'reference':name,'reference_role':'NORMAL_ANATOMY_ACHROMATIC_MICRODETAIL_ONLY',
            'outer_influence_angle_deg':round(math.degrees(OUTER_ANGLE_RAD),3),
            'inner_detail_angle_deg':round(math.degrees(INNER_ANGLE_RAD),3),
            'max_luminance_deviation_per_channel':round(MAX_DELTA_LUMA*DETAIL_GAIN,3),
            'source_reference_rgb_not_transplanted':True}
    arr=np.uint8(np.rint(out))
    # Same Stage81 brown palette swatch for 3D stem and sepals; 
    # the reserved pixels are away from legitimate photo-derived areas.
    arr[251:259,1008:1023]=[79,49,31]
    arr[265:273,1008:1023]=[62,43,29]
    unchanged=np.ones((512,1024),dtype=bool)
    unchanged[:int(math.ceil(OUTER_ANGLE_RAD*h/math.pi))+1,:]=False
    unchanged[h-int(math.ceil(OUTER_ANGLE_RAD*h/math.pi))-1:,:]=False
    unchanged[251:273,1008:1023]=False
    assert np.array_equal(arr[unchanged],src[unchanged]),'Side photo pixels must be identical'
    return Image.fromarray(arr,'RGB'),report

class Stage82Surface(Stage68Surface):
    def __init__(self,assets,output,refs):
        super().__init__(assets,output)
        self.refs=Path(refs);self.out82=Path(output)/'stage82_pole_color_preserved'
        self.out82.mkdir(parents=True,exist_ok=True)
        manifest=json.loads((Path(assets)/'manifest.json').read_text(encoding='utf8'))
        self.items=[]
        for i,rec in enumerate(manifest['assets']):
            aid=rec['asset_id']
            hn=f'head-{i%3+1}.jpg';tn=f'tail-{(i+1)%3+1}.jpg'
            for n in (hn,tn):
                if not (self.refs/n).is_file():raise FileNotFoundError(self.refs/n)
            base=Path(super().path(aid))
            im,shape_info=pole_detail_preserving_chroma(
               Image.open(base),Image.open(self.refs/hn),Image.open(self.refs/tn),hn,tn)
            target=self.path(aid);im.save(target,compress_level=6)
            self.items.append({'asset_id':aid,'base_sha256':sha(base),'result_sha256':sha(target),
                'source_uv_sha256':sha(Path(assets)/rec['texture_path']),
                'source_photo_sha256':rec['source_sha256'],
                'head_reference_sha256':sha(self.refs/hn),'tail_reference_sha256':sha(self.refs/tn),
                'reference_grade_excluded_from_generation':True,'shape_info':shape_info})
        (Path(output)/'stage82_color_provenance.json').write_text(json.dumps({
            'stage':82,'date_jst':'2026/10/23','visual_geometry_from_stage81':True,
            'head_tail_refs_only_for_normal_anatomy':True,
            'source_images_unmodified':True,
            'head_tail_normal_samples_NOT_paired_with_8_fruit_photos':True,
            'NOT_360_photogrammetry_or_orchard_accuracy':True,
            'items':self.items},ensure_ascii=False,indent=2),encoding='utf8')
    def path(self,aid):return self.out82/(aid+'_narrow_normal_poles.png')
