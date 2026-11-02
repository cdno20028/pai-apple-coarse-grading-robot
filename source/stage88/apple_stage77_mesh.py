# -*- coding: utf-8 -*-
"""Stage81 head/tail organic bowl plus raised stem and low-profile calyx.
The original Stage79 reference design is preserved as a base profile.

Restore a plausible softly scalloped shoulder at each depression, retaining
Stage77 v2 small independent pole offsets. Neither marks nor colour defects
are drawn onto existing user-derived textures. Original grading is untouched.
Collision remains GEOM_SPHERE; real PyBullet validation pending.
2026/10/22 JST.
"""
import math,json
from pathlib import Path

# Coordinates are in millimetres for a NOMINAL 40 mm radius fruit.
# These are explicit demonstration hypotheses, not measured apple statistics.
# Grade-neutral: variation is selected by chronological visual spawn sequence,
# without reference to class label or photo identity.
REFERENCE_RADIUS_MM=40.0
OFFSET_PROFILES_MM=(
    ((0.0,0.0),(0.0,0.0)),
    ((0.6,-0.4),(-0.5,0.6)),
    ((-0.8,0.5),(0.8,-0.3)),
    ((1.0,0.2),(-0.6,-0.5)),
    ((-0.5,-0.7),(0.4,0.5)),
    ((0.2,1.0),(-0.9,0.3)),
    ((-0.9,-0.2),(0.7,0.8)),
    ((0.5,0.8),(-0.3,-0.7)),
)
# Stage81: head (stem cavity) and tail (calyx cavity) are NOT identical.
# Visual parameters are illustrative, not digitized measurements from six JPGs.
# 6 reference photos show front-on NORMAL examples, not paired sides of one fruit.
# r=1 corresponds to 40mm nominal fruit radius; all offsets are grade-neutral.
TOP_DEPTH_R=0.14
TOP_SIGMA_RAD=0.22
TOP_RING_R=0.027
TOP_RING_THETA_RAD=0.285
TOP_RING_SIGMA_RAD=0.18
BOTTOM_CENTER_LIFT_R=0.17
BOTTOM_SIGMA_RAD=0.24
BOTTOM_RING_R=0.030
BOTTOM_RING_THETA_RAD=0.29
BOTTOM_RING_SIGMA_RAD=0.18
_visual_serial=0

# Pole-center derivative must vanish: both centers are smooth bowls, NEVER cones.
# A little five-fold calyx anatomy lives NEAR the bottom inner well, not at the pole.
# Head shoulder may be subtly asymmetric as in head-1,2,3; do not impose
# axis centering. The offset profiles below apply independently to head/tail.
TOP_LOBE_AMPLITUDE_R=0.010
BOTTOM_LOBE_AMPLITUDE_R=0.018
TOP_LOBE_HEIGHT_R=0.0035
BOTTOM_LOBE_HEIGHT_R=0.010

def _lip(theta,location,width):
    # multiply by t^2 to give d(lip)/d(theta)=0 at a pole
    return (theta/max(location,1e-6))**2*math.exp(-((theta-location)/width)**2)

def unit_profile(theta):
    # The normal head opening is a shallow bowl with a raised shoulder.
    # The normal bottom is a more structured calyx well, with a central floor.
    r=math.sin(theta)
    tail=math.pi-theta
    top=-TOP_DEPTH_R*math.exp(-(theta/TOP_SIGMA_RAD)**2) \
        +TOP_RING_R*_lip(theta,TOP_RING_THETA_RAD,TOP_RING_SIGMA_RAD)
    bottom=+BOTTOM_CENTER_LIFT_R*math.exp(-(tail/BOTTOM_SIGMA_RAD)**2) \
        -BOTTOM_RING_R*_lip(tail,BOTTOM_RING_THETA_RAD,BOTTOM_RING_SIGMA_RAD)
    return r,math.cos(theta)+top+bottom

def vertex_for_angles(theta,phi,top_xy_mm,bottom_xy_mm):
    r,z=unit_profile(theta)
    tail=math.pi-theta
    # Softly scalloped stem shoulder and five-lobed calyx, not 5 damage spots.
    # Both angular terms vanish exactly at the poles so no cone-like polar spike.
    top_env=_lip(theta,0.37,0.16)
    bot_env=_lip(tail,0.21,0.12)
    top_lobes=math.cos(5*phi+0.18)
    bot_lobes=math.cos(5*phi+0.63)
    r*=1.0+TOP_LOBE_AMPLITUDE_R*top_env*top_lobes \
          +BOTTOM_LOBE_AMPLITUDE_R*bot_env*bot_lobes
    z+=TOP_LOBE_HEIGHT_R*top_env*top_lobes \
       -BOTTOM_LOBE_HEIGHT_R*bot_env*bot_lobes
    # Small independent XY pole offsets (including zero) are retained.
    top_weight=math.exp(-(theta/0.49)**4)
    bottom_weight=math.exp(-(tail/0.49)**4)
    dx=(top_xy_mm[0]*top_weight+bottom_xy_mm[0]*bottom_weight)/REFERENCE_RADIUS_MM
    dy=(top_xy_mm[1]*top_weight+bottom_xy_mm[1]*bottom_weight)/REFERENCE_RADIUS_MM
    return (r*math.cos(phi)+dx,r*math.sin(phi)+dy,z)

def build_uv_sphere(path,latitude=64,longitude=128,
                    top_xy_mm=(0.,0.),bottom_xy_mm=(0.,0.)):
    latitude=int(latitude);longitude=int(longitude)
    if latitude<32 or longitude<64:
        raise ValueError('At least 32 latitude and 64 longitude steps are required')
    if any(abs(float(x))>2.0 for x in (*top_xy_mm,*bottom_xy_mm)):
        raise ValueError('Offset >2mm not included in this demonstrated range')
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    vertices=[];uvs=[];faces=[]
    def add(x,y,z,u,v):
        vertices.append((x,y,z));uvs.append((u,v));return len(vertices)
    add(*vertex_for_angles(0.,0.,top_xy_mm,bottom_xy_mm),0.5,1.)
    for i in range(1,latitude):
        theta=math.pi*i/latitude
        for j in range(longitude+1):
            a=2*math.pi*j/longitude
            add(*vertex_for_angles(theta,a,top_xy_mm,bottom_xy_mm),j/longitude,1-i/latitude)
    # Stage88 REAL17: photographic head already contains its bent stalk.
    # Do not construct a duplicate 3D stalk. Keep original physical collision.
    # Calyx sepals are shallow textured 3D ridges, not damage textures.
    bx,by=[float(v)/REFERENCE_RADIUS_MM for v in bottom_xy_mm]
    basez=unit_profile(math.pi)[1]
    sepal_faces=[]
    for k in range(5):
        phi=2*math.pi*k/5.+0.28
        nx,ny=math.cos(phi),math.sin(phi)
        tx0,ty0=-ny,nx
        # Each closed tetrahedron is less than one millimetre thick.
        a=add(bx+.014*tx0,by+.014*ty0,basez-.009,0.991,0.476)
        b=add(bx-.014*tx0,by-.014*ty0,basez-.009,0.991,0.476)
        c=add(bx+.155*nx,by+.155*ny,unit_profile(math.pi-.16)[1]-.009,0.991,0.476)
        d=add(bx+.048*nx,by+.048*ny,basez+.010,0.991,0.476)
        sepal_faces.extend(((a,b,c),(a,d,b),(b,d,c),(c,d,a)))
    bottom_index=add(*vertex_for_angles(math.pi,0.,top_xy_mm,bottom_xy_mm),0.5,0.)
    first=2
    for j in range(longitude):faces.append((1,first+j,first+j+1))
    for row in range(latitude-2):
        a=first+row*(longitude+1);b=a+longitude+1
        for j in range(longitude):
            faces.append((a+j,b+j,a+j+1));faces.append((a+j+1,b+j,b+j+1))
    last=first+(latitude-2)*(longitude+1)
    for j in range(longitude):faces.append((last+j,bottom_index,last+j+1))
    faces.extend(sepal_faces)
    lines=['# Stage79 distinct smooth stem well / calyx well; photographs only reference',
           '# Collision uses original Stage76 physical sphere, not this mesh.']
    lines.extend('v %.9f %.9f %.9f'%v for v in vertices)
    lines.extend('vt %.9f %.9f'%v for v in uvs)
    lines.extend('f %d/%d %d/%d %d/%d'%(a,a,b,b,c,c) for a,b,c in faces)
    path.write_text('\n'.join(lines)+'\n',encoding='ascii')
    return path

def build_offset_variants(assets):
    assets=Path(assets);assets.mkdir(parents=True,exist_ok=True)
    info=[]
    for i,(top,bottom) in enumerate(OFFSET_PROFILES_MM):
        name=f'apple_uv_offset_{i:02}.obj'
        build_uv_sphere(assets/name,top_xy_mm=top,bottom_xy_mm=bottom)
        info.append({'variant':i,'filename':name,'top_center_xy_mm':top,'bottom_center_xy_mm':bottom})
    (assets/'stage77_v2_offset_profiles.json').write_text(json.dumps({
        'scope':'VISUAL_MESH_ONLY','nominal_apple_radius_mm':REFERENCE_RADIUS_MM,
        'offsets_are_trial_assumptions_not_measured':True,
        'same_shape_family_for_all_photo_grades':True,
        'no_hard_center_constraint':True,'visual_morphology':'head_stem_bowl_and_tail_calyx_well_with_five_lobes','source_photo_uv_unmodified':True,'profiles':info},ensure_ascii=False,indent=2),encoding='utf8')
    return info

def choose_visual_mesh(assets):
    global _visual_serial
    assets=Path(assets)
    i=_visual_serial%len(OFFSET_PROFILES_MM)
    target=assets/f'apple_uv_offset_{i:02}.obj'
    if not target.is_file():raise FileNotFoundError(f'Visual offset mesh missing: {target}')
    record={'spawn_index':_visual_serial,'variant':i,
        'top_xy_mm':OFFSET_PROFILES_MM[i][0],
        'bottom_xy_mm':OFFSET_PROFILES_MM[i][1],
        'mesh_file':target.name}
    with (assets/'stage77_v2_spawn_log.jsonl').open('a',encoding='utf8') as f:
        f.write(json.dumps(record,ensure_ascii=False)+'\n')
    _visual_serial+=1
    return target
