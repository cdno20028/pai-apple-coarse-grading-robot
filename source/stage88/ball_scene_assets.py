#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create modelled spherical sports balls for actual PyBullet camera/plate tests.

All meshes and textures are procedural approximations, NOT scans or photographs.
Collision uses GEOM_SPHERE and independently measured-like, nominal masses.
"""
from __future__ import annotations
from pathlib import Path
import math
import numpy as np
from PIL import Image

KINDS = ('baseball','tennis','red_ball')
SPECS = {
  'baseball': {'radius_m':.0365, 'mass_kg':.145, 'label':'baseball'},
  'tennis': {'radius_m':.0335, 'mass_kg':.057, 'label':'tennis'},
  'red_ball': {'radius_m':.0380, 'mass_kg':.180, 'label':'red_ball'},
}


def unit_sphere_mesh(folder:Path,segments:int=64,rings:int=32)->Path:
    """Unit sphere with UV coordinates, designed for p.GEOM_MESH rendering."""
    folder.mkdir(exist_ok=True,parents=True)
    obj=folder/'sports_ball_unit_sphere.obj'
    if obj.exists():return obj
    lines=['# procedural UV sphere; radius 1; generated for stage34', 'o sports_ball']
    # Direct lat/lon, duplicate seam vertices for proper UV mapping
    for k in range(rings+1):
        v=k/rings
        theta=math.pi*v
        for j in range(segments+1):
            u=j/segments
            phi=2*math.pi*u
            x=math.sin(theta)*math.cos(phi)
            y=math.sin(theta)*math.sin(phi)
            z=math.cos(theta)
            lines.append(f'v {x:.8f} {y:.8f} {z:.8f}')
    for k in range(rings+1):
        v=k/rings
        for j in range(segments+1):
            u=j/segments
            lines.append(f'vt {u:.8f} {1-v:.8f}')
    for k in range(rings):
        for j in range(segments):
            a=k*(segments+1)+j+1
            b=a+1;c=a+segments+2;d=a+segments+1
            # double triangles oriented outward based on param orientation
            lines.append(f'f {a}/{a} {c}/{c} {b}/{b}')
            lines.append(f'f {a}/{a} {d}/{d} {c}/{c}')
    obj.write_text('\n'.join(lines)+'\n',encoding='utf8')
    return obj


def make_texture(kind:str, variant:int, folder:Path)->Path:
    if kind not in SPECS:raise ValueError(kind)
    folder.mkdir(exist_ok=True,parents=True)
    dest=folder/f'{kind}_variant{variant}.png'
    if dest.exists():return dest
    width,height=512,256
    y,x=np.mgrid[:height,:width]
    u=x/width
    v=y/height
    rng=np.random.default_rng(3401+17*variant+{'baseball':11,'tennis':23,'red_ball':37}[kind])
    noise=rng.normal(0,1.8,(height,width,1))
    if kind=='baseball':
        color=np.array([235,229,214],float)
        surface=np.broadcast_to(color,(height,width,3)).astype(float).copy()+noise
        # Baseball red sewn seams: two winding meanders on the projected sphere texture.
        seam_x=(0.20+0.055*np.sin((v+variant*.15)*2*np.pi))%1
        seam_x2=(.70-.055*np.sin((v+variant*.15)*2*np.pi))%1
        seam=(np.minimum(abs(u-seam_x),1-abs(u-seam_x))<.006)|\
             (np.minimum(abs(u-seam_x2),1-abs(u-seam_x2))<.006)
        stitches=(np.sin(v*80*np.pi+u*30)>-.4)
        surface[seam & stitches]=[156,47,45]
    elif kind=='tennis':
        color=np.array([192,215,48],float)
        surface=np.broadcast_to(color,(height,width,3)).astype(float).copy()+noise*2.0
        # Curvy off-white seams follow ball UV, independent of fruit color.
        seam=(abs(v-(.32+.14*np.sin(2*np.pi*(u+variant*.09))))<.013)|\
             (abs(v-(.68-.14*np.sin(2*np.pi*(u+variant*.09))))<.013)
        surface[seam]=[236,239,207]
    else:
        color=np.array([204,38,31],float)
        surface=np.broadcast_to(color,(height,width,3)).astype(float).copy()+noise*.9
        # Uniform rubber, modest speckles.  No apple calyx/stem.
        spots=rng.uniform(0,1,(height,width))<.001
        surface[spots]=[174,32,27]
    Image.fromarray(np.uint8(np.clip(surface,0,255)),'RGB').save(dest)
    return dest


def spawn_ball(p, kind:str, variant:int, at_xyz:list[float], folder:Path)->int:
    """A physically simulated sphere rests directly on the existing contact plate."""
    spec=SPECS[kind]
    obj=unit_sphere_mesh(folder)
    texture=make_texture(kind,variant,folder)
    radius=spec['radius_m']
    collision=p.createCollisionShape(p.GEOM_SPHERE,radius=radius)
    visual=p.createVisualShape(p.GEOM_MESH,fileName=str(obj),meshScale=[radius]*3,
                               rgbaColor=[1.,1.,1.,1.])
    q=p.getQuaternionFromEuler([0.,0.,variant*.57])
    body=p.createMultiBody(baseMass=spec['mass_kg'],baseCollisionShapeIndex=collision,
                            baseVisualShapeIndex=visual,basePosition=at_xyz,baseOrientation=q)
    tex_id=p.loadTexture(str(texture))
    p.changeVisualShape(body,-1,textureUniqueId=tex_id,rgbaColor=[1.,1.,1.,1.])
    p.changeDynamics(body,-1,lateralFriction=.8,restitution=.02,linearDamping=.03,angularDamping=.03)
    return body


def static_checks():
    folder=Path(__file__).resolve().parent/'sports_ball_assets'
    unit_sphere_mesh(folder)
    for kind in KINDS:
        for v in (0,1):make_texture(kind,v,folder)
    obj=folder/'sports_ball_unit_sphere.obj'
    assert obj.is_file() and obj.stat().st_size>100000
    for kind in KINDS:
        for v in (0,1):
            a=np.array(Image.open(folder/f'{kind}_variant{v}.png'))
            assert a.shape==(256,512,3)
            assert float(np.std(a))>3.
    return {'mesh':obj.name,'kinds':list(KINDS),'variants':2,'images':6}

if __name__=='__main__':print(static_checks())
