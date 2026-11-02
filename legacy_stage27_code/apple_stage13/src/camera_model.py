"""v1.0: CPU pinhole rendering and 3D ray/capsule occlusion of a dual-role finger.
NOT a rigid-body dynamics simulator. Rendering uses actual 3D ray intersections;
no camera aim-tracking, no idealized 'fruit in FOV = observable' shortcut.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import math, sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mechanism_v09_reference import G, q_at, fk, commanded_wrist

APPLE_C=np.array([0.,0.,G.apple_z]); APPLE_R=G.apple_r
CAM_FOV=80.

@dataclass(frozen=True)
class PadDesign:
    length_mm: float=18.
    radial_offset_mm: float=8.
    downward_offset_mm: float=10.
    cam_up_mm: float=3.
    cam_tangent_mm: float=0.
    cam_radial_out_mm: float=0.
    radius_mm: float=3.


def radial3(r,z,a): return np.array([r*np.cos(a),r*np.sin(a),z],dtype=float)
def tangent3(a): return np.array([-np.sin(a),np.cos(a),0.],dtype=float)
def norm(v): return v/np.maximum(1e-15,np.linalg.norm(v,axis=-1,keepdims=True))


def finger_parts(t,design=PadDesign()):
    """World-space *actual* FK and fixed geometry, 120 degree layout."""
    p=fk(q_at(float(t))) # root, elbow, wrist in radial-z plane
    out=[]
    for k in range(3):
        az=2*np.pi*k/3
        p0,p1,p2=[radial3(x,z,az) for x,z in p[:3]]
        camera=p2+np.array([0,0,design.cam_up_mm/1000]) + design.cam_tangent_mm/1000*tangent3(az) + design.cam_radial_out_mm/1000*np.array([np.cos(az),np.sin(az),0.])
        padcenter=radial3(p[2,0]-design.radial_offset_mm/1000,
                         p[2,1]-design.downward_offset_mm/1000,az)
        dl=design.length_mm/2000
        pad_a=padcenter+np.array([0,0,-dl]);pad_b=padcenter+np.array([0,0,dl])
        # Camera yaw is a fixed mounting angle, calibrated only at nominal wrist radius.
        inward=np.array([-np.cos(az),-np.sin(az),0.])
        optical=norm(inward-(design.cam_tangent_mm/65.)*tangent3(az))
        out.append({'k':k,'az':az,'root':p0,'elbow':p1,'wrist':p2,'camera':camera,
                    'dir':optical,'right':tangent3(az),'up':np.array([0.,0.,1.]),
                    'pad_a':pad_a,'pad_b':pad_b,'padcenter':padcenter})
    return out


def ray_sphere(o,ds,c,r):
    oc=np.asarray(o)-np.asarray(c)
    b=np.sum(ds*oc,axis=-1);cc=np.sum(oc*oc,axis=-1)-r*r
    det=b*b-cc
    t=-b-np.sqrt(np.maximum(det,0.))
    return np.where((det>=0)&(t>1e-7),t,np.inf)


def ray_capsule(o,ds,a,b,r):
    """First hit with capsule = finite round cylinder plus sphere end caps.
    Ray directions normalised, shape-safe broadcasting for Nx3 arrays.
    """
    o=np.asarray(o);a=np.asarray(a);b=np.asarray(b)
    axis=b-a;ll=np.linalg.norm(axis)
    if ll<1e-10:return ray_sphere(o,ds,a,r)
    v=axis/ll;w=o-a
    du=np.sum(ds*v,axis=-1);wu=np.sum(w*v,axis=-1)
    da=ds-du[...,None]*v;wa=w-wu[...,None]*v
    aa=np.sum(da*da,axis=-1);bb=np.sum(da*wa,axis=-1);cc=np.sum(wa*wa,axis=-1)-r*r
    disc=bb*bb-aa*cc
    ts=(-bb-np.sqrt(np.maximum(disc,0)))/np.maximum(aa,1e-12)
    h=wu+ts*du
    ts=np.where((disc>=0)&(aa>1e-12)&(ts>1e-7)&(h>=0)&(h<=ll),ts,np.inf)
    return np.minimum(ts,np.minimum(ray_sphere(o,ds,a,r),ray_sphere(o,ds,b,r)))


def obstacles(parts,design=PadDesign(),with_links=True):
    arr=[]
    for p in parts:
        arr.append(('pad',p['k'],p['pad_a'],p['pad_b'],design.radius_mm/1000))
        if with_links:
            arr.append(('link',p['k'],p['root'],p['elbow'],.003))
            arr.append(('link',p['k'],p['elbow'],p['wrist'],.003))
    return arr


def cam_spec(finger):
    return {'origin':finger['camera'],'direction':finger['dir'],'right':finger['right'],'up':finger['up']}

def palm_spec():
    return {'origin':np.array([0.,0.,.175-.028]),'direction':np.array([0.,0.,-1.]),
            'right':np.array([1.,0.,0.]),'up':np.array([0.,1.,0.])}

def plate_spec():
    return {'origin':np.array([0.,0.,G.plate_z+.001]),'direction':np.array([0.,0.,1.]),
            'right':np.array([1.,0.,0.]),'up':np.array([0.,1.,0.])}

def camera_rays(cam,n=160):
    xs=(np.arange(n)+.5-n/2)/(n/2)*math.tan(math.radians(CAM_FOV/2))
    xx,yy=np.meshgrid(xs,-xs)
    rays=(cam['direction'][None,None,:]+xx[...,None]*cam['right'][None,None,:]
          +yy[...,None]*cam['up'][None,None,:])
    return norm(rays).reshape(-1,3)


def get_ray_intersections(cam,parts,design=PadDesign(),n=160,apple_center=APPLE_C,
                          with_links=True):
    o=cam['origin'];ds=camera_rays(cam,n)
    fruit_t=ray_sphere(o,ds,apple_center,APPLE_R)
    closest_t=np.full(len(ds),np.inf);kind=np.zeros(len(ds),dtype=np.int8)
    for name,_,a,b,r in obstacles(parts,design,with_links):
        ob_t=ray_capsule(o,ds,a,b,r)
        change=ob_t<closest_t
        closest_t=np.minimum(closest_t,ob_t)
        kind[change]=2 if name=='pad' else 3
    fruit_mask=np.isfinite(fruit_t)
    blocked=fruit_mask&(closest_t<fruit_t-1e-7)
    label=np.zeros(len(ds),np.int8)
    label[fruit_mask]=1
    label[blocked]=kind[blocked]
    points=o+fruit_t[np.isfinite(fruit_t),None]*ds[np.isfinite(fruit_t)]
    return {'label':label.reshape(n,n),'blocked_count':int(blocked.sum()),
            'fruit_count':int(fruit_mask.sum()),'fruit_hit_t':fruit_t,'rays':ds,
            'object_hit_t':closest_t}


def render_cam(cam,parts,design=PadDesign(),n=160,apple_center=APPLE_C,with_links=True):
    rr=get_ray_intersections(cam,parts,design,n,apple_center,with_links)
    label=rr['label'].reshape(n,n)
    rgb=np.empty((n,n,3),dtype=np.uint8);rgb[:]=[241,243,246]
    yy,xx=np.mgrid[:n,:n];rgb[((xx//16+yy//16)%2)==0]=[235,239,244]
    fruit=label==1
    ds=rr['rays'];ft=rr['fruit_hit_t'];good=np.isfinite(ft)
    points=np.zeros((n*n,3));points[good]=cam['origin']+ft[good,None]*ds[good]
    nrm=(points-apple_center)/APPLE_R
    lighting=np.clip(.45+.5*np.maximum(0,nrm@norm(np.array([-.3,-.35,1.]))),0,1)
    mott=5*np.sin(nrm[:,0]*77)*np.cos(nrm[:,2]*59)
    # Synthetic apple, includes blemish on a fixed spherical region.
    scar=(np.sum((nrm-norm(np.array([.96,.2,.2])))**2,axis=1)<.035)
    body=np.stack([171+69*lighting+mott,35+36*lighting+mott,29+33*lighting+mott],axis=-1)
    body[scar]=[86,49,30]
    body=np.uint8(np.clip(body,0,255)).reshape(n,n,3)
    rgb[fruit]=body[fruit]
    rgb[label==2]=[244,151,40]
    rgb[label==3]=[65,115,171]
    # Secondary obstacles not overlapping apple also shown.
    obstruction=rr['object_hit_t']<rr['fruit_hit_t']
    other=(~np.isfinite(rr['fruit_hit_t']))&np.isfinite(rr['object_hit_t'])
    rgb.reshape(-1,3)[other]=[124,145,170]
    return rgb,rr


def sphere_points(n=5000):
    i=np.arange(n)+.5
    z=1.-2*i/n
    phi=np.pi*(3-np.sqrt(5))*i
    r=np.sqrt(1-z*z)
    return APPLE_C+APPLE_R*np.stack([r*np.cos(phi),r*np.sin(phi),z],axis=-1)


def projected_surface_visibility(cam,points,parts,design,with_links=True,apple_center=APPLE_C):
    ray=points-cam['origin'];dist=np.linalg.norm(ray,axis=1);ds=ray/dist[:,None]
    outward=norm(points-apple_center)
    facing=(np.sum(outward*(-ds),axis=1)>1e-6)
    # Strict square pinhole FOV, not circular cone FOV
    dot_forward=ds@cam['direction'];right=np.abs(ds@cam['right']);up=np.abs(ds@cam['up'])
    hf=math.tan(math.radians(CAM_FOV/2))
    facing &= (right<=dot_forward*hf)&(up<=dot_forward*hf)
    remaining=facing.copy()
    for _,_,a,b,r in obstacles(parts,design,with_links):
        t=ray_capsule(cam['origin'],ds,a,b,r)
        remaining &= ~(t<dist-2e-4)
    return remaining


def pad_apple_clearance(parts,design):
    """Min surface distance from capsule to apple sphere, assuming spherical apple."""
    res=[]
    for p in parts:
        a,b=p['pad_a'],p['pad_b'];v=b-a
        h=np.clip(np.dot(APPLE_C-a,v)/np.dot(v,v),0.,1.)
        near=a+h*v
        res.append(np.linalg.norm(near-APPLE_C)-APPLE_R-design.radius_mm/1000)
    return float(min(res))
