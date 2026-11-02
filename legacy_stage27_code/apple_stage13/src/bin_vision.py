"""v2.1: crate occupancy based on simulated palm RGB-D observation, not hidden object states.

The existing *palm camera* is positioned over each crate prior to accepting the
next apple. Pixel depths are transformed to world coordinates using camera
pose. The planning stage only sees depth hits (no PyBullet body-ID shortcut).
Requires a physics-engine run to validate any scene outcome.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np

@dataclass(frozen=True)
class BinPlan:
    free: bool
    target_xy: tuple[float,float] | None
    support_z: float | None
    visible_floor_fraction: float
    reason: str


def choose_clear_floor(points_xyz, center_xy, floor_z=.040, inside_half=.099,
                       apple_radius=.040, clearance=.004):
    """Use depth-observed 3D hits, never GT apple position. Flat bed preferred.

    Output safe center of apple in world XY or None if there is no visually
    confirmed clear footprint. Unseen space is unknown, NOT considered free.
    """
    points=np.asarray(points_xyz,dtype=float).reshape(-1,3)
    good=points[np.isfinite(points).all(axis=1)]
    cx,cy=map(float,center_xy)
    radius=apple_radius+clearance
    # Candidate positions are conservative and avoid the physical crate walls.
    xs=(-.052,0.,.052)
    candidates=[(cx+dx,cy+dy) for dx in xs for dy in xs]
    scored=[]
    for x,y in candidates:
        if abs(x-cx)+radius>inside_half or abs(y-cy)+radius>inside_half:
            continue
        d=np.linalg.norm(good[:,:2]-[x,y],axis=1)
        local=good[d<radius]
        # Require substantial floor sampling; no invented free space.
        if len(local)<25: continue
        floor=np.abs(local[:,2]-floor_z)<.012
        occupied=local[:,2]>floor_z+.017
        coverage=float(np.mean(floor))
        if np.any(occupied) or coverage<.90:
            continue
        # Prefer center; do not move farther than necessary.
        score=(x-cx)**2+(y-cy)**2
        scored.append((score,x,y,coverage))
    if not scored:
        return BinPlan(False,None,None,0.,'NO_CONFIRMED_FREE_FLOOR')
    _,x,y,coverage=min(scored)
    return BinPlan(True,(float(x),float(y)),floor_z,float(coverage),'OBSERVED_CLEAR_FLOOR')


def rgbd_from_existing_palm(p,robot,joints,crate_xy, fov_deg=75.,size=128):
    """Depth image from the existing palm link in its *actual* simulated pose.
    Unknown/invisible floor cannot be reported as vacant. Returns RGB + hits.
    """
    idx=joints['palm_camera_fixed']
    state=p.getLinkState(robot,idx,computeForwardKinematics=True)
    eye=np.asarray(state[4],dtype=float)
    R=np.asarray(p.getMatrixFromQuaternion(state[5]),dtype=float).reshape(3,3)
    forward=-(R@np.array([0.,0.,1.]))
    up=R@np.array([0.,1.,0.])
    right=np.cross(forward,up)
    right/=max(np.linalg.norm(right),1e-12)
    near,far=.01,1.0
    view=p.computeViewMatrix(eye.tolist(),(eye+forward).tolist(),up.tolist())
    proj=p.computeProjectionMatrixFOV(fov=fov_deg,aspect=1.,nearVal=near,farVal=far)
    _,_,rgba,depth,_=p.getCameraImage(size,size,viewMatrix=view,projectionMatrix=proj,
                                        renderer=p.ER_TINY_RENDERER)
    rgb=np.asarray(rgba,dtype=np.uint8).reshape(size,size,4)[:,:,:3]
    d=np.asarray(depth,dtype=float).reshape(size,size)
    v,u=np.indices((size,size))
    tangent=math.tan(math.radians(fov_deg)/2.)
    px=(2.*(u+.5)/size-1.)*tangent
    py=(1.-2.*(v+.5)/size)*tangent
    dist=near*far/(far-(far-near)*d)
    pts=(eye[None,None,:]+dist[:,:,None]*(forward[None,None,:]+px[:,:,None]*right[None,None,:]+py[:,:,None]*up[None,None,:]))
    valid=(d<.995)&(dist>.01)
    return rgb,pts[valid],eye
