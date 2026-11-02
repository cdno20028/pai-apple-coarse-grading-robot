"""v0.9: FK and analytic IK for fixed-length 3R radial fingers.
Each finger has ONE rigid wrist head: camera at wrist, pad +6 mm inward and
7 mm behind (below), both driven by SAME three rotational joints.
No independent Cartesian tip interpolation, no telescoping links.
Timestamp: 2026/10/14 01:46 JST.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np

@dataclass(frozen=True)
class Design:
    apple_r: float=.040
    apple_z: float=.090
    plate_z: float=.050
    root_r: float=.125
    root_z: float=.175
    link1: float=.075
    link2: float=.073
    pad_inward: float=.006
    pad_below: float=.007
    pad_radius: float=.0045
    camera_fov_deg: float=80.
    scan_s: float=.52
    close_s: float=.08
    # t=0: camera sees low side from a practical height, not down by the plate.
    scan_start_r: float=.0615
    scan_start_z: float=.078
    scan_end_r: float=.054
    scan_end_z: float=.097
    # Closing moves the SAME wrist an additional 4.3 mm inward, not downward.
    grasp_camera_r: float=.0497
    grasp_camera_z: float=.097
    nominal_joint_velocity_limit: float=6.0  # rad/s, test assumption only

G=Design()

def smootherstep(x: float):
    x=max(0.,min(1.,x))
    return x*x*x*(x*(x*6-15)+10)

def commanded_wrist(t: float,g=G):
    """Target wrist (global cylindrical radius r, height z); IK must realize it."""
    t=float(max(0,min(g.scan_s+g.close_s,t)))
    if t<=g.scan_s:
        s=smootherstep(t/g.scan_s)
        return (g.scan_start_r+(g.scan_end_r-g.scan_start_r)*s,
                g.scan_start_z+(g.scan_end_z-g.scan_start_z)*s)
    s=smootherstep((t-g.scan_s)/g.close_s)
    return (g.scan_end_r+(g.grasp_camera_r-g.scan_end_r)*s,
            g.grasp_camera_z+(g.grasp_camera_z-g.scan_end_z)*s)

def ik(r: float,z: float,g=G):
    """2-link planar IK, elbow-outward branch, +wrist joint levels camera.
    Planar x points INWARD (toward apple center), z is up.
    """
    x=g.root_r-r; h=z-g.root_z
    c=(x*x+h*h-g.link1**2-g.link2**2)/(2*g.link1*g.link2)
    if c < -1.-1e-9 or c > 1.+1e-9:
        raise ValueError(f'Unreachable: r={r:.4f}, z={z:.4f}, cos={c:.3f}')
    q2=math.acos(max(-1.,min(1.,c)))
    q1=math.atan2(h,x)-math.atan2(g.link2*math.sin(q2),g.link1+g.link2*math.cos(q2))
    q3=-(q1+q2)
    return np.array([q1,q2,q3])

def fk(q,g=G):
    q1,q2,q3=np.asarray(q,dtype=float)
    # Global radial coordinates are outward positive.
    p0=np.array([g.root_r,g.root_z]);p1=p0+g.link1*np.array([-math.cos(q1), math.sin(q1)])
    p2=p1+g.link2*np.array([-math.cos(q1+q2),math.sin(q1+q2)])
    theta=q1+q2+q3
    # Local x inward; local z upward; normal/orientation fixed by wrist q3.
    pad=p2+g.pad_inward*np.array([-math.cos(theta),math.sin(theta)])+g.pad_below*np.array([-math.sin(theta),-math.cos(theta)])
    return np.vstack([p0,p1,p2,pad])

def q_at(t,g=G): return ik(*commanded_wrist(t,g),g)

def sphere_clearance(point, radius: float=0, g=G):
    return float(np.linalg.norm(np.asarray(point)-np.array([0.,g.apple_z]))-g.apple_r-radius)

def segment_clearance(a,b,thickness,g=G):
    p=np.asarray(a);v=np.asarray(b)-p;center=np.array([0.,g.apple_z])
    u=float(np.clip(np.dot(center-p,v)/max(1e-15,np.dot(v,v)),0.,1.))
    return sphere_clearance(p+u*v,thickness,g)

def derived_state(t,g=G):
    q=q_at(t,g);points=fk(q,g)
    cam=points[2];pad=points[3]
    az=g.apple_z-cam[1]
    # Camera mounted in wrist, optical +x inward (world radial negative).
    offaxis=math.degrees(math.atan2(abs(az),cam[0]))
    return dict(t_s=float(t),joints_rad=q.tolist(),camera_r_m=float(cam[0]),camera_z_m=float(cam[1]),
       pad_r_m=float(pad[0]),pad_z_m=float(pad[1]),
       pad_apple_clearance_mm=round(1000*sphere_clearance(pad,g.pad_radius,g),4),
       link_apple_clearance_mm=round(1000*min(segment_clearance(points[0],points[1],.003,g),
          segment_clearance(points[1],points[2],.003,g)),4),
       min_table_clearance_mm=round(1000*(min(points[0,1]-.003,points[1,1]-.003,
              cam[1]-.004,pad[1]-g.pad_radius)-g.plate_z),4),
       camera_offaxis_deg=round(offaxis,3))

def full_trajectory(g=G,dt=1./240.):
    ts=np.arange(0,g.scan_s+g.close_s+dt*.2,dt)
    if ts[-1] < g.scan_s+g.close_s-1e-10:ts=np.r_[ts,g.scan_s+g.close_s]
    ts=np.minimum(ts,g.scan_s+g.close_s)
    return ts,np.array([q_at(t,g) for t in ts])
