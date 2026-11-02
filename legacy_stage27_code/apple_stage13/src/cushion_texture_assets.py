"""Stage13: physical concave handoff nests, UV visual apples, synthetic textures.

Dimensions in metres. Cushion is a STATIC concave triangle mesh (not deformable
foam). Apple collision is unchanged GEOM_SPHERE; the UV sphere is VISUAL only.
Assets are reproducible and intentionally do not stand in for real orchard photos.
"""
from pathlib import Path
import math
import numpy as np
from PIL import Image

HERE=Path(__file__).resolve().parents[1]
ASSETS=HERE/'assets'/'stage13'
NEST_WIDTH_M=.100
NEST_PITCH_M=.115
CUP_DIAMETER_M=.094
CUP_DEPTH_M=.015
CUP_RIM_Z_M=.015
CUP_CENTER_FLOOR_Z_M=0.0
CUP_SLOPE_INNER_R_M=.045

PALETTES={
    'A':(202,48,35),
    'B':(203,101,39),
    'C':(100,47,27),
    'D':(125,151,54),
}


def build_uv_sphere(path:Path, latitude=30, longitude=64):
    """Unit-radius textured UV sphere, intended for Bullet visual meshScale."""
    path.parent.mkdir(parents=True,exist_ok=True)
    lines=['# Stage13 apple visual sphere, has OBJ vt UV coordinates']
    for i in range(latitude+1):
        theta=math.pi*i/latitude
        for j in range(longitude+1):
            u=j/longitude
            phi=2*math.pi*u
            x=math.sin(theta)*math.cos(phi)
            y=math.sin(theta)*math.sin(phi)
            z=math.cos(theta)
            # Slightly realistic apple shoulder and inset near stem, 4% max difference.
            rr=1.-.035*math.exp(-((theta)/.24)**2)
            lines.append(f'v {rr*x:.8f} {rr*y:.8f} {rr*z:.8f}')
            lines.append(f'vt {u:.8f} {1-i/latitude:.8f}')
    for i in range(latitude):
        for j in range(longitude):
            a=i*(longitude+1)+j+1
            b=a+longitude+1
            # outer normals; texture coordinates share vertex indices
            for x,y,z in ((a,b,a+1),(a+1,b,b+1)):
                lines.append(f'f {x}/{x} {y}/{y} {z}/{z}')
    path.write_text('\n'.join(lines)+'\n',encoding='ascii')
    return path


def build_cup_mesh(path:Path, sectors=48):
    """Watertight upward-opening dish: center floor=0, lip=+15mm.

    Mechanical approximation: stiff collision surface, not real cushioning.
    Positions are *relative to the original flat tray upper surface* so the
    baseline gripper reach is not silently changed.
    """
    path.parent.mkdir(parents=True,exist_ok=True)
    # Outer radius 49mm within 100mm nominal nest footprint.
    rings=[0.00001,.008,.016,.024,.032,.039,.045,.047,.049]
    lines=['# Stage13 open concave cushion, 100mm nominal handoff nest']
    vertices=[]
    for r in rings:
        for j in range(sectors):
            a=2*math.pi*j/sectors
            z=CUP_DEPTH_M*min((r/CUP_SLOPE_INNER_R_M)**2,1.)
            vertices.append((r*math.cos(a),r*math.sin(a),z))
    for j in range(sectors):
        a=2*math.pi*j/sectors
        vertices.append((.049*math.cos(a),.049*math.sin(a),-.009))
    vertices.append((0.,0.,-.009))
    vertices.append((0.,0.,0.))  # cap the very small center opening
    for x,y,z in vertices:
        lines.append(f'v {x:.8f} {y:.8f} {z:.8f}')
    faces=[]
    # top faces point upward
    for i in range(len(rings)-1):
        for j in range(sectors):
            a=i*sectors+j+1
            an=i*sectors+(j+1)%sectors+1
            b=(i+1)*sectors+j+1
            bn=(i+1)*sectors+(j+1)%sectors+1
            faces.extend(((a,b,bn),(a,bn,an)))
    # outside wall, bottom faces close mesh to static trimesh
    for j in range(sectors):
        a=(len(rings)-1)*sectors+j+1
        an=(len(rings)-1)*sectors+(j+1)%sectors+1
        b=len(rings)*sectors+j+1
        bn=len(rings)*sectors+(j+1)%sectors+1
        faces.extend(((a,b,bn),(a,bn,an)))
        bottom_center=len(vertices)-1
        faces.append((bottom_center,bn,b))
        top_center=len(vertices)
        faces.append((top_center,j+1,(j+1)%sectors+1))
    lines.extend(f'f {a} {b} {c}' for a,b,c in faces)
    path.write_text('\n'.join(lines)+'\n',encoding='ascii')
    return path


def apple_texture(label, path:Path, h=256, w=512):
    """Synthetic spherical equirectangular fruit appearance.

    No claim to photo texture. C has a large localized blemish.
    Uses the old classifier's rough color categories where possible.
    """
    if label not in PALETTES:raise ValueError(label)
    path.parent.mkdir(parents=True,exist_ok=True)
    y,x=np.mgrid[:h,:w];u=x/w;v=y/h
    rng=np.random.default_rng(20261017 + ord(label))
    fine=rng.normal(0,4,(h,w))
    stripe=(np.cos(2*math.pi*(6*u+1.5*np.sin(v*10)))+1)/2
    dapple=np.sin(2*math.pi*(u*31+v*5))*np.cos(2*math.pi*(u*19-v*8))
    base=np.array(PALETTES[label],dtype=float)
    rgb=np.tile(base,(h,w,1)).copy()
    delta=12*stripe+5*dapple+fine
    rgb+=delta[...,None]*np.array([.8,.35,.05])
    if label=='B':
        patch=np.exp(-(((u-.54)/.17)**2+((v-.52)/.30)**2)*2)
        rgb[...,0]+=24*patch;rgb[...,1]+=24*patch
    elif label=='C':
        # bruised region, dark brown and angular scars
        patch=np.exp(-(((u-.42)/.13)**2+((v-.51)/.22)**2)*2)
        rgb=rgb*(1-patch[...,None]*.6)+np.array([66,39,23])*patch[...,None]*.6
        scar=np.exp(-((u-(.49+.075*(v-.5)))**2)/(.007**2))*np.exp(-((v-.5)**2)/(.25**2))
        rgb=rgb*(1-scar[...,None]*.7)+np.array([35,20,14])*scar[...,None]*.7
    elif label=='D':
        rgb[...,0]+=5*np.sin(20*u+10*v)
        rgb[...,1]+=12*stripe
    # Polar stem and calyx zone (lat-lon UV v near either pole)
    pol=np.exp(-((v-.99)/.035)**2)+np.exp(-(v/.032)**2)
    rgb=rgb*(1-.25*pol[...,None])
    Image.fromarray(np.uint8(np.clip(rgb,0,255)),'RGB').save(path)
    return path


def ensure_assets():
    ASSETS.mkdir(parents=True,exist_ok=True)
    build_uv_sphere(ASSETS/'apple_uv.obj')
    build_cup_mesh(ASSETS/'cushion_cup.obj')
    for label in 'ABCD':apple_texture(label,ASSETS/f'apple_{label}.png')
    return ASSETS


def create_cushion(p, grade, x, y, z_floor):
    """One static cup; support sensor uses contacts to this EXACT body ID."""
    obj=ASSETS/'cushion_cup.obj'
    if not obj.exists():ensure_assets()
    col=p.createCollisionShape(p.GEOM_MESH,fileName=str(obj),
         flags=p.GEOM_FORCE_CONCAVE_TRIMESH)
    vis=p.createVisualShape(p.GEOM_MESH,fileName=str(obj),
         rgbaColor=[.19,.20,.23,1])
    body=p.createMultiBody(baseMass=0,baseCollisionShapeIndex=col,
         baseVisualShapeIndex=vis,basePosition=[x,y,z_floor])
    p.changeDynamics(body,-1,lateralFriction=.9,restitution=0.)
    return body


def spawn_textured_apple(p,label,mass,radius,position,friction):
    """Physical sphere + independently UV-textured visual sphere."""
    if label not in PALETTES:raise ValueError(label)
    if not (ASSETS/'apple_uv.obj').exists():ensure_assets()
    c=p.createCollisionShape(p.GEOM_SPHERE,radius=radius)
    v=p.createVisualShape(p.GEOM_MESH,fileName=str(ASSETS/'apple_uv.obj'),
           meshScale=[radius,radius,radius],rgbaColor=[1,1,1,1])
    body=p.createMultiBody(baseMass=mass,baseCollisionShapeIndex=c,
           baseVisualShapeIndex=v,basePosition=position)
    tex=p.loadTexture(str(ASSETS/f'apple_{label}.png'))
    p.changeVisualShape(body,-1,textureUniqueId=tex)
    p.changeDynamics(body,-1,lateralFriction=friction,restitution=0,
                     rollingFriction=.002)
    return body


def capture_in_cup(center_xy,fruit_xy,supported,velocity,threshold=.030):
    """Preliminary experiment metric, NOT certified fruit retention.

    Nominal 30mm xy tolerance is NOT a sponge compression model. Combined with
    actual Bullet support and speed it gives a conservative evaluation output.
    """
    error=math.hypot(float(center_xy[0])-float(fruit_xy[0]),
                     float(center_xy[1])-float(fruit_xy[1]))
    return bool(supported and float(velocity)<.10 and error<=threshold),error
