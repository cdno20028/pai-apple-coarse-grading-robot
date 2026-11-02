"""URDF generation. Mechanically constrained 3R x 3 fingers, camera+pad on same wrist."""
from pathlib import Path
from xml.etree import ElementTree as ET
from xml.dom import minidom
from mechanism_v09_reference import G
from camera_model import PadDesign
import math

def child(el,tag,**attrs):return ET.SubElement(el,tag,{k:str(v) for k,v in attrs.items()})
def body(robot,name,mass=0.03):
    b=child(robot,'link',name=name);inert=child(b,'inertial');child(inert,'origin',xyz='0 0 0',rpy='0 0 0');child(inert,'mass',value=mass)
    child(inert,'inertia',ixx='0.00008',iyy='0.00008',izz='0.00008',ixy='0',ixz='0',iyz='0')
    return b

def geom(b,shape,size,xyz='0 0 0',rpy='0 0 0',color='0.75 0.80 0.87 1',collide=True):
    for tag in (('visual','collision') if collide else ('visual',)):
        item=child(b,tag);child(item,'origin',xyz=xyz,rpy=rpy);g=child(item,'geometry')
        if shape=='box':child(g,'box',size=size)
        elif shape=='sphere':child(g,'sphere',radius=size)
        elif shape=='cylinder':child(g,'cylinder',radius=size[0],length=size[1])
        if tag=='visual':mat=child(item,'material',name='mat');child(mat,'color',rgba=color)

def joint(robot,name,parent,leaf,xyz='0 0 0',rpy='0 0 0',axis='0 -1 0',kind='revolute',limits=(-3.,3.),effort=8,velocity=6):
    j=child(robot,'joint',name=name,type=kind);child(j,'parent',link=parent);child(j,'child',link=leaf)
    child(j,'origin',xyz=xyz,rpy=rpy)
    if kind!='fixed':
        child(j,'axis',xyz=axis)
        child(j,'limit',lower=limits[0],upper=limits[1],effort=effort,velocity=velocity)
        child(j,'dynamics',damping='0.02',friction='0.0')
    return j

def build(path):
    robot=ET.Element('robot',name='apple_3_finger_dual_role_v10')
    design=PadDesign(length_mm=18,cam_tangent_mm=11)
    body(robot,'base',1.0)
    body(robot,'stage_x',1.0);body(robot,'stage_y',1.0);head=body(robot,'head',.5)
    geom(head,'cylinder',(.017,.010),color='0.22 0.25 0.32 1')
    joint(robot,'stage_x_move','base','stage_x',axis='1 0 0',kind='prismatic',limits=(-.4,.4),effort=500,velocity=1.0)
    joint(robot,'stage_y_move','stage_x','stage_y',axis='0 1 0',kind='prismatic',limits=(-.4,.4),effort=500,velocity=1.0)
    joint(robot,'stage_z_move','stage_y','head',axis='0 0 1',kind='prismatic',limits=(.15,.45),effort=500,velocity=.5)
    # Palm camera is a rigid appendage, its optical axis is downward.
    palm=body(robot,'palm_camera',.004)
    joint(robot,'palm_camera_fixed','head','palm_camera',xyz='0 0 -0.028',kind='fixed')
    geom(palm,'box','0.011 0.011 0.007',color='0.13 0.42 0.86 1',collide=False)
    for k in range(3):
        a=k*2*math.pi/3
        mount=body(robot,f'mount_{k}',.015)
        # Local +X points inwards. Local +Z is vertical.
        joint(robot,f'mount_{k}_fixed','head',f'mount_{k}',
              xyz=f'{G.root_r*math.cos(a):.8f} {G.root_r*math.sin(a):.8f} 0',
              rpy=f'0 0 {a+math.pi:.9f}',kind='fixed')
        first=body(robot,f'proximal_{k}',.040)
        joint(robot,f'shoulder_{k}',f'mount_{k}',f'proximal_{k}',axis='0 -1 0',limits=(-2.8,.8))
        geom(first,'cylinder',(.003,G.link1),xyz=f'{G.link1/2} 0 0',rpy=f'0 {math.pi/2} 0',color='0.60 0.67 0.75 1')
        second=body(robot,f'distal_{k}',.035)
        joint(robot,f'elbow_{k}',f'proximal_{k}',f'distal_{k}',xyz=f'{G.link1} 0 0',axis='0 -1 0',limits=(0.0,2.8))
        geom(second,'cylinder',(.003,G.link2),xyz=f'{G.link2/2} 0 0',rpy=f'0 {math.pi/2} 0',color='0.28 0.50 0.75 1')
        wrist=body(robot,f'wrist_{k}',.028)
        joint(robot,f'wrist_{k}_joint',f'distal_{k}',f'wrist_{k}',xyz=f'{G.link2} 0 0',axis='0 -1 0',limits=(-2.7,2.7))
        # Camera + a long, rounded contact pad, both rigidly fixed to SAME wrist.
        # local +x inward; local -y = world positive tangential direction.
        geom(wrist,'box','0.006 0.007 0.006',xyz='-0.002 0 0',color='0.32 0.45 0.65 1')
        geom(wrist,'cylinder',(.003,design.length_mm/1000),
             xyz=f'{design.radial_offset_mm/1000:.5f} 0 {-design.downward_offset_mm/1000:.5f}',
             color='1.0 0.53 0.08 1')
        for zoff in [-design.downward_offset_mm/1000-design.length_mm/2000,
                     -design.downward_offset_mm/1000+design.length_mm/2000]:
            geom(wrist,'sphere',design.radius_mm/1000,
                 xyz=f'{design.radial_offset_mm/1000:.5f} 0 {zoff:.5f}',color='1.0 0.53 0.08 1')
        # Fixed camera attachment to SAME wrist. Its yaw is calibrated once.
        camera=body(robot,f'finger_camera_{k}',.004)
        yaw=math.atan(design.cam_tangent_mm/65.)
        joint(robot,f'finger_camera_{k}_fixed',f'wrist_{k}',f'finger_camera_{k}',
              xyz=f'0 {-design.cam_tangent_mm/1000:.5f} {design.cam_up_mm/1000:.5f}',
              rpy=f'0 0 {yaw:.10f}',kind='fixed')
        geom(camera,'box','0.006 0.006 0.004',xyz='-0.004 0 0',
             color='0.10 0.45 0.95 1',collide=False)
    xml=minidom.parseString(ET.tostring(robot)).toprettyxml(indent='  ')
    Path(path).write_text(xml,encoding='utf-8')
    return path

if __name__=='__main__':build(Path(__file__).resolve().parents[1]/'robot_v10.urdf')
