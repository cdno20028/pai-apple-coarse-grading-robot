# -*- coding: utf-8 -*-
"""Static-only tests for the Stage87 close-up camera hook."""
from pathlib import Path
import ast
import hashlib
s=Path(__file__).resolve().parent
main=(s/'run_stage70_continuous.py').read_text(encoding='utf-8')
cam=(s/'stage87_grip_camera.py').read_text(encoding='utf-8')
ast.parse(main)
ast.parse(cam)
assert main.count('from stage87_grip_camera import Stage87ActualGripCamera')==1
assert main.count("grip_rec=Stage87GripCamera")==0
assert main.count('grip_rec=Stage87ActualGripCamera(p,result_dir)')==1
assert main.count("grip_rec.on_tick(live['case_number'],stage,robot,apple,joints)")==1
assert main.count('grip_rec.close()')==1
assert cam.count('p.getCameraImage(')==2
assert 'p.stepSimulation(' not in cam
assert 'resetBasePositionAndOrientation(' not in cam
assert 'setJointMotorControl' not in cam
assert 'im.paste(Image.fromarray(pa)' in cam
assert 'im.paste(Image.fromarray(pb)' in cam
assert "'gravity_lift'" in cam and "'scan_and_close'" in cam
print('STAGE87_STATIC_CAMERA_HOOK_PASS: actual PyBullet getCameraImage; no motor/physics changes; two camera angles; no Colab physics claimed')
