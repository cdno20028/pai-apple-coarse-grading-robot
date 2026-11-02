#!/usr/bin/env python3
"""Validate new public standalone source inputs without PyBullet. Not a physics test."""
import sys,time
from pathlib import Path
repo=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(repo/'stage88'))
import run_stage35_motion as motion
from stage67_highview_gate import train_shared_stage67_high_gate
from compare_stage30 import source_clone
from tempfile import TemporaryDirectory
s=repo/'stage88'
model=motion.prepare_models(s/'models')
print('TRAIN_MODELS_PASS',model,flush=True)
gate,info=train_shared_stage67_high_gate(s)
assert len(info['high_view_training_and_replay_checks'])==4
assert info['synthetic_negative_false_accepts']==0
print('PUBLIC_SYNTHETIC_GATE_TRAIN_PASS',flush=True)
with TemporaryDirectory() as tmp:
 engine=source_clone('baseline_safe',Path(tmp)/'experiment')
 assert (engine/'apple_stage13/src/run_physics.py').exists()
print('PYBULLET_ENGINE_SOURCE_CLONE_PASS; actual physics NOT executed',flush=True)
