#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""Rebuild known-photo nearest-prototype model from archived real camera RGB.
Training labels are used ONLY in training; inference accepts camera frames only.
"""
from pathlib import Path
import json,sys,numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'stage88'))
from stage_real17_coarse_camera_ai import embed_finger_images, infer_known_photo_grade

def main():
    folder=ROOT/'stage88'/'real17_coarse_camera_training'
    manifest=json.loads((folder/'TRAINING_PROVENANCE.json').read_text('utf8'))
    feats=[];labels=[]
    for item in manifest['labels']:
        images=[np.asarray(Image.open(folder/n).convert('RGB')) for n in item['frames']]
        feats.append(embed_finger_images(images));labels.append(item['grade'])
    P=np.stack(feats).astype(np.float32)
    assert P.shape==(9,9216),P.shape
    assert len(labels)==9
    # Check all prototypes against their own class; this is in-sample.
    model=folder/'camera_rgb_prototypes.npz'
    np.savez_compressed(model,prototypes=P,labels=np.asarray(labels,dtype='U1'))
    for i,item in enumerate(manifest['labels']):
        images=[np.asarray(Image.open(folder/n).convert('RGB')) for n in item['frames']]
        frames={('finger',j,0):img for j,img in enumerate(images)}
        r=infer_known_photo_grade(frames)
        assert r['accepted'] and r['class']==item['grade'] and r['nearest_prototype_distance']==0,r
    print('KNOWN_CAMERA_TRAINING_REPLAY_9_OF_9_MATCH; NOT_INDEPENDENT',flush=True)
    print('MODEL_FILE',model,model.stat().st_size,flush=True)
    return 0

if __name__=='__main__':raise SystemExit(main())
