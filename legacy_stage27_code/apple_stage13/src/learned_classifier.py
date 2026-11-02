"""Minimal SUPERVISED learned policy for synthetic texture trials (not orchard grading).
Independent training labels are simulated RGB categories and scale measurements.
The model receives measured camera RGB features, not the scene's target class.
"""
from __future__ import annotations
import numpy as np
from functools import lru_cache
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

LABELS=('A','B','C','D')
# RGB appearances chosen for proof-of-concept: not realistic fruit grading.
APPEARANCES={
    'A':((0.79,0.12,0.09),0.18),
    'B':((0.79,0.42,0.10),0.22),
    'C':((0.38,0.16,0.08),0.15),
    'D':((0.49,0.58,0.13),0.17),
}

@lru_cache(maxsize=8)
def train(seed=111):
    rng=np.random.default_rng(seed)
    X=[]; y=[]
    for label,(rgb,mass) in APPEARANCES.items():
        for _ in range(350):
            # Strong nuisance light/white balance variation.
            gain=rng.uniform(.42,1.12)
            noise=rng.normal(0,.035,3)
            rgb_v=np.clip(np.array(rgb)*gain+noise,0,1)
            ms=mass+rng.normal(0,.012)
            X.append([*rgb_v,ms]);y.append(label)
    model=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000))
    model.fit(X,y)
    return model

def rgb_features(rgb_array):
    # Image-only foreground mask (no Bullet segmentation IDs supplied).
    rgb=np.asarray(rgb_array,dtype=np.float32)[...,:3]/255.
    r,g,b=rgb[...,0],rgb[...,1],rgb[...,2]
    red_fruit=(r>0.13)&(r>g*1.20)&(r>b*1.60)&((r-g)>.04)
    green_fruit=(g>.19)&(g>r*1.06)&(g>b*1.4)
    fruit=red_fruit|green_fruit
    if int(fruit.sum())<18: return None
    # Robust to smooth illumination changes; no simulated class accessed here.
    values=rgb[fruit]
    return [float(x) for x in np.median(values,axis=0)]

def infer(model, images, scale_mass_kg):
    f=[rgb_features(x) for x in images]
    valid=[z for z in f if z is not None]
    if not valid: return {'status':'NO_FRUIT_PIXELS','class':None,'confidence':None,'usable_frames':0}
    if scale_mass_kg is None or not (0.08<=scale_mass_kg<=0.4):
        return {'status':'NO_VALID_SCALE_READING','class':None,'confidence':None,'usable_frames':len(valid)}
    # Each observed angle has equal weight; use median to limit specular effects.
    rgb=np.median(valid,axis=0)
    features=np.array([*rgb,scale_mass_kg],dtype=float).reshape(1,-1)
    probs=model.predict_proba(features)[0];idx=int(np.argmax(probs))
    return {'status':'PREDICTED','class':str(model.classes_[idx]),
            'confidence':float(probs[idx]),
            'scores':{str(c):float(p) for c,p in zip(model.classes_,probs)},
            'usable_frames':len(valid),'rgb_median':rgb.tolist(),'scale_mass_kg':scale_mass_kg,
            'disclaimer':'Synthetic learned colour+mass labels, not learned scratches/bruise/quality recognition'}
