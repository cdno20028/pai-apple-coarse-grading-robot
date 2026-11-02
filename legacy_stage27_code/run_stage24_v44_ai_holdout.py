#!/usr/bin/env python3
"""Stage24 v4.4: *feature-level synthetic* validation, not real-camera/real-apple grading.

A held-out generated distribution uses fresh samples and seeds, independent from
training. This is a feature-space surrogate; the Bullet camera pipeline must be
separately assessed with visual physics experiments.
"""
from __future__ import annotations
import sys, json, argparse, math, time
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'apple_stage13/src'))
from learned_classifier import APPEARANCES,LABELS,train


def independent_features(seed,n_each,gains=(.42,1.12),color_noise=.035,mass_noise=.012):
    rng=np.random.default_rng(seed);X=[];y=[]
    for label in LABELS:
        rgb,mass=APPEARANCES[label]
        for _ in range(n_each):
            g=rng.uniform(*gains)
            z=np.clip(np.array(rgb)*g+rng.normal(0,color_noise,3),0,1)
            m=mass+rng.normal(0,mass_noise)
            X.append([*z,m]);y.append(label)
    return np.array(X,float),np.array(y)


def evaluate(model,X,y):
    proba=model.predict_proba(X);pred=model.classes_[np.argmax(proba,axis=1)]
    conf={gt:{guess:int(((y==gt)&(pred==guess)).sum()) for guess in LABELS} for gt in LABELS}
    return {'n':len(y),'correct':int((pred==y).sum()),'accuracy':float(np.mean(pred==y)),
        'confusion_true_rows_pred_cols':conf,'confidence_on_errors_mean':float(np.max(proba[pred!=y],axis=1).mean()) if (pred!=y).any() else None}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',default='stage24_v44_ai_feature_holdout.json');args=parser.parse_args()
    model=train(seed=111)
    assert model is train(seed=111),'trained model should be reused, not refitted every fruit'
    settings={
        'heldout_similar':{'seed':20261018,'n_each':350,'gains':(.42,1.12),'color_noise':.035,'mass_noise':.012},
        'lighting_shift_proxy':{'seed':20261019,'n_each':350,'gains':(.25,.70),'color_noise':.06,'mass_noise':.018},
        'colour_mass_shift_proxy':{'seed':20261020,'n_each':350,'gains':(.42,1.12),'color_noise':.08,'mass_noise':.025}
    }
    results={name:evaluate(model,*independent_features(**kw)) for name,kw in settings.items()}
    results['model_note']='Fresh, held-out synthetic feature vector tests. Not actual camera frames and not actual fruit quality.'
    results['cache_size']=train.cache_info().currsize
    results['train_seed']=111;results['labels']=list(LABELS)
    target=Path(args.out);target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:{'n':v['n'],'correct':v['correct'],'accuracy':round(v['accuracy'],4)} for k,v in results.items() if isinstance(v,dict)},ensure_ascii=False,indent=2))
    print('AI HOLDOUT JSON:',target.resolve())
    return results

if __name__=='__main__':main()
