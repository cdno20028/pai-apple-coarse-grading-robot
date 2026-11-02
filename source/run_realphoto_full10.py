#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run user-photo-asset GATE8/Stage88, with fresh models and honest 10-case audit.

Usage after extracting the release ZIP:
  python run_realphoto_full10.py --repo . --out ./REAL17_RESULTS --install
The runner NEVER substitutes labels for predicted classes and never edits robotics control.
"""
from __future__ import annotations
# RUNFIX_20261029: unique audit dir and virgin Stage70 physics output dir
import argparse, datetime, hashlib, json, os, pathlib, shutil, subprocess, sys, traceback, zipfile
P=pathlib.Path
JST=datetime.timezone(datetime.timedelta(hours=9))

def log_time():return datetime.datetime.now(JST).strftime('%Y/%m/%d %H:%M JST')
def digest(p):return hashlib.sha256(P(p).read_bytes()).hexdigest()
def check_input(root):
    m=json.loads((root/'realphoto/asset_swap_manifest.json').read_text('utf-8'))
    assert m['original_real_photos_count']==17
    assert len(m['assignment'])==8
    for n,s in m['original_photo_sha256'].items():
        p=root/'realphoto/original_images'/n
        assert p.is_file() and digest(p)==s, 'RAW_SOURCE_INTEGRITY_FAILED '+n
    for rel,hs in m['replaced_assets'].items():
        p=root/rel
        assert p.is_file() and digest(p)==hs['real_photo_sha256'], 'ASSET_SWAP_FAILED '+rel
    print('REAL17_17_RAW_PHOTOS_HASH_OK; 8_PHYSICAL_FRUITS_MAPPED',flush=True)

def call(args,cwd,logfile,env=None):
    print('RUN',*map(str,args),flush=True)
    with P(logfile).open('a',encoding='utf-8') as f:
        f.write('\nCMD: '+' '.join(map(str,args))+'\n');f.flush()
        proc=subprocess.Popen(list(map(str,args)),cwd=cwd,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,errors='replace',bufsize=1)
        for line in proc.stdout:
            print(line,end='',flush=True);f.write(line)
        rc=proc.wait();f.write('\nRETURN_CODE '+str(rc)+'\n')
    return rc

def make_result(out,result_zip):
    with zipfile.ZipFile(result_zip,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file() and 'engine' not in p.relative_to(out).parts and p.suffix.lower() in {'.json','.csv','.txt','.mp4','.png','.md'}:
                z.write(p,p.relative_to(out))
    assert zipfile.ZipFile(result_zip).testzip() is None
    print('RESULT_ZIP',result_zip,'SIZE',result_zip.stat().st_size,flush=True)

def main():
    a=argparse.ArgumentParser();a.add_argument('--repo',default='.');a.add_argument('--out',default='REAL17_RESULTS');a.add_argument('--install',action='store_true');a.add_argument('--static-only',action='store_true')
    args=a.parse_args()
    root=P(args.repo).resolve();src=root/'stage88';requested_out=P(args.out).resolve()
    # Stage70 rejects existing output directories. The audit folder is NOT its output.
    # Allocate a distinct audit folder for every execution, preserving past attempts.
    out=requested_out
    seq=0
    while out.exists():
        seq+=1
        out=requested_out.with_name(f'{requested_out.name}_{seq:03d}')
    out.mkdir(parents=True)
    physics_out=out/'stage70_physics'  # must not exist when Stage70 starts
    result_zip=requested_out.parent/'PAI_G8_REAL17_FULL10_RESULT.zip'
    if result_zip.exists():
        backseq=1
        while (result_zip.parent/f'PAI_G8_REAL17_FULL10_RESULT_previous_{backseq:03d}.zip').exists():
            backseq+=1
        result_zip.rename(result_zip.parent/f'PAI_G8_REAL17_FULL10_RESULT_previous_{backseq:03d}.zip')
    logfile=out/'EXECUTION_LOG.txt';logfile.write_text(f'REAL17 execution {log_time()}\n',encoding='utf-8')
    print('REAL17_NEW_AUDIT_DIR',out,'STAGE70_OUTPUT_NOT_YET_CREATED',not physics_out.exists(),flush=True)
    status={'created_jst':log_time(),'new_real_photo_experiment':True,'g8_original_evidence_not_new_result':True,'static_pass':False,'physics_run':False,'physics_rc':None,'all_pass':False,'n_episodes':None}
    try:
        check_input(root)
        if args.install:
            if call([sys.executable,'-m','pip','install','-q','-r',src/'requirements.txt'],root,logfile)!=0:raise RuntimeError('PIP_INSTALL_FAILED')
        import sklearn
        print('SKLEARN_RUNTIME_VERSION',sklearn.__version__,flush=True)
        if sklearn.__version__!='1.8.0':raise RuntimeError('SKLEARN_VERSION_NOT_1.8.0')
        for rel in ['scripts/check_ai_asset_integration.py','scripts/check_public_completeness.py']:
            if call([sys.executable,root/rel],root,logfile):raise RuntimeError('CHECK_FAILED '+rel)
        status['static_pass']=True
        if args.static_only:
            print('STATIC_ONLY: PyBullet NOT EXECUTED',flush=True)
            return 0
        # Train coarse known-photo grade model afresh from camera RGB, no test-case label routing.
        if call([sys.executable,root/'scripts/train_real17_coarse_camera.py'],root,logfile):raise RuntimeError('COARSE_CAMERA_MODEL_TRAIN_FAILED')
        # Strict fresh training. Never reuse calibrated object-gate models from a prior Colab session.
        model_dir=src/'models'
        for p in model_dir.glob('*.joblib'):p.unlink()
        manifest=model_dir/'training_manifest.json'
        if manifest.exists():manifest.unlink()
        if call([sys.executable,'run_stage35_motion.py','--train-only'],src,logfile):raise RuntimeError('MODEL_TRAIN_FAILED')
        t=json.loads(manifest.read_text('utf-8'))
        if t.get('reused_existing') is not False:raise RuntimeError('NOT_FRESHLY_TRAINED')
        print('FRESH_AI_MODELS_TRAINED',flush=True)
        if call([sys.executable,'test_stage67_preflight.py'],src,logfile):raise RuntimeError('OBJECT_GATE_PREFLIGHT_FAILED')
        env=os.environ.copy();env.pop('STAGE84_SMALL_SCAR_DEMO',None);env['REAL17_COARSE_CAMERA_MODEL']='1'
        assert not physics_out.exists(), 'PHYSICS_OUTPUT_ALREADY_EXISTS'
        status['physics_run']=True
        status['physics_output_relpath']='stage70_physics/'
        rc=call([sys.executable,'-u','run_stage70_continuous.py','--output',physics_out],src,logfile,env)
        status['physics_rc']=rc
        summary=physics_out/'summary.json'
        if summary.is_file():
            s=json.loads(summary.read_text('utf-8'))
            status['n_episodes']=s.get('n_episodes')
            status['all_pass']=bool(s.get('all_pass') and s.get('n_episodes')==10 and rc==0)
        print('REAL17_10_CASES',status['n_episodes'],'ALL_PASS',status['all_pass'],'PYBULLET_RC',rc,'PHYSICS_DIR',physics_out,flush=True)
        return 0 if status['all_pass'] else 2
    except BaseException as exc:
        print('ERROR',str(exc),flush=True)
        with logfile.open('a',encoding='utf-8') as f:f.write(traceback.format_exc())
        status['error']=str(exc)
        return 3
    finally:
        (out/'RUN_STATUS_REAL17.json').write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8')
        (out/'EXPERIMENT_NOTES_REAL17.md').write_text('実写版の新たな物理試験のみを本結果として扱う。GATE8原本の旧画像10/10成功は別の実験。\n',encoding='utf-8')
        make_result(out,result_zip)

if __name__=='__main__':sys.exit(main())
