from pathlib import Path
import json, hashlib, ast
r=Path(__file__).parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
expect={
 'source/stage88/stage84_camera_ai.py':'490bca16f9966bec42b817eb8cf8fc3a6bb1f0e06540f963ba9f38b56e2630e4',
 'source/stage88/stage90_wound_evidence.py':'dd5c008cf7b3c4d56552e4646b2b5a3bb011926b186b2a68f078f0a1def5bda0',
 'reproduction/p104_run.py':'3d85d266c31b3e5f465f15c43b0e42048e9dcac6f563ca2ea5891e4f565a5e73'
}
for n,h in expect.items():
 p=r/n;assert sha(p)==h,('SOURCE_HASH_MISMATCH',n)
p104=json.loads((r/'evidence/P104_VERDICT.json').read_text())
p105=json.loads((r/'evidence/P105_CAPTURE_VERDICT.json').read_text())
assert p104['ALL_10_PHYSICAL_PASS_VERIFIED'] is True
assert len(p104['grading']['cases'])==10
assert all(v['pass'] for v in p104['grading']['cases'].values())
assert p105['reproduction_claim'] is True and p105['rerun_physics_10of10'] is True
cam=p105['extra_cameras'];assert cam['camera_frame_counts']=={'grip':656,'follow':656}
assert cam['real_pybullet_getCameraImage'] is True
print('STATIC_EVIDENCE_HASH_AND_10CASE_AND_P105_PASS')
print('CAUTION: stored proof validation only; not a new physical run')
