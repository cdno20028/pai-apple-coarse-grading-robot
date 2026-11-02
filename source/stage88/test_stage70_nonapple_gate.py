#!/usr/bin/env python3
from pathlib import Path
import json,copy
from stage70_nonapple_gate import audit_nonapple_safe_rejection
row=json.loads((Path(__file__).parent/'stage70_case09_observed.json').read_text(encoding='utf8'))
good=audit_nonapple_safe_rejection(row)
assert good['pass'],good
assert good['predicted_object']=='baseball'  # exact species NOT corrected to tennis
# No bypass possible for unsafe movement / false acceptance.
for changes in [
 {'descent_executed':True}, {'gripper_closed':True}, {'assigned_nest':'A3'},
 {'status':'PLACED_ON_SUPPORT'}, {'stage32_gate':{'allow_descent':True}},
 {'stage32_gate':{'object':{'accepted':True,'predicted_object':'apple'}}},
 {'classification':{'status':'PREDICTED','class':'A'}}
]:
  bad=copy.deepcopy(row)
  for k,v in changes.items():
    if isinstance(v,dict) and isinstance(bad.get(k),dict):bad[k].update(v)
    else:bad[k]=v
  result=audit_nonapple_safe_rejection(bad)
  assert not result['pass'],(changes,result)
print('STAGE70_SAFE_NONAPPLE_GATE_REGRESSION_OK; real Stage69 tennis was safely rejected; 7 unsafe mutations rejected')
