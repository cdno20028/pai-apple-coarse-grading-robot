# Stage70: explicit conservative evaluation for NON-APPLE reject
# Source: observed Stage69 case09 tennis-ball PyBullet evidence.
# Scope is not exact ball-species classification.
def audit_nonapple_safe_rejection(row):
    gate=(row or {}).get('stage32_gate') or {}
    obj=gate.get('object') or {}
    predicted=obj.get('predicted_object')
    checks={
      'load_triggered': (gate.get('load') or {}).get('triggered') is True,
      'palm_image_captured':gate.get('upper_image_captured') is True,
      'gate_rejected':gate.get('allow_descent') is False,
      'object_not_accepted':obj.get('accepted') is False,
      'nonapple_predicted':predicted in ('baseball','red_ball','tennis'),
      'explicit_nonapple_status':obj.get('status')=='NOT_APPLE',
      'no_physical_descent':row.get('descent_executed') is False,
      'fingers_never_closed':row.get('gripper_closed') is False,
      'nothing_transported':row.get('assigned_nest') is None,
      'no_grading':(row.get('classification') or {}).get('status')=='NOT_STARTED',
      'stopped_before_descent':row.get('status')=='STAGE32_REJECTED_BEFORE_DESCENT',
    }
    return {
      'pass':all(checks.values()),
      'checks':checks,
      'predicted_object':predicted,
      'confidence':obj.get('confidence'),
      'apple_score':(obj.get('scores') or {}).get('apple'),
      'species_prediction_not_equated_with_nonapple_rejection':True,
      'classification_reference_not_passed_to_model':True,
    }
