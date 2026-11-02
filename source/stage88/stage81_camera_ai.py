# -*- coding: utf-8 -*-
"""Stage81: only the per-object visible texture changes; all original cameras,
AI, object gate, physics, controls, and strict grading semantics are inherited.
"""
from stage68_camera_ai import Stage68CameraAI
from stage81_pole_surface import Stage81Surface
class Stage81CameraAI(Stage68CameraAI):
    def __init__(self,root,results,assets,object_ai):
        super().__init__(root,results,assets,object_ai)
        self.repair=Stage81Surface(assets,results,self.root/'stage79_normal_pole_reference')
