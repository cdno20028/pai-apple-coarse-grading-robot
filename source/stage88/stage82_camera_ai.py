# -*- coding:utf-8 -*-
"""Stage82 inherits EXACT Stage68/81 camera capture and physical kinematics.
Only the colour-conserving pole UV mapping is updated by Stage82Surface.
"""
from stage68_camera_ai import Stage68CameraAI
from stage82_pole_surface import Stage82Surface
class Stage82CameraAI(Stage68CameraAI):
    def __init__(self,root,results,assets,object_ai):
        super().__init__(root,results,assets,object_ai)
        self.repair=Stage82Surface(assets,results,self.root/'stage79_normal_pole_reference')
