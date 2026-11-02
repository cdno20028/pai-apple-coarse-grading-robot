# -*- coding:utf-8 -*-
"""Same five cameras, same Stage82 C/D branch and stage80 physical registry.
Only visually rendered (therefore physically captured) UV changed.
"""
from stage68_camera_ai import Stage68CameraAI
from stage83_pole_surface import Stage83Surface
class Stage83CameraAI(Stage68CameraAI):
    def __init__(self,root,results,assets,object_ai):
        super().__init__(root,results,assets,object_ai)
        self.repair=Stage83Surface(assets,results,self.root/'stage79_normal_pole_reference')
