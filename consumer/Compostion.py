import os
import json
import numpy as np

class Composition:
    def __init__(self,):
        super().__init__()
        self._parse_scene_composition()

    @staticmethod
    def _get_top_bottom_points(coordinates):
        p1, p2 = coordinates
        x1, y1 = p1
        x2, y2 = p2
        if x1 > x2 and y1 < y2:
            adjusted_rectangle = [[x2, y1], [x1, y2]]
        elif x1 < x2 and y1 < y2:
            adjusted_rectangle = coordinates
        elif x1 > x2 and y1>y2:
            adjusted_rectangle = [[x2, y2], [x1, y1]]
        
        return adjusted_rectangle


    def _parse_scene_composition(self):
        path_to_config = os.path.join(os.getcwd(), "scene_composition.json")
        with open(path_to_config, 'r') as file:
            info = json.load(file)
        

        self._img_height = info["imageHeight"]
        self._img_width = info["imageWidth"]

        self._zones = {shape["label"]:self._get_top_bottom_points(shape["points"])  for shape in info["shapes"] if shape["label"].__contains__("z")}
        self._car_cells = [self._get_top_bottom_points(shape["points"])  for shape in info["shapes"] if shape["label"] == "car_cell"]


