import os
import json
import cv2
import numpy as np

class ColorPallete:
    ZONES = {
            'z1': (123, 45, 67),
            'z12': (234, 56, 78),
            'z2': (12, 89, 200),
            'z3': (45, 12, 67),
            'z4': (200, 100, 50),
            'z5': (55, 155, 255),
            'z6': (67, 89, 23),
            'z7': (123, 234, 45),
            'z8': (0, 123, 255),
            'z9': (123, 0, 255),
            'z14': (89, 67, 45),
            'z11': (255, 123, 0),
            'z10': (12, 255, 78),
            'z13': (34, 56, 78),
            'pz2': (78, 123, 255),
            'pz1': (200, 45, 123),
            'zz': (55, 200, 123)
        }

class Visualizer:
    def __init__(self):
        self._parse_scene_composition()
    
    def _parse_scene_composition(self):
        path_to_config = os.path.join(os.getcwd(), "scene_composition.json")
        with open(path_to_config, 'r') as file:
            info = json.load(file)

        self._img_height = info["imageHeight"]
        self._img_width = info["imageWidth"]

        self._zones = {shape["label"]:shape["points"]  for shape in info["shapes"] if shape["label"].__contains__("z")}
        print(self._zones)

    def _draw_zones(self, frame):
        for k, v in self._zones.items():
            print(v)
            opacity=0.2
            overlay = frame.copy()
            points = np.array(v, np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(overlay, [points], ColorPallete.ZONES[k])
            cv2.polylines(overlay, [points], isClosed=True, color=ColorPallete.ZONES[k], thickness=2)
            cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0, frame)

        return overlay

    def __call__(self, frame):
        frame = self._draw_zones(frame)

        cv2.imwrite("frame.png", frame)

if __name__ == "__main__":
    vis = Visualizer()
    vis(cv2.imread(r"C:\Users\ASUS\Desktop\github_projects\Parking\important_images\0045.png"))