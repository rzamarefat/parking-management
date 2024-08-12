import os
import json
import cv2
import numpy as np
from Compostion import Composition


class ColorPallete:
    ZONES = {
            'z1': (255, 131, 67),
            'z12': (234, 56, 78),
            'z2': (23, 155, 174),
            'z3': (65, 88, 166),
            'z4': (239, 90, 111),
            'z5': (212, 189, 172),
            'z6': (67, 89, 23),
            'z7': (123, 234, 45),
            'z8': (0, 123, 255),
            'z9': (123, 0, 255),
            'z14': (89, 67, 45),
            'z11': (255, 123, 0),
            'z10': (12, 255, 78),
            'z13': (34, 56, 78),
            'pz2': (200, 200, 0),
            'pz1': (159, 159, 159),
            'zz': (255, 2, 255),
            "nothing": (0,0,0)
        }

class Visualizer(Composition):
    def __init__(self):
        super().__init__()
        pass
        

    def draw_cells(self, frame, filled_cells_indixes):
        overlay = frame.copy()
    
        for index, rect in enumerate(self._car_cells):
            top_left = (int(rect[0][0]), int(rect[0][1]))
            bottom_right = (int(rect[1][0]), int(rect[1][1]))
            if index in filled_cells_indixes:
                color = (0, 0, 255)
            else:
                color = (0, 255, 0)

            cv2.rectangle(overlay, top_left, bottom_right, color, -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.8, 0, frame)
        
        return frame

    def draw_zones(self, frame):
        for k, v in self._zones.items():
            opacity=0.2
            overlay = frame.copy()
            points = np.array(v, np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(overlay, [points], ColorPallete.ZONES[k])
            cv2.polylines(overlay, [points], isClosed=True, color=ColorPallete.ZONES[k], thickness=2)
            cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0, frame)

        return overlay
    
    def draw_single_car(self, frame, box, car_id, color_holder, speed, cars_with_bad_parking_style):
        if speed is None:
            speed_txt = "N"
        else:
            speed_txt = str(speed)

        if int(car_id.item()) not in cars_with_bad_parking_style:
            frame = cv2.rectangle(frame, (int(box[0].item()), int(box[1].item())), (int(box[2].item()), int(box[3].item())), (0,0,0), 2)
        else:
            frame = cv2.rectangle(frame, (int(box[0].item()), int(box[1].item())), (int(box[2].item()), int(box[3].item())), (0,0,255), thickness=cv2.FILLED)

        if speed != 0.0:
            cv2.putText(frame, speed_txt, (int(box[0].item()), int(box[1].item())), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)

        return frame

if __name__ == "__main__":
    vis = Visualizer()
    vis(cv2.imread(r"C:\Users\ASUS\Desktop\github_projects\Parking\important_images\0045.png"))