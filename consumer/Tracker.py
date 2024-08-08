from Configuration import Configuration as CONFIG
from Visualizer import Visualizer
from Compostion import Composition

import numpy as np
from ultralytics import YOLO
import cv2
import random
import os
import json
import torch
import copy

class Tracker(Composition):
    def __init__(self):
        self._tracker_model = YOLO(CONFIG.PATH_TO_DETECTION_CKPT)
        self._color_holder = {}
        self._parse_scene_composition()
        self._visualizer = Visualizer()

        self._box_ids_history = None
        self._speed_stats = None

    def _get_filled_cells(self, bounding_boxes):
        device = bounding_boxes.device
        rectangles = torch.tensor(self._car_cells, device=device)  # Shape: [num_rectangles, 2, 2]

        
        rect_top_left = rectangles[:, 0]  # Shape: [num_rectangles, 2]
        rect_bottom_right = rectangles[:, 1]  # Shape: [num_rectangles, 2]
        
        rect_x_min = rect_top_left[:, 0]  # Shape: [num_rectangles]
        rect_y_min = rect_top_left[:, 1]  # Shape: [num_rectangles]
        rect_x_max = rect_bottom_right[:, 0]  # Shape: [num_rectangles]
        rect_y_max = rect_bottom_right[:, 1]  # Shape: [num_rectangles]

        x_min = bounding_boxes[:, 0]  # Shape: [num_bounding_boxes]
        y_min = bounding_boxes[:, 1]  # Shape: [num_bounding_boxes]
        x_max = bounding_boxes[:, 2]  # Shape: [num_bounding_boxes]
        y_max = bounding_boxes[:, 3]  # Shape: [num_bounding_boxes]

        x_center = (x_min + x_max) / 2  # Shape: [num_bounding_boxes]
        y_center = (y_min + y_max) / 2  # Shape: [num_bounding_boxes]

        center_inside_x = (x_center.unsqueeze(0) >= rect_x_min.unsqueeze(1)) & (x_center.unsqueeze(0) <= rect_x_max.unsqueeze(1))  # Shape: [num_rectangles, num_bounding_boxes]
        center_inside_y = (y_center.unsqueeze(0) >= rect_y_min.unsqueeze(1)) & (y_center.unsqueeze(0) <= rect_y_max.unsqueeze(1))  # Shape: [num_rectangles, num_bounding_boxes]
        
        center_inside = center_inside_x & center_inside_y 

        true_indices = torch.nonzero(center_inside, as_tuple=False)
        
        rows = true_indices[:, 0]

        return rows
    
    def _get_filled_cells__for_each_zone(self, bounding_boxes):
        result = {}
        for zone_name, zone_polygon in self._zones.items():

            device = bounding_boxes.device
            rectangles = torch.tensor([zone_polygon], device=device)  # Shape: [num_rectangles, 2, 2]

            
            rect_top_left = rectangles[:, 0]  # Shape: [num_rectangles, 2]
            rect_bottom_right = rectangles[:, 1]  # Shape: [num_rectangles, 2]
            
            rect_x_min = rect_top_left[:, 0]  # Shape: [num_rectangles]
            rect_y_min = rect_top_left[:, 1]  # Shape: [num_rectangles]
            rect_x_max = rect_bottom_right[:, 0]  # Shape: [num_rectangles]
            rect_y_max = rect_bottom_right[:, 1]  # Shape: [num_rectangles]

            x_min = bounding_boxes[:, 0]  # Shape: [num_bounding_boxes]
            y_min = bounding_boxes[:, 1]  # Shape: [num_bounding_boxes]
            x_max = bounding_boxes[:, 2]  # Shape: [num_bounding_boxes]
            y_max = bounding_boxes[:, 3]  # Shape: [num_bounding_boxes]

            x_center = (x_min + x_max) / 2  # Shape: [num_bounding_boxes]
            y_center = (y_min + y_max) / 2  # Shape: [num_bounding_boxes]

            center_inside_x = (x_center.unsqueeze(0) >= rect_x_min.unsqueeze(1)) & (x_center.unsqueeze(0) <= rect_x_max.unsqueeze(1))  # Shape: [num_rectangles, num_bounding_boxes]
            center_inside_y = (y_center.unsqueeze(0) >= rect_y_min.unsqueeze(1)) & (y_center.unsqueeze(0) <= rect_y_max.unsqueeze(1))  # Shape: [num_rectangles, num_bounding_boxes]
            
            center_inside = center_inside_x & center_inside_y 

            true_indices = torch.nonzero(center_inside, as_tuple=False)
            
            rows = true_indices[:, 0]
            num_cars_inside_zone = len(rows)

            result[zone_name] = num_cars_inside_zone
        
        print(result)
        return result
    
    def _calculate_speed(self):
        if self._box_ids_history is not None and self._current_box_ids is not None:
            self._box_ids_history = self._box_ids_history.cuda()
            self._current_box_ids = self._current_box_ids.cuda()
            
            history_ids = self._box_ids_history[:, 0].long()
            current_ids = self._current_box_ids[:, 0].long()
            
            history_boxes = self._box_ids_history[:, 1:]
            current_boxes = self._current_box_ids[:, 1:]
            
            mutual_ids = torch.tensor(np.intersect1d(history_ids.cpu().numpy(), current_ids.cpu().numpy())).to(self._box_ids_history.device)
            
            if mutual_ids.numel() == 0:
                print("No mutual IDs found")
                return torch.tensor([]).cuda()
            
            history_idx = torch.nonzero(history_ids[:, None] == mutual_ids, as_tuple=False)[:, 0]
            current_idx = torch.nonzero(current_ids[:, None] == mutual_ids, as_tuple=False)[:, 0]
            
            history_mutual_boxes = history_boxes[history_idx]
            current_mutual_boxes = current_boxes[current_idx]
            
            if history_mutual_boxes.shape[0] == 0 or current_mutual_boxes.shape[0] == 0:
                print("No mutual boxes found after indexing")
                return torch.tensor([]).cuda()
            
            history_centers = (history_mutual_boxes[:, :2] + history_mutual_boxes[:, 2:]) / 2.0
            current_centers = (current_mutual_boxes[:, :2] + current_mutual_boxes[:, 2:]) / 2.0
            
            distances = torch.norm(history_centers - current_centers, dim=1)
            result = torch.stack((mutual_ids.float(), distances), dim=1)

            distances = result[:, 1]
            updated_velocities = torch.where(distances > 1, (distances * CONFIG.PIX_TO_KM_RATE) / (9.258333333e-6), 0)
            result[:, 1] = updated_velocities
            result = result.to("cpu").numpy().tolist()
            
            return {int(k[0]):round(k[1], 2) for k in result}
        else:
            return {}

            


    def _track_cars(self, frame):
        results = self._tracker_model.track(frame, persist=True,device=CONFIG.DEVICE,tracker="bytetrack.yaml" ,conf=CONFIG.TRACKER_CONFIDENCE, iou=CONFIG.TRACKER_IOU, verbose=False)

        # frame = self._visualizer.draw_zones(frame)

        self._boxes = results[0].boxes.xyxy.to(CONFIG.DEVICE)
        self._confs = results[0].boxes.conf.cpu().numpy()
        self._ids = results[0].boxes.id.to(CONFIG.DEVICE)
        self._current_box_ids = torch.cat((self._ids.unsqueeze(1), self._boxes), dim=1)

        
        self._filled_cells_indixes = self._get_filled_cells(self._boxes)
        self._filled_cells_stats_for_each_zone = self._get_filled_cells__for_each_zone(self._boxes)
        self._visualizer.draw_cells(frame, self._filled_cells_indixes)


        self._speed_stats = self._calculate_speed()

        self._box_ids_history = self._current_box_ids.clone()
        
        for box, car_id, conf in zip(self._boxes, self._ids, self._confs):

            
            if car_id not in self._color_holder.keys():
                self._color_holder[car_id] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            
            
            speed = self._speed_stats.get(int(car_id.item()), None)
            self._visualizer.draw_single_car(frame, box, car_id, self._color_holder, speed)

        return frame


    def __call__(self, frame):
        frame = self._track_cars(frame)
        info = {
            "number_of_cars": len(self._boxes),
            "number_of_cars_in_zones": self._filled_cells_stats_for_each_zone,
            "number_of_filled_cells": len(self._filled_cells_indixes),
            "number_empty_cells": len(self._car_cells) - len(self._filled_cells_indixes),
            "number_of_car_cells": len(self._car_cells)

        }
        return frame, info



        