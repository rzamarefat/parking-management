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
        self._boxes_history = None
        self._ids_history = None

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
    
    def _calculate_speed(self):
        if self._boxes_history is not None and self._ids_history is not None:
            ids_diff = [item for item in self._ids_history if item not in self._ids]
            extra_ids = [item for item in self._ids if item not in self._ids_history]

            # print("++++++++++++++++++++++++++++")
            # print("ids_diff",ids_diff)
            # print("extra_ids",extra_ids)
            

            mask = torch.ones(self._boxes_history.size(0), dtype=torch.bool)
            mask[ids_diff] = False
            filtered_boxes_from_history = self._boxes_history[mask]
            filtered_boxes_from_history = filtered_boxes_from_history.cuda()

            mask = torch.ones(self._boxes.size(0), dtype=torch.bool)
            mask[extra_ids] = False
            filtered_boxes_current = self._boxes[mask]

            
            # print("filtered_boxes_from_history.shape", filtered_boxes_from_history.shape)
            # print("self._boxes.shape", self._boxes.shape)
            # print("++++++++++++++++++++++++++++")
            
            centers1 = (filtered_boxes_from_history[:, :2] + filtered_boxes_from_history[:, 2:]) / 2.0
            centers2 = (filtered_boxes_current[:, :2] + filtered_boxes_current[:, 2:]) / 2.0
            
            distances = torch.norm(centers1 - centers2, dim=1)
            print(distances)

            


    def _track_cars(self, frame):
        results = self._tracker_model.track(frame, persist=True,device=CONFIG.DEVICE,tracker="bytetrack.yaml" ,conf=CONFIG.TRACKER_CONFIDENCE, iou=CONFIG.TRACKER_IOU, verbose=False)

        # frame = self._visualizer.draw_zones(frame)

        self._boxes = results[0].boxes.xyxy.to(CONFIG.DEVICE)
        self._confs = results[0].boxes.conf.cpu().numpy()

        if results[0].boxes.id is None:
            self._ids = [None for _ in range(len(self._boxes))]
        else:
            self._ids = results[0].boxes.id.cpu().numpy().astype(int)

        
        self._filled_cells_indixes = self._get_filled_cells(self._boxes)
        self._visualizer.draw_cells(frame, self._filled_cells_indixes)


        self._calculate_speed()


        self._boxes_history = self._boxes.clone()
        self._ids_history = copy.deepcopy(self._ids)
        
        for box, car_id, conf in zip(self._boxes, self._ids, self._confs):

            
            if car_id not in self._color_holder.keys():
                self._color_holder[car_id] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

            self._visualizer.draw_single_car(frame, box, car_id, self._color_holder)

        return frame


    def __call__(self, frame):
        frame = self._track_cars(frame)
        info = {
            "number_of_cars": len(self._boxes),
            "number_of_cars_in_zones": {
                "z1": 20,
                "z10":  10,

            },

            "number_of_filled_cells": len(self._filled_cells_indixes),
            "number_empty_cells": len(self._car_cells) - len(self._filled_cells_indixes),
            "number_of_car_cells": len(self._car_cells)

        }
        return frame, info



        