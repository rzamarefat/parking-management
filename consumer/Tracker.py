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
        if self._box_ids_history is not None and self._current_box_ids is not None:
            print("self._box_ids_history", self._box_ids_history.shape)
            print("self._current_box_ids", self._current_box_ids.shape)

            # Ensure the tensors are on CUDA
            self._box_ids_history = self._box_ids_history.cuda()
            self._current_box_ids = self._current_box_ids.cuda()
            
            # Extract IDs and bounding box coordinates
            history_ids = self._box_ids_history[:, 0].long()
            current_ids = self._current_box_ids[:, 0].long()
            
            history_boxes = self._box_ids_history[:, 1:]
            current_boxes = self._current_box_ids[:, 1:]
            
            # Find mutual IDs using numpy and move back to CUDA
            mutual_ids = torch.tensor(np.intersect1d(history_ids.cpu().numpy(), current_ids.cpu().numpy())).to(self._box_ids_history.device)
            
            if mutual_ids.numel() == 0:
                print("No mutual IDs found")
                return torch.tensor([]).cuda()
            
            # Get indices of mutual IDs in history and current
            history_idx = torch.nonzero(history_ids[:, None] == mutual_ids, as_tuple=False)[:, 0]
            current_idx = torch.nonzero(current_ids[:, None] == mutual_ids, as_tuple=False)[:, 0]
            
            # Filter the boxes for mutual IDs
            history_mutual_boxes = history_boxes[history_idx]
            current_mutual_boxes = current_boxes[current_idx]
            
            if history_mutual_boxes.shape[0] == 0 or current_mutual_boxes.shape[0] == 0:
                print("No mutual boxes found after indexing")
                return torch.tensor([]).cuda()
            
            # Calculate centers
            history_centers = (history_mutual_boxes[:, :2] + history_mutual_boxes[:, 2:]) / 2.0
            current_centers = (current_mutual_boxes[:, :2] + current_mutual_boxes[:, 2:]) / 2.0
            
            # Compute distances
            distances = torch.norm(history_centers - current_centers, dim=1)
            
            # Combine mutual IDs and distances into the result tensor
            result = torch.stack((mutual_ids.float(), distances), dim=1)

            distances = result[:, 1]
            updated_distances = torch.where(distances > 0.5, distances * 6, 0)
            
            # Update the result tensor
            result[:, 1] = updated_distances

            print(result.to("cpu").numpy().tolist())
            
            return result

            


    def _track_cars(self, frame):
        results = self._tracker_model.track(frame, persist=True,device=CONFIG.DEVICE,tracker="bytetrack.yaml" ,conf=CONFIG.TRACKER_CONFIDENCE, iou=CONFIG.TRACKER_IOU, verbose=False)

        # frame = self._visualizer.draw_zones(frame)

        self._boxes = results[0].boxes.xyxy.to(CONFIG.DEVICE)
        self._confs = results[0].boxes.conf.cpu().numpy()
        self._ids = results[0].boxes.id.to(CONFIG.DEVICE)
        self._current_box_ids = torch.cat((self._ids.unsqueeze(1), self._boxes), dim=1)

        
        self._filled_cells_indixes = self._get_filled_cells(self._boxes)
        self._visualizer.draw_cells(frame, self._filled_cells_indixes)


        self._calculate_speed()

        self._box_ids_history = self._current_box_ids.clone()
        
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



        