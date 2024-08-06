from typing import Any
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from Configuration import Configuration as CONFIG
from bytetracker.core import ByteTrack
import numpy as np
from ultralytics import YOLO
from deep_sort import DeepSort
import cv2

COLORS_10 =[(144,238,144),(178, 34, 34),(221,160,221),(  0,255,  0),(  0,128,  0),(210,105, 30),(220, 20, 60),
            (192,192,192),(255,228,196),( 50,205, 50),(139,  0,139),(100,149,237),(138, 43,226),(238,130,238),
            (255,  0,255),(  0,100,  0),(127,255,  0),(255,  0,255),(  0,  0,205),(255,140,  0),(255,239,213),
            (199, 21,133),(124,252,  0),(147,112,219),(106, 90,205),(176,196,222),( 65,105,225),(173,255, 47),
            (255, 20,147),(219,112,147),(186, 85,211),(199, 21,133),(148,  0,211),(255, 99, 71),(144,238,144),
            (255,255,  0),(230,230,250),(  0,  0,255),(128,128,  0),(189,183,107),(255,255,224),(128,128,128),
            (105,105,105),( 64,224,208),(205,133, 63),(  0,128,128),( 72,209,204),(139, 69, 19),(255,245,238),
            (250,240,230),(152,251,152),(  0,255,255),(135,206,235),(  0,191,255),(176,224,230),(  0,250,154),
            (245,255,250),(240,230,140),(245,222,179),(  0,139,139),(143,188,143),(255,  0,  0),(240,128,128),
            (102,205,170),( 60,179,113),( 46,139, 87),(165, 42, 42),(178, 34, 34),(175,238,238),(255,248,220),
            (218,165, 32),(255,250,240),(253,245,230),(244,164, 96),(210,105, 30)]

def draw_boxes(img, bbox, identities=None, offset=(0,0)):

        for i,box in enumerate(bbox):
            x1,y1,x2,y2 = [int(i) for i in box]
            x1 += offset[0]
            x2 += offset[0]
            y1 += offset[1]
            y2 += offset[1]
            # box text and bar
            id = int(identities[i]) if identities is not None else 0    
            color = COLORS_10[id%len(COLORS_10)]
            label = '{}{:d}'.format("", id)
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2 , 2)[0]
            cv2.rectangle(img,(x1, y1),(x2,y2),color,3)
            # cv2.rectangle(img,(x1, y1),(x1+t_size[0]+3,y1+t_size[1]+4), color,-1)
            # cv2.putText(img,label,(x1,y1+t_size[1]+4), cv2.FONT_HERSHEY_PLAIN, 2, [255,255,255], 2)

        return img

class Tracker:
    def __init__(self):
        # self._detection_model = AutoDetectionModel.from_pretrained(
        #     model_type='yolov8',
        #     model_path=CONFIG.PATH_TO_DETECTION_CKPT,
        #     confidence_threshold=CONFIG.SAHI_THRESHOLD,
        #     device="cuda:0",
            
        # )
        # if CONFIG.TRACKER_NAME == "bytetrack":
        #     self._tracker = ByteTrack()

        self._tracker_model = YOLO(CONFIG.PATH_TO_DETECTION_CKPT)
        # self.deepsort_ckpt_path = r"C:\Users\ASUS\Desktop\github_projects\Parking\ckpt.t7"
        # self.deepsort = DeepSort(self.deepsort_ckpt_path, use_cuda="cuda")


        self._color_holder = {}


    def _track_cars(self, frame):
        results = self._tracker_model.track(frame, persist=True,device=CONFIG.DEVICE,tracker="bytetrack.yaml" ,conf=CONFIG.TRACKER_CONFIDENCE, iou=CONFIG.TRACKER_IOU, verbose=False)
        boxes = results[0].boxes.xyxy.to(CONFIG.DEVICE)
        confs = results[0].boxes.conf.cpu().numpy()
        # classes = results[0].boxes.cls.cpu().numpy().astype(int)

        if results[0].boxes.id is None:
            ids = [None for _ in range(len(boxes))]
        else:
            ids = results[0].boxes.id.cpu().numpy().astype(int)

        import random
        

        for box, car_id, conf in zip(boxes, ids, confs):
            if car_id not in self._color_holder.keys():
                self._color_holder[car_id] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

            cv2.rectangle(frame, (int(box[0].item()), int(box[1].item())), (int(box[2].item()), int(box[3].item())), self._color_holder[car_id], 2)

        return frame


    def __call__(self, frame):
        frame = self._track_cars(frame)
        info = {
            "number_of_cars": 200
        }
        return frame, info
        




        