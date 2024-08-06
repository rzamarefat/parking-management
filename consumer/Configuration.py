import os
import gdown
import torch

class Configuration:
    _root = os.getcwd()
    _root_to_save_ckpts = os.path.join(_root, "ckpts")

    os.makedirs(_root_to_save_ckpts, exist_ok=True)

    # PATH_TO_DETECTION_CKPT = os.path.join(_root_to_save_ckpts, "top_view_car_det.pt")
    PATH_TO_DETECTION_CKPT = r"C:\Users\ASUS\Desktop\github_projects\Parking\runs\detect\train\weights\best.pt"

    # if not(os.path.isfile(PATH_TO_DETECTION_CKPT)):
    #     print("Downloading the car detection module ...")
    #     gdown.download(
    #             "https://drive.google.com/uc?id=1CJpsBzhHw9klxBham9lE6XJ6LmYuf6u8",
    #             PATH_TO_DETECTION_CKPT,
    #             quiet=False
    #         )
        
    SAHI_THRESHOLD = 0.5

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    TRACKER_NAME = "bytetrack"


    TRACKER_CONFIDENCE = 0.5
    TRACKER_IOU = 0.5 