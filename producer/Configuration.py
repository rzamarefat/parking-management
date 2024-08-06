import os
import gdown
import torch

class Configuration:
    RABBIT_CREDENTIALS_USERNAME = "guest"
    RABBIT_CREDENTIALS_PASSWORD = "guest"
    RABBIT_IP = "127.0.0.1"
    QUEUE_NAME = "parking-producer"