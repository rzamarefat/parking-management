import os
import gdown
import torch

class Configuration:
    RABBIT_CREDENTIALS_USERNAME = "guest"
    RABBIT_CREDENTIALS_PASSWORD = "guest"
    RABBIT_IP = "127.0.0.1"
    QUEUE_NAME = "parking-producer"

    DB_NAME = "Parking" 
    DB_USER = "postgres"
    DB_PASSWORD = "postgres"
    DB_HOST = "localhost"
    DB_PORT = "5432"
    DB_TABLE_NAME = "public.parking"