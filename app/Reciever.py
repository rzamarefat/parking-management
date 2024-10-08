import sys
import cv2
import pika
import msgpack
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from Configuration import Configuration as CONFIG


class Receiver(QThread):
    new_frame = pyqtSignal(np.ndarray, dict)

    def run(self):
        credentials = pika.PlainCredentials(CONFIG.RABBIT_USERNAME, CONFIG.RABBIT_PASSWORD)
        parameters = pika.ConnectionParameters(
            host=CONFIG.RABBIT_HOST,
            port=CONFIG.RABBIT_PORT,
            virtual_host=CONFIG.RABBIT_VIRTUAL_HOST,
            credentials=credentials
        )

        connection = pika.BlockingConnection(parameters)
        channel = connection.channel()
        channel.queue_declare(queue='parking-consumed')

        def callback(ch, method, properties, body):
            try:
                message = msgpack.unpackb(body, raw=False)
                image_bytes = message['image']
                metadata = message['metadata']  # Extract metadata
                frame = self.convert_bytes_to_image(image_bytes)
                
                # Resize the frame to 1/3 of its original width and height
                frame_resized = self.resize_frame(frame)
                
                self.new_frame.emit(frame_resized, metadata)  # Emit frame and metadata
            except Exception as e:
                print(f"Failed to process message: {e}")

        channel.basic_consume(queue='parking-consumed', on_message_callback=callback, auto_ack=True)
        channel.start_consuming()

    def convert_bytes_to_image(self, image_bytes):
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return frame

    def resize_frame(self, frame):
        height, width = frame.shape[:2]
        new_width = width // 3
        new_height = height // 3
        resized_frame = cv2.resize(frame, (new_width, new_height))
        return resized_frame
