from RabbitPublisher import RabbitPublisher
from Configuration import Configuration as CONFIG
import cv2
import pika

class Producer:
    def __init__(self):
        self._rabbit_publisher = RabbitPublisher(CONFIG.QUEUE_NAME)
        self._rabbit_publisher.start()

    @staticmethod
    def _convert_image_to_bytes(frame):
        ret, buffer = cv2.imencode('.jpg', frame)
        image_bytes = buffer.tobytes()
        return image_bytes


    def __call__(self, video_path):
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print("Erroor reading file")

        
        frame_count = 0
        while True:
            
            ret, frame = cap.read()
                
            if not ret:
                break
                
            self._rabbit_publisher.publish(self._convert_image_to_bytes(frame))

            frame_count += 1

        cap.release()
        cv2.destroyAllWindows()