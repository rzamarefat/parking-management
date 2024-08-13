from RabbitPublisher import RabbitPublisher
from Configuration import Configuration as CONFIG
import cv2
from DatabaseHandler import DatabaseHandler 
from datetime import datetime
import base64

class Producer:
    def __init__(self):
        self._rabbit_publisher = RabbitPublisher(CONFIG.QUEUE_NAME)
        self._rabbit_publisher.start()

        self._db_handler = DatabaseHandler()
    
    @staticmethod
    def _encode_img_to_base64(frame):
        _, buffer = cv2.imencode('.jpg', frame)
        image_bytes = buffer.tobytes()
        base64_encoded = base64.b64encode(image_bytes).decode('utf-8')

        return base64_encoded

    def __call__(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Erroor reading file")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            timestamp = datetime.today().strftime('%Y-%m-%d')
            last_index = self._db_handler.get_last_not_analyzed_index(timestamp=timestamp)

            data_to_publish = {
                "img":self._encode_img_to_base64(frame),
                "metadata": f"{timestamp}__{last_index+1}"
            }
            
            self._rabbit_publisher.publish(data_to_publish)
            print("Published ...")
            if last_index == -1:
                self._db_handler.push_frame_to_db(index=0, timestamp=timestamp)
            else:
                self._db_handler.push_frame_to_db(index=last_index+1, timestamp=timestamp)

        cap.release()
        cv2.destroyAllWindows()