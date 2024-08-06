from RabbitPublisher import RabbitPublisher
from Configuration import Configuration as CONFIG
import cv2
import pika

class Producer:
    def __init__(self):
        self._rabbit_publisher = RabbitPublisher(CONFIG.QUEUE_NAME)
        self._rabbit_publisher.start()
        self._rabbitmq_config = {
                'host': 'localhost',
                'port': 5672,
                'username': 'guest',
                'password': 'guest',
                'virtual_host': '/'
            }

    @staticmethod
    def _convert_image_to_bytes(frame):
        ret, buffer = cv2.imencode('.jpg', frame)
        image_bytes = buffer.tobytes()
        return image_bytes
    
    

    def publish_image(self, image_bytes):
        credentials = pika.PlainCredentials(self._rabbitmq_config['username'], self._rabbitmq_config['password'])
        parameters = pika.ConnectionParameters(
            host=self._rabbitmq_config['host'],
            port=self._rabbitmq_config['port'],
            virtual_host=self._rabbitmq_config['virtual_host'],
            credentials=credentials
        )
        connection = pika.BlockingConnection(parameters)
        channel = connection.channel()
        channel.queue_declare(queue='video_frames')
        channel.basic_publish(exchange='', routing_key='video_frames', body=image_bytes)
        connection.close()


    def __call__(self, video_path):
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print("Erroor reading file")

        
        frame_count = 0
        while True:
            print(frame_count)
            ret, frame = cap.read()

            if not ret:
                break
                
            self._rabbit_publisher.publish(self._convert_image_to_bytes(frame))

            frame_count += 1

        cap.release()
        cv2.destroyAllWindows()