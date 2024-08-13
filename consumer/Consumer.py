from typing import Any
from Tracker import Tracker
import pika
from Configuration import Configuration as CONFIG
import numpy as np
import cv2
import base64
import json
import msgpack
from DatabaseHandler import DatabaseHandler

class Consumer:
    def __init__(self):
        # try:
        self._tracker_handler = Tracker()
        # except Exception as e:
        #     print(e)
        
        self._connection = pika.BlockingConnection(pika.ConnectionParameters(host="127.0.0.1"))
        self._channel = self._connection.channel()
        self._channel.queue_declare(queue=CONFIG.PRODUCER_QUEUE_NAME)


        self._database_handler = DatabaseHandler()

    @staticmethod
    def _convert_image_to_bytes(frame):
        ret, buffer = cv2.imencode('.jpg', frame)
        image_bytes = buffer.tobytes()
        return image_bytes


    def _publish_image_with_metadata(self, frame, metadata):
        image_bytes = self._convert_image_to_bytes(frame)
    
        payload = {
            'image': image_bytes,
            'metadata': metadata
        }
    
        payload_json = msgpack.packb(payload, use_bin_type=True)

        credentials = pika.PlainCredentials(CONFIG.RABBIT_CREDENTIALS_USERNAME, CONFIG.RABBIT_CREDENTIALS_PASSWORD)
        parameters = pika.ConnectionParameters(
            host=CONFIG.RABBIT_HOST,
            port=CONFIG.RABBIT_PORT,
            virtual_host="/",
            credentials=credentials
        )

        connection = pika.BlockingConnection(parameters)
        channel = connection.channel()

        channel.queue_declare(queue=CONFIG.CONSUMED_QUEUE_NAME)

        channel.basic_publish(exchange='', routing_key=CONFIG.CONSUMED_QUEUE_NAME, body=payload_json)
        connection.close()

    @staticmethod
    def _convert_bytes_to_image(image_bytes):
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return frame
    
    @staticmethod
    def _convert_base64_to_img(base64_img):
        return base64.b64decode(base64_img)
        
    def __call__(self, frame=None):
        # frame, info = self._tracker_handler(frame)
        # return frame, info

        while True:
            def callback(ch, method, properties, body):
                fetched_data = json.loads(body.decode('utf-8'))
                frame = base64.b64decode(fetched_data['img'])
                
                frame = np.frombuffer(frame, dtype=np.uint8)
                
                frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)

                metadata = fetched_data['metadata']
                timestamp = metadata.split("__")[0]
                index = int(metadata.split("__")[1])
                
                frame, info = self._tracker_handler(frame)
                if index == 0:
                    self._publish_image_with_metadata(frame, info)
                    self._database_handler.update_frame_stats(index=index, timestamp=timestamp, stat=info)
                    print("Successfull 1")
                else:
                    is_prev_frame_analyzed = self._database_handler.is_there_the_frame(index=str(index - 1), timestamp=timestamp)
                    prev_frame_found = is_prev_frame_analyzed
                    print(index)
                    while not(prev_frame_found):
                        is_prev_frame_analyzed = self._database_handler.is_there_the_frame(index=str(index - 1), timestamp=timestamp)

                        if is_prev_frame_analyzed:
                            prev_frame_found = True

                            self._publish_image_with_metadata(frame, info)
                            self._database_handler.update_frame_stats(index=index, timestamp=timestamp, stat=info)
                            print("Successfull 2")
                
                print("Published Successfully")

            self._channel.basic_consume(queue=CONFIG.PRODUCER_QUEUE_NAME, on_message_callback=callback, auto_ack=True)

            print(' [*] Waiting for messages. To exit press CTRL+C')
            self._channel.start_consuming()
