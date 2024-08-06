import cv2
import pika
import msgpack
import numpy as np
from flask import Flask
from flask_socketio import SocketIO, emit
from flask_cors import CORS
app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# RabbitMQ configuration
rabbitmq_config = {
    'host': 'localhost',
    'port': 5672,
    'username': 'guest',
    'password': 'guest',
    'virtual_host': '/'
}

def convert_image_to_bytes(frame):
    ret, buffer = cv2.imencode('.jpg', frame)
    image_bytes = buffer.tobytes()
    return image_bytes



def consume_frames():
    credentials = pika.PlainCredentials(rabbitmq_config['username'], rabbitmq_config['password'])
    parameters = pika.ConnectionParameters(
        host=rabbitmq_config['host'],
        port=rabbitmq_config['port'],
        virtual_host=rabbitmq_config['virtual_host'],
        credentials=credentials
    )

    connection = pika.BlockingConnection(parameters)
    channel = connection.channel()
    channel.queue_declare(queue='parking-consumed')

    def callback(ch, method, properties, body):
        # try:
            message = msgpack.unpackb(body, raw=False)
            image_bytes = message['image']
            metadata = message['metadata']
            
            frame = convert_bytes_to_image(image_bytes)
            
            _, buffer = cv2.imencode('.jpg', frame)
            frame_data = buffer.tobytes()
            socketio.emit('frame', {'image': frame_data, 'metadata': metadata})
        # except Exception as e:
        #     print(f"Failed to process message: {e}")

    channel.basic_consume(queue='parking-consumed', on_message_callback=callback, auto_ack=True)
    channel.start_consuming()

def convert_bytes_to_image(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return frame

# Example usage:
if __name__ == '__main__':
    # Start consuming images (this will run indefinitely)
    from threading import Thread
    consumer_thread = Thread(target=consume_frames)
    consumer_thread.start()
    socketio.run(app, host='0.0.0.0', port=5000)
