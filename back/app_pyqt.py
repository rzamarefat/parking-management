import sys
import cv2
import pika
import msgpack
import numpy as np
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QThread, pyqtSignal

# RabbitMQ configuration
rabbitmq_config = {
    'host': 'localhost',
    'port': 5672,
    'username': 'guest',
    'password': 'guest',
    'virtual_host': '/'
}

class FrameReceiver(QThread):
    new_frame = pyqtSignal(np.ndarray)

    def run(self):
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
            try:
                print("Received message:", body[:100], "...")  # Debug print
                message = msgpack.unpackb(body, raw=False)
                image_bytes = message['image']
                frame = self.convert_bytes_to_image(image_bytes)
                
                # Resize the frame to 1/3 of its original width and height
                frame_resized = self.resize_frame(frame)
                
                self.new_frame.emit(frame_resized)
            except Exception as e:
                print(f"Failed to process message: {e}")

        channel.basic_consume(queue='parking-consumed', on_message_callback=callback, auto_ack=True)
        channel.start_consuming()

    def convert_bytes_to_image(self, image_bytes):
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return frame

    def resize_frame(self, frame):
        # Resize frame to 1/3 of its original width and height
        height, width = frame.shape[:2]
        new_width = width // 3
        new_height = height // 3
        resized_frame = cv2.resize(frame, (new_width, new_height))
        return resized_frame

class VideoDisplay(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('RabbitMQ Video Stream')
        self.image_label = QLabel(self)
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.image_label)
        self.setLayout(self.layout)
        self.frame_receiver = FrameReceiver()
        self.frame_receiver.new_frame.connect(self.update_image)
        self.frame_receiver.start()

    def update_image(self, frame):
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        self.image_label.setPixmap(QPixmap.fromImage(q_img))

def main():
    app = QApplication(sys.argv)
    display = VideoDisplay()
    display.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
