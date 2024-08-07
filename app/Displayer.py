import numpy as np
from PyQt5.QtWidgets import QLabel, QVBoxLayout, QWidget
from PyQt5.QtGui import QImage, QPixmap

class Displayer(QWidget):
    def __init__(self, frame_reciever):
        super().__init__()
        self.frame_receiver = frame_reciever
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('RabbitMQ Video Stream')
        self.image_label = QLabel(self)
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.image_label)
        self.setLayout(self.layout)
        
        self.frame_receiver.new_frame.connect(self.update_image)
        self.frame_receiver.start()

    def update_image(self, frame):
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        self.image_label.setPixmap(QPixmap.fromImage(q_img))

