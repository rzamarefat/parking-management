import numpy as np
from PyQt5.QtWidgets import QLabel, QVBoxLayout, QWidget,QTextEdit
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot

class Displayer(QWidget):
    def __init__(self, frame_reciever):
        super().__init__()
        self.frame_receiver = frame_reciever
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('RabbitMQ Video Stream')
        self.image_label = QLabel(self)
        self.metadata_text = QTextEdit(self)
        self.metadata_text.setReadOnly(True)
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.image_label)
        self.layout.addWidget(self.metadata_text)
        self.setLayout(self.layout)
        self.frame_receiver.new_frame.connect(self.update_image)
        self.frame_receiver.start()

    @pyqtSlot(np.ndarray, dict)
    def update_image(self, frame, metadata):
        print("Updating image and metadata...")  # Debug print
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        self.image_label.setPixmap(QPixmap.fromImage(q_img))
        self.metadata_text.setText(str(metadata))

