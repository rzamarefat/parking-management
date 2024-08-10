import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QLabel, QVBoxLayout, QWidget, QScrollArea, QFormLayout, QHBoxLayout, QPushButton
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import pyqtSlot

class Displayer(QWidget):
    def __init__(self, frame_receiver):
        super().__init__()
        self.frame_receiver = frame_receiver
        self.initUI()
        self.showFullScreen()  # Make the window full screen
        self.setFixedSize(self.size())  # Disable resizing

    def initUI(self):
        self.setWindowTitle('RabbitMQ Video Stream')
        
        self.image_label = QLabel(self)
        self.metadata_layout = QFormLayout()
        self.metadata_widget = QWidget()
        self.metadata_widget.setLayout(self.metadata_layout)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.metadata_widget)

        self.image_layout = QVBoxLayout()
        self.image_layout.addWidget(self.image_label)
        self.image_layout.addWidget(self.scroll_area)

        self.cell_chart = self._create_chart(0, 0)

        self.charts_layout = QVBoxLayout()
        self.charts_layout.addWidget(self.cell_chart)

        self.exit_button = QPushButton('Exit', self)
        self.exit_button.clicked.connect(self.close_application)

        self.sidebar_layout = QVBoxLayout()
        self.sidebar_layout.addLayout(self.charts_layout)
        self.sidebar_layout.addWidget(self.exit_button)

        self.main_layout = QHBoxLayout()
        self.main_layout.addLayout(self.image_layout)
        self.main_layout.addLayout(self.sidebar_layout)

        self.setLayout(self.main_layout)

        self.frame_receiver.new_frame.connect(self.update_image)
        self.frame_receiver.start()

    def _create_chart(self, total, part):
        fig, ax = plt.subplots(figsize=(4, 4))

        categories = ['Part', 'Remaining']
        values = [part, total - part]
        
        ax.bar(categories, values, color=['#ff9999', '#66b3ff'])

        ax.set_xlabel('Category')
        ax.set_ylabel('Value')
        ax.set_title('Bar Chart of Part and Remaining')

        for i, value in enumerate(values):
            ax.text(i, value + 0.01 * total, f'{value}', ha='center')

        canvas = FigureCanvas(fig)
        return canvas

    def _update_bar_chart(self, canvas, total, part):
        canvas.figure.clf()
        ax = canvas.figure.add_subplot(111)
        categories = ['Filled', 'Empty']
        values = [part, total - part]
        ax.bar(categories, values, color=['#FF0000', '#00ff00'])
        ax.set_xlabel('Number')
        ax.set_title('Car Cells Status')
        for i, value in enumerate(values):
            ax.text(i, value + 0.01 * total, f'{value}', ha='center')

        canvas.draw()

    def close_application(self):
        self.close()

    @pyqtSlot(np.ndarray, dict)
    def update_image(self, frame, metadata):
        try:
            print("Updating image and metadata...")
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
            self.image_label.setPixmap(QPixmap.fromImage(q_img))

            for i in reversed(range(self.metadata_layout.count())):
                self.metadata_layout.itemAt(i).widget().setParent(None)

            for key, value in metadata.items():
                if key == "number_of_cars_in_zones":
                    for zone_name, actual_value in value.items():
                        key_label = QLabel(f"<b>{zone_name}:</b>")
                        value_label = QLabel(str(actual_value))
                        self.metadata_layout.addRow(key_label, value_label)
                else:
                    key = key.replace("_", " ")
                    key = " ".join([k.upper() for k in key.split(" ")])
                    key_label = QLabel(f"<b>{key}:</b>")
                    value_label = QLabel(str(value))
                    self.metadata_layout.addRow(key_label, value_label)

            self._update_bar_chart(self.cell_chart, metadata["number_of_car_cells"], metadata["number_of_filled_cells"])

        except Exception as e:
            print(f"Failed to update image: {e}")

