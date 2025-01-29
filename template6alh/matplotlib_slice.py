import numpy as np
from PyQt5.QtWidgets import QMainWindow, QSlider, QVBoxLayout, QWidget, QApplication
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas


class ImageSlicer(QMainWindow):
    def __init__(self, volume: np.ndarray, app: QApplication):
        super().__init__()
        # set volume dynamic range
        self.volume = (volume * (254 / np.max(volume))).astype(np.uint8)
        self.app = app
        self.current_slice = 0
        # Create main widget and layout
        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout(self.main_widget)
        # Create matplotlib canvas
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)
        # Create slicer
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(volume.shape[0] - 1)
        self.slider.setValue(0)
        self.slider.valueChanged.connect(self.update_slice)
        self.layout.addWidget(self.slider)
        # Initialize plot
        self.update_slice(0)

    def update_slice(self, slice_idx: int):
        self.current_slice = slice_idx
        self.ax.clear()
        self.ax.imshow(
            self.volume[slice_idx, :, :].T,
            cmap="gray",
            aspect="equal",
            vmin=0,
            vmax=255,
        )
        self.ax.set_title(f"Slice {slice_idx}")
        self.ax.axis("off")
        self.canvas.draw()

    def quit(self):
        # causes segfault
        self.app.quit()


def get_slicer(volume: np.ndarray, title: str) -> ImageSlicer:
    app = QApplication([])
    slicer = ImageSlicer(volume, app)
    slicer.resize(800, 600)
    slicer.setWindowTitle(title)
    slicer.show()
    return slicer
