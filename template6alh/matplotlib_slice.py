from typing import Protocol, Generator, TypeAlias
from dataclasses import dataclass
from logging import getLogger
from pathlib import Path

import numpy as np
import nrrd
from PyQt5.QtWidgets import (
    QMainWindow,
    QSlider,
    QVBoxLayout,
    QWidget,
    QApplication,
    QLabel,
)
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backend_bases import MouseEvent
from scipy import ndimage as ndi

from .utils import get_spacings

logger = getLogger(__name__)

Coords: TypeAlias = tuple[int, float, float]


@dataclass
class CoordsSet:
    """
    3 coordinates to define the orientation of the VNC
    """

    brain: Coords
    "based in pixels"
    sez: Coords
    "based in pixels"
    tip: Coords
    "based in pixels"
    scale: np.ndarray

    def to_cmtk(self):
        """
        returns a string that can be written to disk
        """
        lines = [f"{z} {y} {x} {name}" for name, (z, y, x) in self.to_dict().items()]
        return "\n".join(lines)

    def to_array(self) -> np.ndarray:
        return np.stack([list(v) for v in self.to_dict().values()])

    def to_dict(self) -> dict[str, Coords]:
        unscaled_dict = {"brain": self.brain, "sez": self.sez, "tip": self.tip}
        return {
            k: tuple((np.array(list(v)) * self.scale).tolist())
            for k, v in unscaled_dict.items()
        }


class CoordsCallback(Protocol):
    def __call__(self, coords_set: CoordsSet):
        ...


def proc_image(
    image: np.ndarray, gamma: float | None, scale: tuple[float, float, float] | None
) -> np.ndarray:
    """
    processes the image for easier visualization
    """
    image = np.rint((image.astype(float) / (255 * image.max()))).astype(np.uint8)
    if scale is not None:
        scale_array = np.array(scale).clip(0, 1)
        logger.info("setting scale %s", scale_array)
        image = ndi.zoom(image, scale_array)
    if gamma is not None:
        # look up table implementation based on _adjust_gamma_u8 from skimage.exposure
        lut = np.rint(255 * (np.linspace(0, 1, 256) ** gamma)).astype(np.uint8)
        image = lut[image]
    return image


class ImageSlicer(QMainWindow):
    def __init__(
        self,
        volume: np.ndarray,
        app: QApplication,
        click_generator: Generator[str, Coords, None] | None,
        gamma: float | None = None,
        scale: tuple[float, float, float] | None = None,
    ):
        """
        a click_generator yeilds titles when a click occurs
        gamma is the gamma correction
        scale is the fraction scale to use
        """
        super().__init__()
        # set volume dynamic range
        self.volume = proc_image(volume, gamma, scale)
        self.volume = (volume * (254 / np.max(volume))).astype(np.uint8)
        self.app = app
        self.current_slice = 0
        self.click_generator = click_generator
        # Create main widget and layout
        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout(self.main_widget)
        self.message = QLabel("test")
        if click_generator is not None:
            self.layout.addWidget(self.message)
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
        # Connect click callback
        if self.click_generator is not None:
            message = next(self.click_generator)
            self.message.setText(message)
            self.figure.canvas.mpl_connect("button_press_event", self.on_click)  # type: ignore

        # Initialize plot
        self.update_slice(0)

    def on_click(self, event: MouseEvent):
        assert self.click_generator is not None
        if event.inaxes is not None:
            y_coord = event.ydata
            assert y_coord is not None
            x_coord = event.xdata
            assert x_coord is not None
            try:
                self.message.setText(
                    self.click_generator.send((self.current_slice, y_coord, x_coord))
                )
            except StopIteration:
                self.close()

    def update_slice(self, slice_idx: int):
        self.current_slice = slice_idx
        self.ax.clear()
        self.ax.imshow(
            self.volume[slice_idx, :, :],
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


def get_slicer(
    volume: np.ndarray,
    title: str,
    gamma: float | None,
    scale: tuple[float, float, float] | None,
) -> ImageSlicer:
    app = QApplication([])
    slicer = ImageSlicer(volume, app, None, gamma=gamma, scale=scale)
    slicer.resize(800, 600)
    slicer.setWindowTitle(title)
    slicer.show()
    return slicer


def coords_generator(callback: CoordsCallback, scale: np.ndarray):
    brain = yield "Click mid brain lobes"
    assert brain is not None
    logger.info(",".join(str(c) for c in brain))
    sez = yield "Click SEZ"
    assert sez is not None
    logger.info(",".join(str(c) for c in sez))
    tip = yield "Click posterior VNC"
    assert tip is not None
    logger.info(",".join(str(c) for c in tip))
    callback(CoordsSet(brain=brain, sez=sez, tip=tip, scale=scale))


def write_landmarks(in_path: Path, out_path: Path) -> ImageSlicer:
    """
    opens the file in a viewer, prompts landmarks then writes those landmarks
    to disk as a txt file
    """

    def callback(coords_set: CoordsSet):
        out_path.write_text(coords_set.to_cmtk())

    data, md = nrrd.read(str(in_path))
    app = QApplication([])
    slicer = ImageSlicer(data, app, coords_generator(callback, get_spacings(md)))
    slicer.resize(800, 600)
    slicer.setWindowTitle(str(in_path))
    slicer.show()
    return slicer
