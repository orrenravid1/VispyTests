import sys
import numpy as np

from PyQt6 import QtWidgets
from vispy import use, app, scene

# 1) Instantiate QApplication before any visuals
qtapp = QtWidgets.QApplication(sys.argv)

# 2) Configure Vispy for PyQt6 + GL+ (optional for instancing support)
use(app='pyqt6', gl='gl+')

# 3) Mock “morphology” path: a sine wave in Y vs. X
N = 200
t = np.linspace(0, 4*np.pi, N)
points = np.vstack([t, np.sin(t)*0.5, np.zeros_like(t)]).T
radius = 0.05  # constant tube radius

# 4) Build the Vispy scene
canvas = scene.SceneCanvas(keys='interactive', bgcolor='white', show=False)
view   = canvas.central_widget.add_view()
view.camera = 'turntable'

# 5) Create a single Tube visual, with correct params
tube = scene.visuals.Tube(
    points=points,           # (N, 3) path coordinates
    radius=radius,           # float or (N,) array
    tube_points=20,          # cross-section resolution (vs. radial_segments) :contentReference[oaicite:5]{index=5}
    closed=False,            # leave end-caps on :contentReference[oaicite:6]{index=6}
    shading='smooth',        # 'flat' or 'smooth' shading :contentReference[oaicite:7]{index=7}
    color=(0.3, 0.5, 0.8, 1.0),  # uniform RGBA color :contentReference[oaicite:8]{index=8}
    parent=view.scene
)

# 6) Embed into a PyQt6 QMainWindow
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Vispy Tube Demo")
        self.setCentralWidget(canvas.native)
        self.resize(800, 600)

if __name__ == '__main__':
    win = MainWindow()
    win.show()
    app.run()  # starts the Vispy/Qt event loop
