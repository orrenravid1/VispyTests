import sys
import numpy as np

# 1) Create the Qt application first
from PyQt6 import QtWidgets
qtapp = QtWidgets.QApplication(sys.argv)

# 2) Tell Vispy to use PyQt6 **and** request a gl+ context for instanced rendering
from vispy import use
use(app='pyqt6', gl='gl+')

# 3) Now import the rest of Vispy (it will pick up the above settings)
from vispy import app, scene
from vispy.geometry import create_cylinder
from scipy.spatial.transform import Rotation

# --- mock-data setup as before ---
N = 30
radius = 0.05
t = np.linspace(0, 4*np.pi, N+1)
points = np.vstack([np.sin(t), np.zeros_like(t), np.cos(t)]).T

segments = []
for i in range(N):
    p0, p1 = points[i], points[i+1]
    d0 = d1 = radius*2
    segments.append((p0, p1, d0, d1))

instance_positions  = np.zeros((N, 3), np.float32)
instance_transforms = np.zeros((N, 3, 3), np.float32)

for k, (p0, p1, d0, d1) in enumerate(segments):
    v = p1 - p0
    L = np.linalg.norm(v)
    R = 0.5*(d0 + d1)
    z = np.array([0.0, 0.0, 1.0])
    v_norm = v / L
    axis = np.cross(z, v_norm)
    if np.linalg.norm(axis) < 1e-6:
        Rmat = np.eye(3)
    else:
        angle = np.arccos(np.dot(z, v_norm))
        axis /= np.linalg.norm(axis)
        Rmat = Rotation.from_rotvec(axis * angle).as_matrix()
    S = np.diag([R, R, L])
    instance_transforms[k] = Rmat @ S
    instance_positions[k]  = (p0 + p1) / 2

# --- build the Vispy scene ---
canvas = scene.SceneCanvas(keys='interactive', bgcolor='white', show=False)
view   = canvas.central_widget.add_view()
view.camera = 'turntable'

cyl = create_cylinder(20, 20, radius=(1.0, 1.0), length=1.0)
vertices = cyl.get_vertices()
faces    = cyl.get_faces()

mesh = scene.visuals.InstancedMesh(
    vertices=vertices,
    faces=faces,
    instance_positions=instance_positions,
    instance_transforms=instance_transforms,
    parent=view.scene,
    color=(0.3, 0.5, 0.8, 1.0)
)

# --- embed in a QMainWindow ---
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Vispy + NEURON Mock Demo")
        self.setCentralWidget(canvas.native)
        self.resize(800, 600)

if __name__ == '__main__':
    win = MainWindow()
    win.show()
    # 4) Now start Vispy’s (and thus Qt’s) event loop
    app.run()
