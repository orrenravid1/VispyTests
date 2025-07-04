import sys
import numpy as np

# 1) Start the Qt app before creating any visuals/widgets
from PyQt6 import QtWidgets, QtCore
qtapp = QtWidgets.QApplication(sys.argv)

# 2) Configure Vispy
from vispy import use
from vispy.visuals.filters import ColorFilter
use(app='pyqt6', gl='gl+')

# 3) Imports for Vispy and NEURON
from vispy import app, scene
from vispy.scene.visuals import Tube
from neuron import h

# PyQtGraph imports
import pyqtgraph as pg

# 3) Build your Vispy SceneCanvas
canvas3d = scene.SceneCanvas(keys='interactive', bgcolor='white', show=False)
view = canvas3d.central_widget.add_view()
view.camera = 'turntable'

# ──────────────── Build the Vispy “Tube” scene ────────────────
# Mock “morphology” path (here just a straight line)
N = 50
points = np.vstack([
    np.linspace(0, 100, N),
    np.zeros(N),
    np.zeros(N),
]).T
base_radius = 2.0

tube = Tube(
    points=points,
    radius=base_radius,
    tube_points=16,
    closed=False,
    shading='smooth',
    color=(0.3, 0.5, 0.8, 1.0),
    parent=view.scene
)

color_filter = ColorFilter(filter=(0.3, 0.5, 0.8, 1.0))
tube.attach(color_filter)

# ─── Build the PyQtGraph 2D plot ──────────────────────────────
plot2d = pg.PlotWidget(title="Membrane Voltage (mV)")
plot2d.setLabel('bottom', 'Time', units='ms')
plot2d.setLabel('left', 'Voltage', units='mV')
plot2d.setBackground('w')
line2d = plot2d.plot(pen='b')

# 5) Lay them out side by side
main_widget = QtWidgets.QWidget()
hbox = QtWidgets.QHBoxLayout(main_widget)
hbox.addWidget(canvas3d.native)   # Vispy widget
hbox.addWidget(plot2d)           # PyQtGraph canvas
main_widget.setLayout(hbox)

# ──────────────── Worker Thread for NEURON ────────────────
class NeuronWorker(QtCore.QThread):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.running = True

        self.t = 0.0
        self.v = -65

    def run(self):
        # ──────────────── Set up a trivial NEURON model ────────────────
        # Single “soma” section with Hodgkin–Huxley channels
        soma = h.Section(name='soma')
        soma.L = soma.diam = 20
        soma.insert('hh')

        # Add an IClamp at the center of the soma
        icl = h.IClamp(soma(0.5))
        icl.delay = 2
        icl.dur   = 30
        icl.amp   = 0.5    # amplitude in nA

        # Record time and voltage at the middle of the section
        v_vec = h.Vector().record(soma(0.5)._ref_v)
        t_vec = h.Vector().record(h._ref_t)
        i_vec = h.Vector().record(icl._ref_i)

        h.dt = 0.00001     # NEURON time step (ms)
        h.finitialize(-65)  # start at −65 mV
        while self.running and self.t < 1000:
            h.fadvance()
            self.t = float(t_vec[-1])
            self.v = float(v_vec[-1])
            # yield to OS so the UI thread can run
            self.msleep(0)

    def stop(self):
        self.running = False

# ──────────────── Main Window & Hookup ────────────────
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D + Plot Side by Side")
        self.setCentralWidget(main_widget)
        self.resize(1200, 600)

        # Start the NEURON worker
        self.sim = NeuronWorker(self)
        self.sim.start()

        self.trace_t = []
        self.trace_V = []

        self.ui_timer = QtCore.QTimer(self)
        self.ui_timer.timeout.connect(self.update_frame)
        self.ui_timer.start((int(1000/60)))

    def update_frame(self):
        t = self.sim.t
        v = self.sim.v

        norm = np.clip((v + 80) / 130, 0, 1)
        r,b = norm, 1.0 - norm
        color_filter.filter = (r, 0.2, b, 1.0)
        canvas3d.update()

        self.trace_t.append(t)
        self.trace_V.append(v)

        if len(self.trace_t) > 5000:
            self.trace_t.pop(0)
            self.trace_V.pop(0)
        line2d.setData(self.trace_t, self.trace_V)

        if not self.sim.running:
            self.ui_timer.stop()

    def closeEvent(self, ev):
        self.ui_timer.stop()

        # stop the worker thread cleanly
        self.sim.stop()
        self.sim.wait()

        super().closeEvent(ev)

# ──────────────── Run ────────────────
if __name__ == '__main__':
    win = MainWindow()
    win.show()
    app.run()
