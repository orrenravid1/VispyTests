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

# ──────────────── Worker Thread for NEURON ────────────────
class NeuronWorker(QtCore.QThread):
    # emits (time_array, voltage_array) each step
    data_ready = QtCore.pyqtSignal(object, object, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._running = True

    def run(self):
        steps_per_emit = 5     # 10 * dt=0.1ms → emit every 1 ms
        count = 0
        # ──────────────── Set up a trivial NEURON model ────────────────
        # Single “soma” section with Hodgkin–Huxley channels
        soma = h.Section(name='soma')
        soma.L = soma.diam = 20
        soma.insert('hh')

        # Add an IClamp at the center of the soma
        icl = h.IClamp(soma(0.5))
        icl.delay = 2
        icl.dur   = 5
        icl.amp   = 0.5    # amplitude in nA

        # Record time and voltage at the middle of the section
        v_vec = h.Vector().record(soma(0.5)._ref_v)
        t_vec = h.Vector().record(h._ref_t)
        i_vec = h.Vector().record(icl._ref_i)

        h.dt = 0.00005     # NEURON time step (ms)
        h.finitialize(-65)  # start at −65 mV
        while self._running:
            h.fadvance()
            count += 1
            if count >= steps_per_emit:
                count = 0
                T = np.array(t_vec)
                V = np.array(v_vec)
                I = np.array(i_vec)
                self.data_ready.emit(T, V, I)
            # yield to OS so the UI thread can run
            self.msleep(0)

    def stop(self):
        self._running = False

# ──────────────── Build the Vispy “Tube” scene ────────────────
# Mock “morphology” path (here just a straight line)
N = 50
points = np.vstack([
    np.linspace(0, 100, N),
    np.zeros(N),
    np.zeros(N),
]).T
base_radius = 2.0

canvas = scene.SceneCanvas(keys='interactive', bgcolor='white', show=False)
view   = canvas.central_widget.add_view()
view.camera = 'turntable'

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

# ──────────────── Main Window & Hookup ────────────────
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Vispy + NEURON (Threaded) Demo")
        self.setCentralWidget(canvas.native)
        self.resize(800, 600)

        # Start the NEURON worker
        self.worker = NeuronWorker(self)
        self.worker.data_ready.connect(self.on_data)
        self.worker.start()

    def on_data(self, T, V, I):
        if T[-1] > 1000:
            self.worker.stop()
        #print(f"T = {T[-1]}, V = {V[-1]}, I={I[-1]}")
        norm = np.clip((V[-1] + 80) / 130, 0.0, 1.0)
        r, b = norm, 1.0 - norm
        color_filter.filter = (r, 0.2, b, 1.0)
        canvas.update()

    def closeEvent(self, ev):
        # stop the worker thread cleanly
        self.worker.stop()
        self.worker.wait()
        super().closeEvent(ev)

# ──────────────── Run ────────────────
if __name__ == '__main__':
    win = MainWindow()
    win.show()
    app.run()
