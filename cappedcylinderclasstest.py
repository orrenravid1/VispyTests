import numpy as np
from vispy import scene, app, use
from vispy.scene.visuals import InstancedMesh
from vispy.scene.cameras import TurntableCamera

use(gl='gl+')

class CappedCylinderCollection(scene.Node):
    # Static geometry (built once)
    _side_vertices = None
    _side_indices  = None
    _disk_vertices = None
    _disk_indices  = None

    def __init__(self, parent=None, cylinder_segments=32, disk_slices=32):
        super().__init__(parent=parent)

        # 1) Build side‐only cylinder mesh if needed
        if CappedCylinderCollection._side_vertices is None:
            angles = np.linspace(0.0,
                                 2.0 * np.pi,
                                 cylinder_segments,
                                 endpoint=False,
                                 dtype=np.float32)
            circle_pts = np.stack([np.cos(angles),
                                   np.sin(angles)], axis=1)
            bottom_ring = np.column_stack([circle_pts,
                                           -0.5 * np.ones(cylinder_segments,
                                                          dtype=np.float32)])
            top_ring    = np.column_stack([circle_pts,
                                           +0.5 * np.ones(cylinder_segments,
                                                          dtype=np.float32)])
            CappedCylinderCollection._side_vertices = np.vstack([bottom_ring,
                                                                  top_ring])

            side_idx = []
            for i in range(cylinder_segments):
                ni = (i + 1) % cylinder_segments
                side_idx.append([i,
                                 i + cylinder_segments,
                                 ni])
                side_idx.append([ni,
                                 i + cylinder_segments,
                                 ni + cylinder_segments])
            CappedCylinderCollection._side_indices = np.array(side_idx,
                                                              dtype=np.uint32)

        # 2) Build disk‐cap mesh if needed
        if CappedCylinderCollection._disk_vertices is None:
            angles = np.linspace(0.0,
                                 2.0 * np.pi,
                                 disk_slices,
                                 endpoint=False,
                                 dtype=np.float32)
            circle_pts = np.stack([np.cos(angles),
                                   np.sin(angles)], axis=1)
            verts2d = np.vstack([[0.0, 0.0], circle_pts])
            verts3d = np.zeros((verts2d.shape[0], 3),
                               dtype=np.float32)
            verts3d[:, :2] = verts2d
            CappedCylinderCollection._disk_vertices = verts3d

            disk_idx = []
            for i in range(1, disk_slices):
                disk_idx.append([0, i, i+1])
            disk_idx.append([0, disk_slices, 1])
            CappedCylinderCollection._disk_indices = np.array(disk_idx,
                                                              dtype=np.uint32)

        # 3) Create the two InstancedMesh visuals with EMPTY instance arrays
        empty_positions = np.zeros((0, 3), dtype=np.float32)
        empty_transforms = np.zeros((0, 3, 3), dtype=np.float32)

        self._side_mesh = InstancedMesh(
            CappedCylinderCollection._side_vertices,
            CappedCylinderCollection._side_indices,
            instance_positions=empty_positions,
            instance_transforms=empty_transforms,
            parent=self
        )
        self._disk_mesh = InstancedMesh(
            CappedCylinderCollection._disk_vertices,
            CappedCylinderCollection._disk_indices,
            instance_positions=empty_positions,
            instance_transforms=empty_transforms,
            parent=self
        )

        self._instances = []

    def add_cylinder(self, position, radius, height, color):
        """
        position: (x,y,z)
        radius:   float
        height:   float
        color:    array-like RGBA
        """
        self._instances.append((
            np.array(position, dtype=np.float32),
            float(radius),
            float(height),
            np.array(color, dtype=np.float32),
        ))
        self._refresh_instances()

    def _refresh_instances(self):
        count = len(self._instances)

        # — Side instances —
        side_positions = np.vstack([inst[0] for inst in self._instances])
        radii_arr      = np.array([inst[1] for inst in self._instances],
                                  dtype=np.float32)
        heights_arr    = np.array([inst[2] for inst in self._instances],
                                  dtype=np.float32)
        side_colors    = np.vstack([inst[3] for inst in self._instances])

        side_transforms = np.repeat(np.eye(3, dtype=np.float32)[None, ...],
                                    count, axis=0)
        side_transforms[:, 0, 0] = radii_arr
        side_transforms[:, 1, 1] = radii_arr
        side_transforms[:, 2, 2] = heights_arr

        self._side_mesh.instance_positions  = side_positions
        self._side_mesh.instance_transforms = side_transforms
        self._side_mesh.instance_colors     = side_colors
        self._side_mesh.update()

        # — Cap instances (two per cylinder) —
        cap_positions = np.zeros((2*count, 3), dtype=np.float32)
        cap_transforms = np.repeat(np.eye(3, dtype=np.float32)[None, ...],
                                   2*count, axis=0)
        cap_colors = np.zeros((2*count, 4), dtype=np.float32)

        for idx, (pos, rad, ht, col) in enumerate(self._instances):
            bi, ti = 2*idx, 2*idx+1
            cap_positions[bi] = pos + np.array([0, 0, -0.5 * ht],
                                               dtype=np.float32)
            cap_positions[ti] = pos + np.array([0, 0, +0.5 * ht],
                                               dtype=np.float32)

            cap_transforms[bi] = np.diag([rad, rad, 1.0])
            cap_transforms[ti] = np.diag([rad, rad, 1.0])

            cap_colors[bi] = col
            cap_colors[ti] = col

        self._disk_mesh.instance_positions  = cap_positions
        self._disk_mesh.instance_transforms = cap_transforms
        self._disk_mesh.instance_colors     = cap_colors
        self._disk_mesh.update()


# — Usage Example —
canvas = scene.SceneCanvas(keys='interactive', show=True, bgcolor='black')
view   = canvas.central_widget.add_view()
view.camera = TurntableCamera(fov=45,
                              distance=20,
                              elevation=30,
                              azimuth=30,
                              up='+z')

collection = CappedCylinderCollection(parent=view.scene)
for _ in range(50):
    pos = np.random.uniform(-5, 5, 3)
    pos[2] = 0
    collection.add_cylinder(
        position=pos,
        radius=np.random.uniform(0.3, 1.0),
        height=np.random.uniform(1.0, 3.0),
        color=np.append(np.random.uniform(0.2, 1.0, 3), 0.8)
    )

if __name__ == '__main__':
    app.run()
