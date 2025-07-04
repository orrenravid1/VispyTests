import numpy as np
from vispy import scene, use
from vispy.scene.visuals import InstancedMesh
from scipy.spatial.transform import Rotation

# Enable instancing
use(gl='gl+')

class CappedCylinderCollection(scene.Node):
    """
    A deferred collection of capped cylinders. Mesh visuals are created on first refresh,
    then updated on subsequent refresh() calls following refresh-instanced semantics.
    """
    # Shared geometry, built once
    _side_vertices = None
    _side_indices  = None
    _disk_vertices = None
    _disk_indices  = None

    def __init__(self, parent=None, cylinder_segments=32, disk_slices=32):
        super().__init__(parent=parent)
        # Queue of instances: (pos, radius, height, color, orientation)
        self._instances = []
        # Placeholder for visuals, created in refresh()
        self._side_mesh = None
        self._disk_mesh = None
        # Store parameters for geometry generation
        self._segs = cylinder_segments
        self._slices = disk_slices

    def add_cylinder(self, position, radius, height, color, orientation=None):
        """
        Queue a single capped cylinder instance. Call refresh() to build/update visuals.
        """
        pos = np.array(position, dtype=np.float32)
        col = np.array(color, dtype=np.float32)
        ori = (np.array(orientation, dtype=np.float32)
               if orientation is not None else np.eye(3, dtype=np.float32))
        self._instances.append((pos, float(radius), float(height), col, ori))

    def refresh(self):
        """
        Build or update the instanced meshes in one batch. Creates visuals lazily.
        """
        M = len(self._instances)
        if M == 0:
            return
        # Generate shared geometry if needed
        if CappedCylinderCollection._side_vertices is None:
            angles = np.linspace(0.0, 2.0 * np.pi, self._segs,
                                 endpoint=False, dtype=np.float32)
            circle_pts = np.stack([np.cos(angles), np.sin(angles)], axis=1)
            bottom = np.column_stack([circle_pts,
                                      -0.5 * np.ones(self._segs)])
            top    = np.column_stack([circle_pts,
                                      +0.5 * np.ones(self._segs)])
            CappedCylinderCollection._side_vertices = np.vstack([bottom, top]).astype(np.float32)
            idx = []
            for i in range(self._segs):
                ni = (i + 1) % self._segs
                idx.append([i, i + self._segs, ni])
                idx.append([ni, i + self._segs, ni + self._segs])
            CappedCylinderCollection._side_indices = np.array(idx, dtype=np.uint32)
        if CappedCylinderCollection._disk_vertices is None:
            angles = np.linspace(0.0, 2.0 * np.pi, self._slices,
                                 endpoint=False, dtype=np.float32)
            circle_pts = np.stack([np.cos(angles), np.sin(angles)], axis=1)
            verts2d = np.vstack([[0.0, 0.0], circle_pts])
            verts3d = np.zeros((verts2d.shape[0], 3), dtype=np.float32)
            verts3d[:, :2] = verts2d
            CappedCylinderCollection._disk_vertices = verts3d
            idx = [[0, i, i + 1] for i in range(1, self._slices)]
            idx.append([0, self._slices, 1])
            CappedCylinderCollection._disk_indices = np.array(idx, dtype=np.uint32)
        # Build instance arrays
        side_pos = np.vstack([inst[0] for inst in self._instances])
        radii    = np.array([inst[1] for inst in self._instances], dtype=np.float32)
        heights  = np.array([inst[2] for inst in self._instances], dtype=np.float32)
        colors   = np.vstack([inst[3] for inst in self._instances])
        oris     = [inst[4] for inst in self._instances]
        # Transforms for sides
        side_xf = np.zeros((M, 3, 3), dtype=np.float32)
        for i in range(M):
            scale = np.diag([radii[i], radii[i], heights[i]]).astype(np.float32)
            side_xf[i] = oris[i] @ scale
        # Lazy creation of visuals
        if self._side_mesh is None:
            self._side_mesh = InstancedMesh(
                CappedCylinderCollection._side_vertices,
                CappedCylinderCollection._side_indices,
                instance_positions=side_pos,
                instance_transforms=side_xf,
                instance_colors=colors,
                parent=self
            )
        else:
            self._side_mesh.instance_positions  = side_pos
            self._side_mesh.instance_transforms = side_xf
            self._side_mesh.instance_colors     = colors
            self._side_mesh.update()
        # Caps: 2*M instances
        cap_pos = np.zeros((2*M, 3), dtype=np.float32)
        cap_xf  = np.zeros((2*M, 3, 3), dtype=np.float32)
        cap_col = np.zeros((2*M, 4), dtype=np.float32)
        for i, (pos, rad, ht, col, ori) in enumerate(self._instances):
            bi, ti = 2*i, 2*i + 1
            bottom = ori @ np.array([0, 0, -0.5 * ht], dtype=np.float32)
            top    = ori @ np.array([0, 0, +0.5 * ht], dtype=np.float32)
            cap_pos[bi] = pos + bottom
            cap_pos[ti] = pos + top
            scale2 = np.diag([rad, rad, 1.0]).astype(np.float32)
            cap_xf[bi] = ori @ scale2
            cap_xf[ti] = ori @ scale2
            cap_col[bi] = col
            cap_col[ti] = col
        if self._disk_mesh is None:
            self._disk_mesh = InstancedMesh(
                CappedCylinderCollection._disk_vertices,
                CappedCylinderCollection._disk_indices,
                instance_positions=cap_pos,
                instance_transforms=cap_xf,
                instance_colors=cap_col,
                parent=self
            )
        else:
            self._disk_mesh.instance_positions  = cap_pos
            self._disk_mesh.instance_transforms = cap_xf
            self._disk_mesh.instance_colors     = cap_col
            self._disk_mesh.update()


# Simple test when run as a script
if __name__ == '__main__':
    from vispy.scene.cameras import TurntableCamera
    from vispy import app

    canvas = scene.SceneCanvas(keys='interactive', show=True, bgcolor='white')
    view = canvas.central_widget.add_view()
    view.camera = TurntableCamera(fov=45, distance=8, elevation=30, azimuth=30, up='+z')

    collection = CappedCylinderCollection(parent=view.scene)
    # Axis-aligned red cylinder
    collection.add_cylinder(position=[-2, 0, 0], radius=0.3, height=1.5,
                            color=[1, 0, 0, 1], orientation=None)
    # 90° around X (green)
    Rx = Rotation.from_euler('x', 90, degrees=True).as_matrix().astype(np.float32)
    collection.add_cylinder(position=[0, 0, 0], radius=0.2, height=2.0,
                            color=[0, 1, 0, 1], orientation=Rx)
    # 90° around Y (blue)
    Ry = Rotation.from_euler('y', 90, degrees=True).as_matrix().astype(np.float32)
    collection.add_cylinder(position=[2, 0, 0], radius=0.4, height=1.0,
                            color=[0, 0, 1, 1], orientation=Ry)
    
    collection.refresh()

    app.run()
