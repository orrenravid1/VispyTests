import numpy as np
from vispy import scene, app, use
from vispy.scene.visuals import InstancedMesh
from vispy.scene.cameras import TurntableCamera

# Enable the “gl+” backend for instancing support
use(gl='gl+')

# 1) Build a unit‐circle mesh in the XY plane
N = 64
theta = np.linspace(0, 2*np.pi, N, endpoint=False, dtype=np.float32)
circ = np.column_stack([np.cos(theta), np.sin(theta)])              # (N,2)
verts2d = np.vstack([[0.0, 0.0], circ])                             # (N+1,2)
verts = np.zeros((len(verts2d), 3), dtype=np.float32)              # expand to 3D (z=0)
verts[:, :2] = verts2d

# 2) Triangle‐fan faces
faces = np.vstack(
    [ [0, i, i+1] for i in range(1, N) ] +
    [ [0, N, 1] ]
).astype(np.uint32)

# 3) Per‐instance data
M = 200
# random XYZ positions (we keep z=0 so they lie in the XY plane)
instance_positions = np.random.uniform(-5, 5, size=(M, 3)).astype(np.float32)

# 3×3 transforms: we scale X and Y by a random radius
instance_transforms = np.repeat(np.eye(3, dtype=np.float32)[None, ...], M, axis=0)
radii = np.random.uniform(0.2, 1.0, size=(M,))
instance_transforms[:, 0, 0] = radii  # scale X
instance_transforms[:, 1, 1] = radii  # scale Y

# Optional per‐instance colors
instance_colors = np.random.uniform(0.3, 1.0, size=(M, 4)).astype(np.float32)

# 4) Set up canvas & 3D turntable view
canvas = scene.SceneCanvas(keys='interactive', show=True, bgcolor='black')
view = canvas.central_widget.add_view()
view.camera = TurntableCamera(
    fov=45,               # field of view
    distance=15,          # initial camera distance from the origin
    elevation=30,         # tilt up/down
    azimuth=30,           # rotate around Z axis
    up='+z'               # Z‐axis points up
)

# 5) Create and add the instanced mesh
mesh = InstancedMesh(
    verts, faces,
    instance_positions=instance_positions,
    instance_transforms=instance_transforms,
    instance_colors=instance_colors,
    parent=view.scene
)

if __name__ == '__main__':
    app.run()
