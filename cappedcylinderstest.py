import numpy as np
from vispy import scene, app, use
from vispy.scene.visuals import InstancedMesh
from vispy.scene.cameras import TurntableCamera

# ─── Enable GL+ (instancing) ───────────────────────────────────────────────
use(gl='gl+')

# ─── Build unit-cylinder side mesh ─────────────────────────────────────────
segments = 32
θ = np.linspace(0, 2*np.pi, segments, endpoint=False, dtype=np.float32)
ring = np.stack([np.cos(θ), np.sin(θ)], axis=1)   # (segments,2)

# bottom circle at z=-0.5, top at z=+0.5
bottom = np.column_stack([ring, np.full(segments, -0.5, dtype=np.float32)])
top    = np.column_stack([ring, np.full(segments, +0.5, dtype=np.float32)])
cyl_verts = np.vstack([bottom, top])               # (2*segments,3)

# two triangles per quad around the side
cyl_faces = []
for i in range(segments):
    ni = (i + 1) % segments
    cyl_faces.append([i,       i+segments, ni])
    cyl_faces.append([ni,      i+segments, ni+segments])
cyl_faces = np.array(cyl_faces, dtype=np.uint32)

# ─── Build unit-disk mesh for caps ─────────────────────────────────────────
N = 32
phi = np.linspace(0, 2*np.pi, N, endpoint=False, dtype=np.float32)
circle2d = np.column_stack([np.cos(phi), np.sin(phi)])
disk_verts2d = np.vstack([[0,0], circle2d])       # (N+1,2)
disk_verts = np.zeros((len(disk_verts2d), 3), np.float32)
disk_verts[:, :2] = disk_verts2d

disk_faces = np.vstack(
    [[0, i, i+1] for i in range(1, N)] +
    [[0, N, 1]]
).astype(np.uint32)

# ─── Instance data ─────────────────────────────────────────────────────────
M = 100
positions = np.random.uniform(-8, 8, (M, 3)).astype(np.float32)
positions[:, 2] = 0

radii   = np.random.uniform(0.3, 1.0, M).astype(np.float32)
heights = np.random.uniform(1.0, 3.0, M).astype(np.float32)
colors  = np.random.uniform(0.3, 1.0, (M, 4)).astype(np.float32)

# 3×3 transforms for cylinders: scale X/Y by radius, Z by height
cyl_transforms = np.repeat(np.eye(3, dtype=np.float32)[None], M, axis=0)
cyl_transforms[:, 0, 0] = radii
cyl_transforms[:, 1, 1] = radii
cyl_transforms[:, 2, 2] = heights

# Prepare caps: bottom & top per cylinder
cap_positions  = np.zeros((2*M, 3), dtype=np.float32)
cap_transforms = np.repeat(np.eye(3, dtype=np.float32)[None], 2*M, axis=0)
cap_colors     = np.zeros((2*M, 4), dtype=np.float32)

for i, (pos, r, h, col) in enumerate(zip(positions, radii, heights, colors)):
    bi, ti = 2*i, 2*i+1
    cap_positions[bi] = pos + (0, 0, -0.5*h)
    cap_positions[ti] = pos + (0, 0, +0.5*h)
    cap_transforms[bi] = np.diag([r, r, 1.0])
    cap_transforms[ti] = np.diag([r, r, 1.0])
    cap_colors[bi] = col
    cap_colors[ti] = col

# ─── Set up canvas & TurntableCamera ───────────────────────────────────────
canvas = scene.SceneCanvas(keys='interactive', show=True, bgcolor='black')
view   = canvas.central_widget.add_view()
view.camera = TurntableCamera(fov=45, distance=20, elevation=30, azimuth=30, up='+z')

# ─── Draw with TWO InstancedMesh calls ─────────────────────────────────────
# 1) sides
cyl_mesh = InstancedMesh(
    cyl_verts, cyl_faces,
    instance_positions=positions,
    instance_transforms=cyl_transforms,
    instance_colors=colors,
    parent=view.scene
)
# 2) caps
cap_mesh = InstancedMesh(
    disk_verts, disk_faces,
    instance_positions=cap_positions,
    instance_transforms=cap_transforms,
    instance_colors=cap_colors,
    parent=view.scene
)

if __name__ == '__main__':
    app.run()
