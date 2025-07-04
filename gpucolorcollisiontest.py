import numpy as np
from vispy import app, scene, use
from vispy.geometry.generation import create_sphere
from vispy.scene.visuals import InstancedMesh
from copy import deepcopy

# 1) Setup VisPy + Qt6
use(app='pyqt6', gl='gl+')
canvas = scene.SceneCanvas(keys='interactive', show=True, bgcolor='white')
view = canvas.central_widget.add_view()
view.camera = scene.TurntableCamera(fov=45, distance=6)

# 2) Build unit sphere mesh
rows, cols, depth, radius = 20, 20, 10, 0.5
meshdata = create_sphere(rows, cols, depth=depth, radius=radius)
verts = meshdata.get_vertices().astype(np.float32)
faces = meshdata.get_faces().astype(np.uint32)

# 3) Three sphere centers
centers = np.array([
    [-1.5, 0.0, 0.0],
    [ 0.0, 0.0, 0.0],
    [ 1.5, 0.0, 0.0]
], dtype=np.float32)
N = len(centers)

# 4) Instance transforms & original colors (blue)
instance_transforms = np.repeat(np.eye(3, dtype=np.float32)[None, :, :], N, axis=0)
orig_colors = np.tile(np.array([0.5, 0.5, 1.0, 1.0], dtype=np.float32)[None, :], (N, 1))
highlight_colors = deepcopy(orig_colors)

# 5) Create the instanced blue spheres
spheres = InstancedMesh(
    vertices=verts,
    faces=faces,
    instance_positions=centers,
    instance_transforms=instance_transforms,
    instance_colors=orig_colors,
    parent=view.scene
)

selected = [False for _ in range (N)]

# 7) Generate exact ID-colors by encoding (i+1) into RGB bytes,
#    normalized to [0,1] so they fit in RGBA8:
id_colors = []
for i in range(N):
    idx = i + 1
    # encode low byte in red channel
    r = (idx & 0xFF) / 255.0
    # for completeness, support >255 ids by using green, blue channels
    g = ((idx >> 8) & 0xFF) / 255.0
    b = ((idx >> 16) & 0xFF) / 255.0
    id_colors.append([r, g, b, 1.0])
id_colors = np.array(id_colors, dtype=np.float32)

_click = None
DRAG_THRESHOLD = 5  # pixels squared

@canvas.events.mouse_press.connect
def on_mouse_press(event):
    global _click
    _click = event.pos

@canvas.events.mouse_release.connect
def on_mouse_release(event):
    global _click
    if _click is None:
        return
    dx = event.pos[0] - _click[0]
    dy = event.pos[1] - _click[1]
    _click = None
    if dx * dx + dy * dy > DRAG_THRESHOLD * DRAG_THRESHOLD:
        # It was a drag, not a click
        return

    # DEBUG STEP 1: logical mouse coords
    x_log, y_log = event.pos
    print(f"\n[DEBUG] Logical mouse pos: ({x_log:.2f}, {y_log:.2f})")

    # DEBUG STEP 2: canvas size & pixel_scale
    w_log, h_log = canvas.size
    ps = canvas.pixel_scale
    print(f"[DEBUG] Canvas size (logical): ({w_log}, {h_log}), pixel_scale: {ps:.2f}")

    # DEBUG STEP 3: framebuffer coords (flip Y)
    x_fb = int(x_log * ps)
    y_fb = int((h_log - y_log - 1) * ps)
    print(f"[DEBUG] Framebuffer coords: ({x_fb}, {y_fb})")

    # DEBUG STEP 4: expected byte IDs
    for idx, col in enumerate(id_colors):
        expected_byte = int(round(col[0] * 255))
        print(f"[DEBUG]   Sphere {idx} expected ID byte: {expected_byte}")

    # DEBUG STEP 5: render 1×1 pick pass
    old_cols = spheres.instance_colors.copy()
    spheres.instance_colors = id_colors
    img = canvas.render(region=(x_fb, y_fb, 1, 1), alpha=False, bgcolor=(0, 0, 0, 0))
    spheres.instance_colors = old_cols
    canvas.update()

    # DEBUG STEP 6: actual sampled bytes
    sampled = img[0, 0].astype(int)  # likely dtype uint8 already
    print(f"[DEBUG] Sampled bytes: {sampled}")

    # DEBUG STEP 7: decode pick index
    pick_id = (sampled[0] + (sampled[1] << 8) + (sampled[2] << 16)) - 1
    print(f"[DEBUG] Decoded pick index: {pick_id}")

    # DEBUG STEP 8: show/hide highlight
    if 0 <= pick_id < N:
        print(f"[DEBUG] Hit sphere {pick_id}")
        selected[pick_id] = not selected[pick_id]
        highlight_colors[pick_id, :] = np.array([1, 0, 0, 1]) if selected[pick_id] else np.array([0.5, 0.5, 1.0, 1.0])
        spheres.instance_colors = highlight_colors
    else:
        print("[DEBUG] No hit")

    canvas.update()

if __name__ == "__main__":
    canvas.show()
    app.run()
