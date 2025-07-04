import numpy as np
from vispy import app, scene
from vispy.scene import visuals, transforms

# 1) Set up canvas + 3D view
canvas = scene.SceneCanvas(keys='interactive', show=True, bgcolor='white')
view = canvas.central_widget.add_view()
view.camera = scene.TurntableCamera(fov=45, distance=6)

# 2) Create some blue spheres
radius = 0.5
centers = [(-1.5, 0, 0), (0, 0, 0), (1.5, 0, 0)]
spheres = []
for c in centers:
    sph = visuals.Sphere(radius=radius, method='latitude',
                         parent=view.scene, color=(0.5, 0.5, 1, 1))
    sph.transform = transforms.STTransform(translate=c)
    spheres.append({'visual': sph, 'center': np.array(c)})

# 3) Create one red “highlight” sphere (initially hidden)
highlight = visuals.Sphere(radius=radius * 1.1, method='latitude',
                           parent=view.scene, color=(1, 0, 0, 1))
highlight.transform = transforms.STTransform(translate=(0, 0, 0))
highlight.visible = False

@canvas.events.mouse_press.connect
def on_mouse_press(event):
    # Unproject click to a ray in world space
    tform = view.scene.transform
    x, y = event.pos
    p0 = tform.imap([x, y, 0, 1])  # near plane
    p1 = tform.imap([x, y, 1, 1])  # far plane
    p0 /= p0[3];  p1 /= p1[3]
    origin = p0[:3]
    direction = p1[:3] - origin
    direction /= np.linalg.norm(direction)

    # Ray–sphere tests
    hit_index = None
    min_t = np.inf
    for i, obj in enumerate(spheres):
        C = obj['center']
        L = origin - C
        b = np.dot(direction, L)
        c = np.dot(L, L) - radius**2
        disc = b * b - c
        if disc < 0:
            continue
        t_hit = -b - np.sqrt(disc)
        if 0 < t_hit < min_t:
            min_t = t_hit
            hit_index = i

    # Show or hide the highlight sphere
    if hit_index is not None:
        print(f"Hit sphere {hit_index} at distance {min_t:.3f}")
        # Move and show the red “highlight”
        highlight.transform.translate = spheres[hit_index]['center']
        highlight.visible = True
    else:
        print("No hit")
        highlight.visible = False

    canvas.update()

if __name__ == '__main__':
    canvas.show()
    app.run()
