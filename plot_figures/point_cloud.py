from felipe_utils.scan_data_utils import *
import open3d as o3d
from scipy.ndimage import gaussian_filter

# FOV along the major axis (in degrees)
fov_major_axis_deg = 100
fov_major_axis_rad = np.radians(fov_major_axis_deg)  # Convert to radians

mask = plt.imread(r'/Users/Patron/PycharmProjects/Indirect-Direct-ToF/data/cow.png')
mask = mask[..., -1]
depth_image = np.load('/Users/Patron/PycharmProjects/Indirect-Direct-ToF/data/cow_depth_map.npy') * mask
depth_image = gaussian_filter(depth_image, sigma=2)
(nr, nc) = depth_image.shape

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

X = np.linspace(-5, 5, nc)
Y = np.linspace(-5, 5, nr)
X, Y = np.meshgrid(X, Y)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, depth_image, cmap='viridis')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()

exit(0)

# Calculate focal length
fx = nc / (2 * np.tan(fov_major_axis_rad / 2))
fy = fx  # Assume square pixels if no other info is provided

# Principal point (image center)
cx, cy = nr / 2, nc / 2

height, width = depth_image.shape

# Generate mesh grid
x, y = np.meshgrid(np.arange(width), np.arange(height))

# Convert depth to meters if necessary (e.g., divide by 1000 if in mm)
z = depth_image.astype(np.float32) / 20.0

# Reproject to 3D using intrinsics
x3d = (x - cx) * z / fx
y3d = (y - cy) * z / fy
points = np.stack((x3d, y3d, z), axis=-1).reshape(-1, 3)

valid_points = ~np.isnan(points).any(axis=1) & (points[:, 2] > 0)
points = points[valid_points]

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

cl, ind = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=1.0)
pcd = pcd.select_by_index(ind)

o3d.visualization.draw_geometries([pcd], window_name="Mesh ")

pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

pcd.orient_normals_consistent_tangent_plane(k=100)

vis = o3d.visualization.Visualizer()

vis.create_window(visible=False)  # Set visible=False to avoid popping up a window
vis.add_geometry(pcd)

# Update and render the scene
vis.poll_events()
vis.update_renderer()

# Add the mesh to the scene
vis.add_geometry(pcd)
# Capture the screenshot and save it to a file
vis.capture_screen_image(f"image.png")  # Save the image as PNG

# Close the visualizer
#vis.destroy_window()

