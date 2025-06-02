from felipe_utils.research_utils.io_ops import load_json
from felipe_utils.scan_data_utils import *
from felipe_utils import tof_utils_felipe
import os
from scipy.ndimage import gaussian_filter
from utils.coding_schemes_utils import ImagingSystemParams, init_coding_list
from felipe_utils.research_utils.np_utils import calc_error_metrics, print_error_metrics
from spad_toflib.coding_schemes import IdentityCoding
import matplotlib.pyplot as plt
from plot_figures.plot_utils import get_scheme_color
from scipy.ndimage import median_filter

import cv2
import open3d as o3d



global_shift = 0
downsamp_factor = 1  # Spatial downsample factor
hist_tbin_factor = 1.0  # increase tbin size to make histogramming faster

scan_data_params = load_json('scan_params.json')
io_dirpaths = load_json('io_dirpaths.json')
hist_img_base_dirpath = io_dirpaths["preprocessed_hist_data_base_dirpath"]

## Load processed scene:
# scene_id = '20190209_deer_high_mu/free'
scene_id = '20190207_face_scanning_low_mu/free'
# scene_id = '20190207_face_scanning_low_mu/ground_truth'

# scene_id = '20181105_face/opt_flux'

# scene_id = '20190207_face_scanning_low_mu/ground_truth'

assert (scene_id in scan_data_params['scene_ids']), "{} not in scene_ids".format(scene_id)
hist_dirpath = os.path.join(hist_img_base_dirpath, scene_id)

## Histogram image params
n_rows_fullres = scan_data_params['scene_params'][scene_id]['n_rows_fullres']
n_cols_fullres = scan_data_params['scene_params'][scene_id]['n_cols_fullres']
(nr, nc) = (n_rows_fullres // downsamp_factor, n_cols_fullres // downsamp_factor)  # dims for face_scanning scene
min_tbin_size = scan_data_params['min_tbin_size']  # Bin size in ps
hist_tbin_size = min_tbin_size * hist_tbin_factor  # increase size of time bin to make histogramming faster
hist_img_tau = scan_data_params['hist_preprocessing_params']['hist_end_time'] - \
               scan_data_params['hist_preprocessing_params']['hist_start_time']
nt = get_nt(hist_img_tau, hist_tbin_size)

## Load histogram image
hist_img_fname = get_hist_img_fname(nr, nc, hist_tbin_size, hist_img_tau, is_unimodal=False)
hist_img_fpath = os.path.join(hist_dirpath, hist_img_fname)
hist_img = np.load(hist_img_fpath)

## Shift histogram image if needed
hist_img = np.roll(hist_img, global_shift, axis=-1)

denoised_hist_img = gaussian_filter(hist_img, sigma=0.75, mode='wrap', truncate=1)
(tbins, tbin_edges) = get_hist_bins(hist_img_tau, hist_tbin_size)

irf_tres = scan_data_params['min_tbin_size']  # in picosecs
irf = get_scene_irf(scene_id, nt, tlen=hist_img_tau, is_unimodal=False)


# FOV along the major axis (in degrees)
fov_major_axis_deg = 40
fov_major_axis_rad = np.radians(fov_major_axis_deg)  # Convert to radians

# Calculate focal length
fx = nc / (2 * np.tan(fov_major_axis_rad / 2))
fy = fx  # Assume square pixels if no other info is provided

# Principal point (image center)
cx, cy = nr / 2, nc / 2

print(f"Estimated Intrinsics:\nfx = {fx:.2f}, fy = {fy:.2f}, cx = {cx:.2f}, cy = {cy:.2f}")

params = {}
params['n_tbins'] = tbins.shape[-1]
# params['dMax'] = 5
# params['rep_freq'] = direct_tof_utils.depth2freq(params['dMax'])
params['rep_freq'] = scan_data_params['laser_rep_freq']
params['rep_tau'] = 1. / params['rep_freq']
params['dMax'] = tof_utils_felipe.freq2depth(params['rep_freq'])
params['T'] = 0.1  # intergration time [used for binomial]
params['depth_res'] = 1000  ##Conver to MM

params['imaging_schemes'] = [
    #ImagingSystemParams('TruncatedFourier', 'Gaussian', 'ifft', n_codes=8, pulse_width=1,  account_irf=True,
    #                   h_irf=irf),
    # ImagingSystemParams('LearnedImpulse', 'Learned', 'zncc',
    #                     model=os.path.join('bandlimited_models', 'version_1'),
    #                     account_irf=True, h_irf=irf),
    ImagingSystemParams('Greys', 'Gaussian', 'ncc', n_bits=8, pulse_width=1, account_irf=True, h_irf=irf),
    # ImagingSystemParams('LearnedImpulse', 'Learned', 'zncc',
    #                     model=os.path.join('bandlimited_models', 'n2188_k8_spaddata'),
    #                     account_irf=True, h_irf=irf),
    # ImagingSystemParams('LearnedImpulse', 'Learned', 'zncc',
    #                    model=os.path.join('bandlimited_models', 'n2188_k8_spaddata_v2'),
    #                    account_irf=True, h_irf=irf),
    # ImagingSystemParams('LearnedImpulse', 'Learned', 'zncc',
    #                     model=os.path.join('bandlimited_models', 'version_4_v2'),
    #                     account_irf=True, h_irf=irf)
    #ImagingSystemParams('Identity', 'Gaussian', 'matchfilt', pulse_width=1, account_irf=True, h_irf=irf),

]

params['meanBeta'] = 1e-4
params['trials'] = 1
params['freq_idx'] = [1]

n_tbins = params['n_tbins']
mean_beta = params['meanBeta']
tau = params['rep_tau']
depth_res = params['depth_res']
t = params['T']
trials = params['trials']
(rep_tau, rep_freq, tbin_res, t_domain, dMax, tbin_depth_res) = \
    (tof_utils_felipe.calc_tof_domain_params(params['n_tbins'], rep_tau=params['rep_tau']))

print(f'Time bin depth resolution {tbin_depth_res * 1000:.3f} mm')
print()

init_coding_list(n_tbins, params, t_domain=t_domain)
imaging_schemes = params['imaging_schemes']

depth_images = np.zeros((nr, nc, len(params['imaging_schemes'])))
error_maps = np.zeros((nr, nc, len(params['imaging_schemes'])))
byte_sizes = np.zeros((len(params['imaging_schemes'])))
rmse = np.zeros((len(params['imaging_schemes'])))
mae = np.zeros((len(params['imaging_schemes'])))

for i in range(len(imaging_schemes)):
    imaging_scheme = imaging_schemes[i]
    coding_obj = imaging_scheme.coding_obj
    coding_scheme = imaging_scheme.coding_id
    rec_algo = imaging_scheme.rec_algo

    if coding_scheme == 'Identity':
        coded_vals = hist_img.reshape(nr * nc, params['n_tbins'])
    else:
        coded_vals = coding_obj.encode_no_noise(hist_img.reshape(nr * nc, params['n_tbins'])).squeeze()

    decoded_depths = coding_obj.max_peak_decoding(coded_vals, rec_algo_id=rec_algo) * time2depth(hist_tbin_size * 1e-12)

    if 'face_scanning' in scene_id:
        mask = plt.imread(io_dirpaths['hist_mask_path'])
        mask = np.logical_not(mask)
        decoded_depths = mask.flatten() * decoded_depths

    normalized_decoded_depths = np.copy(decoded_depths)
    normalized_decoded_depths[normalized_decoded_depths == 0] = np.nan
    #vmin = np.nanmean(normalized_decoded_depths) - 1
    #vmax = np.nanmean(normalized_decoded_depths) + 1
    vmin = 0.04
    vmax = 1.0
    #vim = 0.0
    normalized_decoded_depths[normalized_decoded_depths > vmax] = np.nan
    normalized_decoded_depths[normalized_decoded_depths < vmin] = np.nan
    # normalized_decoded_depths = (normalized_decoded_depths - np.nanmean(normalized_decoded_depths)) / np.nanstd(normalized_decoded_depths)
    depth_images[:, :, i] = np.reshape(normalized_decoded_depths, (nr, nc))

depth_image = median_filter(depth_images[..., 0], size=3)

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

#o3d.visualization.draw_geometries([pcd], window_name="Mesh ")

pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

pcd.orient_normals_consistent_tangent_plane(k=100)

pcd.normals = o3d.utility.Vector3dVector(-np.asarray(pcd.normals))

mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)

mesh = mesh.filter_smooth_laplacian(number_of_iterations=10, lambda_filter=0.5)

mesh.remove_non_manifold_edges()


mesh_subdivided = mesh.subdivide_midpoint(number_of_iterations=2)

vertices_to_remove = densities < np.quantile(densities, 0.05)
mesh.remove_vertices_by_mask(vertices_to_remove)



if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
    raise ValueError("Mesh has no vertices or faces!")


if not mesh.has_vertex_normals():
    mesh.compute_vertex_normals()

# Get the normals
normals = np.asarray(mesh.vertex_normals)

# Normalize the normals to [0, 1] range for color mapping
normals_normalized = (normals + 1) / 2  # This gives the range [0, 1]

# Calculate the intensity for the grey color (based on the average of the normals)
# We can use the Y component of the normal to simulate shading, but here we use the magnitude of the normals
intensity = np.linalg.norm(normals_normalized, axis=1)  # This gives a scalar intensity for each vertex

# Ensure the intensity is within the [0, 1] range
intensity = np.clip(intensity * 0.8, 0, 1)

# Set grey shading: Apply intensity to all RGB channels for each vertex
grey_shading = np.stack([intensity, intensity, intensity], axis=1)  # Apply the same intensity to R, G, B

# Set the colors to the mesh vertices
mesh.vertex_colors = o3d.utility.Vector3dVector(grey_shading)

theta = np.radians(25)

R_y = np.array([[-np.cos(theta), 0, -np.sin(theta)],
                [0, 1, 0],
                [np.sin(theta), 0, -np.cos(theta)]])
# Apply the rotation matrix to the mesh vertices
mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices) @ R_y.T)  # Apply rotation

R_z = np.array([[-1, 0, 0],
                [0, -1, 0],
                [0, 0, 1]])

# Apply the rotation matrix to the mesh vertices
mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices) @ R_z.T)

theta = np.deg2rad(15)  # Convert to radians

# Rotation matrix around X-axis
R_x = np.array([[1, 0, 0],
                [0, np.cos(theta), -np.sin(theta)],
                [0, np.sin(theta), np.cos(theta)]])

# Apply the rotation matrix to the mesh vertices
mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices) @ R_x.T)  # Apply rotation

#o3d.visualization.draw_geometries([pcd], window_name="Mesh ")
#o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True, window_name="Mesh ")
vis = o3d.visualization.Visualizer()

vis.create_window(visible=False)  # Set visible=False to avoid popping up a window
vis.add_geometry(mesh)

# Update and render the scene
vis.poll_events()
vis.update_renderer()

# Add the mesh to the scene
vis.add_geometry(mesh)

# Capture the screenshot and save it to a file
vis.capture_screen_image(f"{coding_scheme}_image.png")  # Save the image as PNG

# Close the visualizer
#vis.destroy_window()

print(f"Image saved as '{coding_scheme}_image.png'")