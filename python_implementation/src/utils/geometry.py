import numpy as np

def angle_between_vectors(u, v):
    u_norm = u / np.linalg.norm(u)
    v_norm = v / np.linalg.norm(v)
    cos_angle = np.real(np.dot(u_norm, v_norm))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)
    angle_deg = angle_rad * 180 / np.pi
    return angle_deg

def generate_random_positions(n_points, area_size, start_point, height=1.5):
    positions = np.random.rand(n_points, 2) * area_size + start_point
    positions_3d = np.column_stack([positions, np.full(n_points, height)])
    return positions_3d

def generate_grid_positions(area_size, spacing, start_point, height=1.5):
    x_points = np.arange(start_point[0], start_point[0] + area_size[0] + spacing, spacing)
    y_points = np.arange(start_point[1], start_point[1] + area_size[1] + spacing, spacing)
    xx, yy = np.meshgrid(x_points, y_points)
    positions_2d = np.column_stack([xx.flatten(), yy.flatten()])
    positions_3d = np.column_stack([positions_2d, np.full(len(positions_2d), height)])
    return positions_3d
