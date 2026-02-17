import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os

def plot_cube(ax, origin, size, alpha=0.1, color='r', edgecolor='r'):
    x, y, z = origin
    dx, dy, dz = size
    
    vertices = [
        [x, y, z], [x+dx, y, z], [x+dx, y+dy, z], [x, y+dy, z],
        [x, y, z+dz], [x+dx, y, z+dz], [x+dx, y+dy, z+dz], [x, y+dy, z+dz]
    ]
    
    faces = [
        [vertices[0], vertices[1], vertices[5], vertices[4]],
        [vertices[7], vertices[6], vertices[2], vertices[3]],
        [vertices[0], vertices[3], vertices[7], vertices[4]],
        [vertices[1], vertices[2], vertices[6], vertices[5]],
        [vertices[0], vertices[1], vertices[2], vertices[3]],
        [vertices[4], vertices[5], vertices[6], vertices[7]]
    ]
    
    cube = Poly3DCollection(faces, alpha=alpha, facecolor=color, edgecolor=edgecolor, linewidths=1)
    ax.add_collection3d(cube)

def plot_room_3d(train_pos, test_pos, interf_pos, mics_pos, room_dim, 
                 is_interference_active, output_dir, filename):
    blue_col = [0, 0.4470, 0.7410]
    orange_col = [0.9290, 0.6940, 0.1250]
    red_col = [0.8500, 0.3250, 0.0980]
    purple_col = [0.4940, 0.1840, 0.5560]
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(train_pos[:, 0], train_pos[:, 1], train_pos[:, 2], 
               c=[blue_col], marker='o', s=60, label='Adaptation', edgecolors='k', linewidths=0.5)
    
    ax.scatter(test_pos[:, 0], test_pos[:, 1], test_pos[:, 2],
               c=[orange_col], marker='p', s=80, label='Operational', edgecolors='k', linewidths=0.5)
    
    if is_interference_active and interf_pos is not None:
        ax.scatter(interf_pos[:, 0], interf_pos[:, 1], interf_pos[:, 2],
                   c=[red_col], marker='>', s=100, label='Interference', edgecolors='k', linewidths=0.5)
    
    ax.scatter(mics_pos[:, 0], mics_pos[:, 1], mics_pos[:, 2],
               c=[purple_col], marker='x', s=150, linewidths=3, label='Array')
    
    plot_cube(ax, [0, 0, 0], room_dim, alpha=0.01, color='r', edgecolor='r')
    
    ax.set_xlabel('x [m]', fontsize=12)
    ax.set_ylabel('y [m]', fontsize=12)
    ax.set_zlabel('z [m]', fontsize=12)
    ax.legend(loc='best', fontsize=12)
    
    ax.set_box_aspect([room_dim[0], room_dim[1], room_dim[2]])
    
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{filename}.jpg'), dpi=150)
    plt.savefig(os.path.join(output_dir, f'{filename}.png'), dpi=150)
    plt.close()
