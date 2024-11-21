import numpy as np
import pathlib
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from skimage.measure import marching_cubes
np.set_printoptions(suppress=True)

from util import *
from dmp import DMP


"""Here we create the trajectory to learn"""

# EXAMPLE-1
# t_f = 1.0 * np.pi # final time
# t_steps = 10**2  # time steps
# t = np.linspace(0, t_f, t_steps)

# a_x = 1.0 / np.pi
# b_x = 1.0  
# a_y = 1.0 / np.pi
# b_y = 1.0

# x = a_x * t * np.cos(b_x*t)
# y = a_y * t * np.sin(b_y*t)

# x_des = np.ndarray([2, t_steps])
# x_des[0,:] = x
# x_des[1,:] = y
# x_des = x_des - x_des[:,[0]]

# EXAMPLE-2
# x_des = np.load("2.npz")["arr_0"].T  # (79,2).T = (2,79)
# x_des -= x_des[:,[0]]  # y_des[:,0] this form will reduce the dimension of the vector on 1, so we'll use y_des[:,[0]] 

# EXAMPLE-3
source_path = str(pathlib.Path(__file__).parent.absolute())  
data = np.load(source_path+'/'+str('example')+'.npz')
x_des = data['demo'].T  # (2,611)
# x_des = x_des[:,::2]  # (2,306)
x_des = x_des - x_des[:,[0]]  # this will make initial condition [0;0]
x_des /= 100 # convert data from mm to m

# add Z-axis to data
x_des = np.vstack((x_des, np.zeros((1,x_des.shape[1]))))

# obstacle dimension
x_c = 0.55
y_c = -0.15
z_c = 0
r1 = 0.1
r2 = 0.2
r3 = 0.1  
m = 2  # (for ellipse = 1, rectangle = 2)

# Setup DMP
dmp = DMP(no_of_DMPs=3, no_of_basis_func=40, K = 1000.0, dt = 0.01, alpha = 3.0, obstacle={"center":[x_c,y_c,z_c], "radius":[r1,r2,r3], "order":m})

"""learn Weights based on Demo"""
dmp.imitate_path(X_des = x_des)
    

"""Imitate Demo without Obstacle"""
X_track, _, _ = dmp.rollout()


"""Imitate Demo with Obstacle"""
X_classical = X_track
dmp.reset_state()
X_track = np.zeros((dmp.no_of_DMPs, 1))
dX_track = np.zeros((dmp.no_of_DMPs, 1))
ddX_track = np.zeros((dmp.no_of_DMPs, 1))

# X0 = np.array([[0.205],[-0.2005]])
# X = np.array([[0.205],[-0.2005]])
# Xg = np.array([[0.2],[-0.2]])
while True:
    # X_step, dX_step, ddX_step = dmp.step(X0, X, Xg, adapt=False)
    X_step, dX_step, ddX_step = dmp.step(adapt=False)
    X_track = np.append(X_track, X_step, axis=1)
    dX_track = np.append(dX_track, dX_step, axis=1)
    ddX_track = np.append(ddX_track, dX_step, axis=1)
    # X = X_step
    if np.linalg.norm(X_step - dmp.X_g) <= 1e-2:
        break


# fig, ax = plt.subplots()
# plt.plot(x_des[0,:], x_des[1,:], color='red', lw=5, label = 'demo')
# plt.plot(X_classical[0,:], X_classical[1,:], '-.', color="blue", lw=3.25, label="Imitation without obstacle")
# plt.plot(X_track[0,:], X_track[1,:], "g--", lw=3, label="Imitation with static obstacle") 

# # Create meshgrid
# x, y = np.meshgrid(np.linspace(-1.5, 1.5, 200), np.linspace(-1.5, 1.5, 200))

# # Define the obstacle function
# F = ((x - x_c) / r1)**(2*m) + ((y - y_c) / r2)**(2*m) - 1

# # Create the contour plot
# contour_level = 0  # The level at which you want to create the "isocurve"
# contour = ax.contour(x, y, F, levels=[contour_level])

# # Extract contour paths
# for path in contour.collections[0].get_paths():
#     x_contour, y_contour = path.vertices[:, 0], path.vertices[:, 1]
#     ax.fill(x_contour, y_contour, 'black', alpha=0.75)

# plt.legend()
# ax.set_xlim(-0.2, 0.8)
# ax.set_ylim(-0.6, 0.4)
# ax.set_aspect('equal')
# plt.show()



# fig, ax = plt.subplots()
# # Plot demonstration and classical imitation trajectories
# ax.plot(x_des[0,:], x_des[1,:], color='red', lw=4.5, label='Demo')
# ax.plot(X_classical[0,:], X_classical[1,:], '-.', color="blue", lw=3.25, label="Imitation without obstacle")

# # Create meshgrid
# x, y = np.meshgrid(np.linspace(-2, 2, 200), np.linspace(-2, 2, 200))

# # Define the obstacle function
# F = ((x - x_c) / r1)**(2*m) + ((y - y_c) / r2)**(2*m) - 1

# # Create the contour plot
# contour_level = 0  # The level at which you want to create the "isocurve"
# contour = ax.contour(x, y, F, levels=[contour_level])

# # Extract contour paths
# for path in contour.collections[0].get_paths():
#     x_contour, y_contour = path.vertices[:, 0], path.vertices[:, 1]
#     ax.fill(x_contour, y_contour, 'black', alpha=0.75)

# # Initialize the line for imitation with static obstacle
# track_line, = ax.plot([], [], "g--", lw=3.5, label="Imitation with static obstacle")

# # Set plot limits and properties
# ax.set_xlim(-0.5, 1.0)
# ax.set_ylim(-0.75, 0.5)
# ax.set_aspect("equal")  # Equal aspect ratio for 3D
# ax.legend()

# def init():
#     track_line.set_data([], [])
#     return track_line,

# def animate(i):
#     track_line.set_data(X_track[0, :i+1], X_track[1, :i+1])
#     return track_line,

# # Create animation
# anim = FuncAnimation(fig, animate, init_func=init, frames=X_track.shape[1], interval=50, blit=True)
# plt.tight_layout()
# plt.show()



fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
# Plot demonstration and classical imitation trajectories
ax.plot(x_des[0,:], x_des[1,:], color='red', lw=4.5, label='Demo')
ax.plot(X_classical[0,:], X_classical[1,:], '-.', color="blue", lw=3.25, label="Imitation without obstacle")

# Define the bounds of your space
x_min, x_max = -1.5, 1.5
y_min, y_max = -1.5, 1.5
z_min, z_max = -1.5, 1.5

# Create a finer grid for more accurate isosurface
n_points = 100
x, y, z = np.meshgrid(np.linspace(x_min, x_max, n_points),
                      np.linspace(y_min, y_max, n_points),
                      np.linspace(z_min, z_max, n_points))

# Define the 3D obstacle function
F = ((x - y_c) / r2)**(2*m) + ((y - x_c) / r1)**(2*m) + ((z - z_c) / r3)**(2*m) - 1

# Create the isosurface plot for 3D obstacle
verts, faces, _, _ = marching_cubes(F, 0)

# Scale and translate vertices to match your coordinate system
verts[:, 0] = verts[:, 0] * (x_max - x_min) / (n_points - 1) + x_min
verts[:, 1] = verts[:, 1] * (y_max - y_min) / (n_points - 1) + y_min
verts[:, 2] = verts[:, 2] * (z_max - z_min) / (n_points - 1) + z_min

# Plot the 3D obstacle
obstacle = ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2], triangles=faces, color='black', alpha=0.6, shade=True)

# Initialize the line for imitation with static obstacle
track_line, = ax.plot([], [], "g--", lw=3.5, label="Imitation with static obstacle")

# Set plot limits and properties
ax.set_xlim(-0.5, 1.0)
ax.set_ylim(-0.75, 0.5)
ax.set_zlim(-0.5, 0.5)
ax.set_box_aspect((1, 1, 1))  # Equal aspect ratio for 3D
ax.legend()

def init():
    track_line.set_data([], [])
    track_line.set_3d_properties([])
    return track_line,

def animate(i):
    track_line.set_data(X_track[0, :i+1], X_track[1, :i+1])
    track_line.set_3d_properties(X_track[2, :i+1])
    return track_line,

# Create animation
anim = FuncAnimation(fig, animate, init_func=init, frames=X_track.shape[1], interval=50, blit=True)

plt.tight_layout()
plt.show()