import sys
import os
import numpy as np
import pathlib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from util import *
from robot_kinematic_model import Robot_KM
from my_DMP.dmp import DMP

# Manipulator parameters
n = 4  # No of Joint

# DH parameters
alpha = np.radians(np.array([0, 90, 0, 0])) 
a = np.array([0, 0, np.sqrt(0.128**2 + 0.024**2), 0.124])
d = np.array([0.077, 0, 0, 0])
theta = np.array([0.93028432, 1.78183731, -1.8493209, -0.78539816])  #np.radians(np.array([0, 90, -79.38, -45]))
Le = 0.126  # End-effector length

# weight matrix
R = np.diag([0.1, 0.1, 0.1, 0.1])
Q = np.diag([10, 10, 10])

# DMP trajectory
source_path = str(pathlib.Path(__file__).parent.absolute())  
data = np.load(source_path+'/'+str('example')+'.npz')
x_des = data['demo'].T  # (2,611)
x_des /= 350 # convert data from mm to m

# add Z-axis to data
x_des = np.vstack((x_des, np.linspace(0.1, 0.18, x_des.shape[1])))  # (3,611)
x_des[0,:] += 0.2
x_des[1,:] += 0.1

# obstacle dimension
obstacles = [{"center": np.array([0.23, 0.04, 0.125]), "radius": np.array([0.04, 0.02, 0.04]), "order": np.array([2, 2, 2]), "orientation": 0, "color": [0.2, 0.2, 0.2]},
                # {"center": np.array([0.26, 0.18, 0.125]), "radius": np.array([0.04, 0.02, 0.02]), "order": np.array([1, 1, 1]), "orientation": 0, "color": [0.2, 0.2, 0.2]}
            ]

T = 4
dt = 0.01

# Robot Kinematic Model
robot = Robot_KM(n,alpha,a,d,Le)

# call DMP
dmp = DMP(no_of_DMPs=3, no_of_basis_func=100, K=400.0, alpha=1, dt=dt, T=T, method="APF", obstacles=obstacles)

"""learn Weights based on Demo"""
dmp.imitate_path(X_des=x_des)
dmp.reset_state()

# Initial condition in Task Space
X_g = x_des[:,[-1]]
# X_g[-1,:] = 0.16
X_0 = x_des[:,[0]]
X_track = X_0

idx=0
robot.plot_start(T, dt, x_des, obstacles=obstacles)

# Simulation loop
while True:
    X_cord, Y_cord, Z_cord = robot.FK(theta)

    Xe, Ye, Ze = robot.EE   # zeta = [Xe, Ye, Ze]

    Ex = np.array([x_des[0,idx] - Xe, x_des[1,idx] - Ye, x_des[2,idx] - Ze])

    robot.memory(X_cord, Y_cord, Z_cord, Ex)
   
    X_step, dX_step = dmp.step_with_APF(X_g)
    X_track = np.append(X_track, X_step, axis=1)
    
    dX = np.array([[X_step[0,0] - Xe], [X_step[1,0] - Ye], [X_step[2,0] - Ze]])
    u = -2*(dX_step - dX/dt)  # control Input
    
    # Jacobian matrix
    Je = robot.J(theta)  # Jacobian matrix of Main task (3x4)
    
    # Damped-Least Square Method
    dtheta_dt = robot.damped_least_square_control(Je, Q, R, u)
    
    # Solving using Newton-Raphson Method
    theta = theta + dtheta_dt * dt  # In degree

    if np.linalg.norm(X_step - X_g) < 0.005:
        break

    idx += 1
    
robot.show_plot(save_video=False)
  