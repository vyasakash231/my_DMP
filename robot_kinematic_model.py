import numpy as np
from math import *

from plots import RobotPlotter

class Robot_KM:
    def __init__(self,n,alpha,a,d,le):
        self.n = n
        self.alpha = alpha
        self.a = a
        self.d = d
        self.le = le

    def _transformation_matrix(self,theta):
        I = np.eye(4)
        R = np.zeros((self.n,3,3))
        O = np.zeros((3,self.n))

        # Transformation Matrix
        for i in range(0,self.n):
            T = np.array([[         cos(theta[i])          ,          -sin(theta[i])         ,           0        ,          self.a[i]           ],
                          [sin(theta[i])*cos(self.alpha[i]), cos(theta[i])*cos(self.alpha[i]), -sin(self.alpha[i]), -self.d[i]*sin(self.alpha[i])],                                               
                          [sin(theta[i])*sin(self.alpha[i]), cos(theta[i])*sin(self.alpha[i]),  cos(self.alpha[i]),  self.d[i]*cos(self.alpha[i])],     
                          [               0                ,                 0               ,           0        ,               1              ]])

            T_new = np.dot(I,T)
            R[i,:,:] = T_new[0:3,0:3]
            O[0:3,i] = T_new[0:3,3]
            I = T_new
            i= i + 1

        T_final = I
        d_nn = np.array([[self.le],[0],[0],[1]])
        P_00_home = np.dot(T_final,d_nn)
        P_00 = P_00_home[0:3]
        return(R,O,P_00)

    def J(self, theta):
        if theta.ndim == 2:
            theta = theta.reshape(-1)

        R, O, _ = self._transformation_matrix(theta)

        R_n_0 = R[self.n-1,:,:]
        O_n_0 = np.transpose(np.array([O[:,self.n-1]]))
        O_E_n = np.array([[self.le],[0],[0]])
        O_E = O_n_0 + np.dot(R_n_0,O_E_n)

        Jz = np.zeros((3,self.n))
        Jw = np.zeros((3,self.n))

        for i in range(0,self.n):
            Z_i_0 = np.transpose(np.array([R[i,:,2]]))
            O_i_0 = np.transpose(np.array([O[:,i]]))
            O_E_i_0 = O_E - O_i_0

            cross_prod = np.cross(Z_i_0,O_E_i_0,axis=0)
            Jz[:,i] = np.reshape(cross_prod,(3,)) # conver 2D of shape (3,1) to 1D of shape (3,)
            Jw[:,i] = np.reshape(Z_i_0,(3,)) # conver 2D of shape (3,1) to 1D of shape (3,)

        J = np.concatenate((Jz,Jw),axis=0)
        return Jz

    def FK(self,theta):
        if theta.ndim == 2:
            theta = theta.reshape(-1)

        _, O, P_00 = self._transformation_matrix(theta)

        X_cord = np.concatenate(([0],O[0,:],P_00[[0],0]))
        Y_cord = np.concatenate(([0],O[1,:],P_00[[1],0]))
        Z_cord = np.concatenate(([0],O[2,:],P_00[[2],0]))
        
        self.EE = P_00.reshape(-1)
        return X_cord, Y_cord, Z_cord
    
    def memory(self, X, Y, Z, Ex=None):
        self.X_plot.append(X)
        self.Y_plot.append(Y)
        self.Z_plot.append(Z)
        if Ex is not None:
            self.position_error_plot.append(Ex)

    def damped_least_square_control(self, Je, Q, R, del_X):
        J_inv = np.linalg.inv(Je.T @ Q @ Je + R) @ (Je.T @ Q)
        d_theta = J_inv @ del_X
        return d_theta.T
    
    def plot_start(self, T, dt, X_des, obstacles):
        self.T = T
        self.time_frame = np.arange(0, T, dt)
        self.obstacles = obstacles

        # For plotting
        self.X_plot = []
        self.Y_plot = []
        self.Z_plot = []
        self.position_error_plot = []
        self.Xd_plot = X_des.T
    
    def show_plot(self, save_video=False):
        """Animation Setup"""
        plotter = RobotPlotter(self)  # by passing `self` we are passing all arguments of this class to `RobotPlotter` class
        anim = plotter.setup_plot_2()

        # Save the animation as an MP4 video
        if save_video is True:
            anim.save('animation_.mp4', writer='ffmpeg', fps=30)

        plotter.show()