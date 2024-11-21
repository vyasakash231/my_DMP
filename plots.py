import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec


class RobotPlotter:
    def __init__(self, simulation, plot_limits=None):
        self.sim = simulation
        self.fig = None
        self.ax = None
        self.axs = None
        
        # Copy necessary attributes from simulation
        self.x_plot = np.array(simulation.X_plot, dtype=np.float64)
        self.y_plot = np.array(simulation.Y_plot, dtype=np.float64)
        self.z_plot = np.array(simulation.Z_plot, dtype=np.float64)
        self.n = simulation.n
        self.end_time = simulation.T
        self.time_frame = simulation.time_frame
        self.obstacles = simulation.obstacles
       
        # generate obstacle shape
        if self.obstacles is not None:
            self._generate_superquadric()
        
        # These might be None for free_fall simulation
        self.Xd_plot = getattr(simulation, 'Xd_plot', None)
        self.e_task = getattr(simulation, 'position_error_plot', None)

        if self.Xd_plot is not None:
            self.Xd_plot = np.array(self.Xd_plot)
        if self.e_task is not None:
            self.e_task = np.array(self.e_task)

    def setup_plot_1(self):
        self.fig = plt.figure(figsize=(12, 10))
        gs = gridspec.GridSpec(1, 2)
        
        self.axs = []
        self.axs.append(self.fig.add_subplot(gs[0,0], projection='3d'))
        self.axs.append(self.fig.add_subplot(gs[0,1]))
        return self.create_animation(self.update_1)
    
    def setup_plot_2(self):
        self.fig = plt.figure(figsize=(12, 10))
        gs = gridspec.GridSpec(1, 3)
        
        self.axs = []
        self.axs.append(self.fig.add_subplot(gs[0,0], projection='3d'))
        self.axs.append(self.fig.add_subplot(gs[0,1], projection='3d'))
        self.axs.append(self.fig.add_subplot(gs[0,2], projection='3d'))
        return self.create_animation(self.update_2)
    
    def create_animation(self, update_func):
        anim = FuncAnimation(self.fig, update_func, frames=self.x_plot.shape[0], interval=30, blit=False, repeat=False)
        plt.subplots_adjust(wspace=0.3, hspace=0.3)
        return anim
    
    def update_1(self, frame):
        k = frame
        
        # Robot visualization
        ax = self.axs[0]
        ax.clear()
        self._plot_robot_state(ax, k)
        self._set_3d_plot_properties(ax, 19, -162)  # set graph properties 

        self._plot_position_errors(k)  # Error plot
        return self.axs
    
    def update_2(self, frame):
        k = frame
        
        # Robot visualization
        ax0  = self.axs[0]
        ax0.clear()
        self._plot_robot_state(ax0, k)
        self._set_3d_plot_properties(ax0, 34, -11)   # set graph properties 

        ax1  = self.axs[1]
        ax1.clear()
        self._plot_robot_state(ax1, k)
        self._set_3d_plot_properties(ax1, 19, -162)   # set graph properties 

        ax2 = self.axs[2]
        ax2.clear()
        self._plot_robot_state(ax2, k)
        self._set_3d_plot_properties(ax2, 90, -90)   # set graph properties 
        return self.axs
    
    """%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"""
    
    def _plot_robot_state(self, ax, k):
        try:
            ax.plot(self.Xd_plot[:,0], self.Xd_plot[:,1], self.Xd_plot[:,2], '-r', linewidth=2)
        except:
            pass  

        if self.obstacles is not None:
            self._plot_multiple_obstacles(ax)  

        for j in range(self.x_plot.shape[1]):
            ax.plot(self.x_plot[k,j:j+2], self.y_plot[k,j:j+2], self.z_plot[k,j:j+2], '-', linewidth=10-j)
            ax.plot(self.x_plot[k,j], self.y_plot[k,j], self.z_plot[k,j], 'ko', linewidth=10)
        
        # trajectory of EE
        ax.plot(self.x_plot[:k+1, -1], self.y_plot[:k+1, -1], self.z_plot[:k+1, -1], linewidth=1.5, color='b')
    
    def _set_3d_plot_properties(self, ax, elev, azim):
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlim(-0.1, 0.3)
        ax.set_ylim(-0.1, 0.3)
        ax.set_zlim(0, 0.4)
        ax.set_box_aspect((1, 1, 1))
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')

    def _plot_position_errors(self, k):
        ax = self.axs[1]
        ax.clear()
        ax.set_xlim(0, self.end_time)
        ax.set_ylim(np.min(self.e_task) - 0.01, np.max(self.e_task) + 0.01)
        
        for i in range(3):
            ax.plot(self.time_frame[:k+1], self.e_task[:k+1, i], label=f'e_{i+1}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Position Error (m)')

    def _generate_superquadric(self, num_points=100):
        self.X_obs = []
        self.Y_obs = []
        self.Z_obs = []

        for _, obstacle in enumerate(self.obstacles):
            xc, yc, zc = obstacle['center']
            r1, r2, r3 = np.array(obstacle['radius'])
            m1, m2, m3 = 2*obstacle['order']

            # Create parametric angles
            eta = np.linspace(-np.pi/2, np.pi/2, num_points)
            omega = np.linspace(-np.pi, np.pi, num_points)
            ETA, OMEGA = np.meshgrid(eta, omega)
            
            # Superquadric equations in parametric form
            cos_eta = np.abs(np.cos(ETA))**(2/m1)
            sin_eta = np.abs(np.sin(ETA))**(2/m3)
            cos_omega = np.abs(np.cos(OMEGA))**(2/m1)
            sin_omega = np.abs(np.sin(OMEGA))**(2/m2)
            
            # Sign corrections
            cos_eta = cos_eta * np.sign(np.cos(ETA))
            sin_eta = sin_eta * np.sign(np.sin(ETA))
            cos_omega = cos_omega * np.sign(np.cos(OMEGA))
            sin_omega = sin_omega * np.sign(np.sin(OMEGA))
            
            # Generate the surface points
            X = r1 * cos_eta * cos_omega + xc
            Y = r2 * cos_eta * sin_omega + yc
            Z = r3 * sin_eta + zc

            self.X_obs.append(X)
            self.Y_obs.append(Y)
            self.Z_obs.append(Z)

    def _plot_multiple_obstacles(self, ax):
        for idx, obstacle in enumerate(self.obstacles):
            ax.plot_surface(self.X_obs[idx], self.Y_obs[idx], self.Z_obs[idx], color=obstacle['color'], alpha=0.8)

    # def transform(self, angle):
    #     if angle == 0:
    #         return np.eye(3)
    #     return np.array([[np.cos(angle), -np.sin(angle), 0],
    #                      [np.sin(angle),  np.cos(angle), 0],
    #                      [      0      ,       0       , 1]])

    def show(self):
        plt.show()