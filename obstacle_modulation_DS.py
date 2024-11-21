import numpy as np
from typing import List, Dict, Tuple, Union

class ObstacleModulation:
    def __init__(self):
        """Initialize the ObstacleModulation class."""
        pass
    
    def single_obstacle_modulation_matrix(self, zeta: np.ndarray,  obstacle: Dict) -> Tuple[np.ndarray, np.ndarray]:
        obstacle_center = obstacle['center']
        r1, r2, r3 = obstacle['radius']
        m1, m2, m3 = obstacle['order']
        eta = obstacle['eta']
        
        # Relative position
        x = (zeta[0, :] - obstacle_center[0]) / eta  # shape [1, 25x25]
        y = (zeta[1, :] - obstacle_center[1]) / eta  # shape [1, 25x25]
        z = (zeta[2, :] - obstacle_center[2]) / eta  # shape [1, 25x25]

        # Gradient wrt x,y,z
        gx = (m1 / (r1**m1)) * (x**(m1-1))  # shape [1, 25x25]
        gy = (m2 / (r2**m2)) * (y**(m2-1))  # shape [1, 25x25]
        gz = (m3 / (r3**m3)) * (z**(m3-1))  # shape [1, 25x25]
        grad = np.vstack([gx, gy, gz])  # shape [3, 25x25]
        
        # Normalize gradient to get normal vector
        n_vec = grad  # shape [3, 25x25]

        # first basis vector
        e1 = np.cross(n_vec.reshape(-1), np.array([0,0,1]))  # shape (3,)
        if np.linalg.norm(e1) < 1e-6:
            e1 = np.cross(n_vec.reshape(-1), np.array([0,1,0]))
            if np.linalg.norm(e1) < 1e-6:
                e1 = np.cross(n_vec.reshape(-1), np.array([1,0,0]))
        
        # second basis vector
        e2 = np.cross(n_vec.reshape(-1), e1)

        E = np.hstack((n_vec, e1.reshape(n_vec.shape), e2.reshape(n_vec.shape)))  # shape (3,3)
        
        # Compute distance to the surface
        d = (x/r1)**m1 + (y/r2)**m2 + (z/r3)**m3
        return E, d
    
    def compute_omega(self, d_k: Union[float, np.ndarray], d_i: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        # Check if inputs are non-scalar
        if isinstance(d_k, np.ndarray) or isinstance(d_i, np.ndarray):
            numerator = d_i - 1
            denominator = (d_i - 1) + (d_k - 1)
            
            # Handle division by zero and cases where d_k or d_i equals 1
            omega = np.zeros_like(d_k)
            valid_indices = (d_k != 1) & (d_i != 1)
            omega[valid_indices] = numerator[valid_indices] / denominator[valid_indices]
            return omega
        else:
            # Scalar operation
            if d_k != 1 and d_i != 1:
                return (d_i - 1) / ((d_i - 1) + (d_k - 1))
            return 0
    
    def compute_modulation_matrix(self, state: np.ndarray, obstacles: List[Dict]) -> np.ndarray:
        # Initialize combined modulation matrix as identity
        M_combined = np.eye(3)
        
        for k in range(len(obstacles)):
            obs_k = obstacles[k]
            # Compute basic parameters for current obstacle
            E_k, d_k = self.single_obstacle_modulation_matrix(state, obs_k)
            
            # Compute omega for current obstacle considering all other obstacles
            omega_k = np.ones_like(d_k)
            for i in range(len(obstacles)):
                if i != k:
                    obs_i = obstacles[i]
                    _, d_i = self.single_obstacle_modulation_matrix(state, obs_i)
                    omega_k = omega_k * self.compute_omega(d_k, d_i)
            
            # Compute eigenvalues
            rho = obstacles[k]['rho']
            lambda_1 = 1 - omega_k/(d_k**(1/rho))
            lambda_2 = 1 + omega_k/(d_k**(1/rho))
            lambda_3 = 1 + omega_k/(d_k**(1/rho))
            
            # Compute modulation matrix for current obstacle
            D_k = np.diag([lambda_1[0], lambda_2[0], lambda_3[0]])
            M_k = E_k @ D_k @ np.linalg.inv(E_k)  # shape [2,2]
            
            # Combine with overall modulation matrix
            M_combined = M_combined @ M_k
        
        return M_combined
