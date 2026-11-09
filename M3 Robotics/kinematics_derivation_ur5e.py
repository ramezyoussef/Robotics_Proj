#!/usr/bin/env python3
"""
MCTR911 - Milestone 3: Velocity & Acceleration Kinematics for UR5e
Technique:
  - DH Forward Kinematics (cumulative transformations ^0T_i)
  - Geometric Jacobian via J_w = ^0Z_(i-1), J_v = ^0Z_(i-1) x (^0O_N - ^0O_(i-1))
  - Singularity-Guarded Inverse Velocity Kinematics (Damped Pseudo-inverse fallback)
  - Central-difference Jacobian acceleration kinematics:
      * X_ddot = J_dot*q_dot + J*q_ddot
      * q_ddot = J^(-1) * (X_ddot - J_dot*q_dot)
"""

import numpy as np

# UR5e Standard DH Parameters: [d, a, alpha]
DH_PARAMS = [
    [0.1625,  0.0,      np.pi / 2],   # Joint 1: Base
    [0.0,    -0.4250,   0.0],         # Joint 2: Shoulder
    [0.0,    -0.3922,   0.0],         # Joint 3: Elbow
    [0.1333,  0.0,      np.pi / 2],   # Joint 4: Wrist 1
    [0.0997,  0.0,     -np.pi / 2],   # Joint 5: Wrist 2
    [0.0996,  0.0,      0.0]          # Joint 6: Wrist 3 / Tool Flange
]

def dh_matrix(theta, d, a, alpha):
    """Calculates single DH transformation matrix ^(i-1)T_i."""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0.0,     sa,       ca,      d],
        [0.0,    0.0,      0.0,    1.0]
    ])

def forward_kinematics(q):
    """Computes all cumulative transformation matrices ^0T_i for i = 0..6."""
    T = np.eye(4)
    T_list = [T.copy()]
    for i in range(6):
        d, a, alpha = DH_PARAMS[i]
        T = T @ dh_matrix(q[i], d, a, alpha)
        T_list.append(T.copy())
    return T_list

def get_end_effector_position(q):
    """Returns Cartesian (x, y, z) coordinates of the end-effector tool flange."""
    T_list = forward_kinematics(q)
    return T_list[6][:3, 3]

def compute_jacobian(q):
    """
    Computes full 6x6 Geometric Jacobian following the lecture technique:
      J_vi = ^0Z_(i-1) x (^0O_N - ^0O_(i-1))
      J_wi = ^0Z_(i-1)
    """
    T_list = forward_kinematics(q)
    O_n = T_list[6][:3, 3]
    
    J = np.zeros((6, 6))
    for i in range(6):
        Z_prev = T_list[i][:3, 2]
        O_prev = T_list[i][:3, 3]
        
        J_v = np.cross(Z_prev, (O_n - O_prev))
        J_w = Z_prev
        
        J[0:3, i] = J_v
        J[3:6, i] = J_w
    return J

def forward_velocity_kinematics(q, q_dot):
    """X_dot = J * q_dot (6x1: [vx, vy, vz, wx, wy, wz]^T)"""
    J = compute_jacobian(q)
    return J @ q_dot

def inverse_velocity_kinematics(q, X_dot, position_only=False, damping=1e-3):
    """
    Computes joint velocities q_dot from Cartesian velocities X_dot.
    Equipped with singularity-damping guard.
    """
    J = compute_jacobian(q)
    if position_only:
        J_pos = J[:3, :]
        # Over-actuated pseudo-inverse with damping
        J_pinv = J_pos.T @ np.linalg.inv(J_pos @ J_pos.T + (damping**2) * np.eye(3))
        return J_pinv @ X_dot
    else:
        # Full 6-DoF inverse with Singularity Guard (Damped Least Squares fallback)
        det_J = np.linalg.det(J)
        if abs(det_J) < 1e-4:
            inv_J = J.T @ np.linalg.inv(J @ J.T + (damping**2) * np.eye(6))
            return inv_J @ X_dot
        return np.linalg.inv(J) @ X_dot

def compute_jacobian_derivative(q, q_dot, dt=1e-4):
    """
    Computes J_dot using 2nd-order Central Difference:
    J_dot = (J(q + q_dot*dt) - J(q - q_dot*dt)) / (2*dt)
    """
    J_plus = compute_jacobian(q + q_dot * dt)
    J_minus = compute_jacobian(q - q_dot * dt)
    return (J_plus - J_minus) / (2.0 * dt)

def forward_acceleration_kinematics(q, q_dot, q_ddot):
    """X_ddot = J_dot * q_dot + J * q_ddot"""
    J = compute_jacobian(q)
    J_dot = compute_jacobian_derivative(q, q_dot)
    return J_dot @ q_dot + J @ q_ddot

def inverse_acceleration_kinematics(q, q_dot, X_ddot, position_only=False, damping=1e-3):
    """
    q_ddot = J^(-1) * (X_ddot - J_dot * q_dot)
    """
    J = compute_jacobian(q)
    J_dot = compute_jacobian_derivative(q, q_dot)
    C_term = J_dot @ q_dot

    if position_only:
        J_pos = J[:3, :]
        C_pos = C_term[:3]
        J_pinv = J_pos.T @ np.linalg.inv(J_pos @ J_pos.T + (damping**2) * np.eye(3))
        return J_pinv @ (X_ddot - C_pos)
    else:
        det_J = np.linalg.det(J)
        if abs(det_J) < 1e-4:
            inv_J = J.T @ np.linalg.inv(J @ J.T + (damping**2) * np.eye(6))
        else:
            inv_J = np.linalg.inv(J)
        return inv_J @ (X_ddot - C_term)

def inverse_kinematics_position(target_pos, q_seed, max_steps=40, tol=1e-3, damping=0.01):
    """Inverse Kinematics position solver using Damped Least Squares."""
    q = np.array(q_seed, dtype=float)
    for _ in range(max_steps):
        curr_pos = get_end_effector_position(q)
        err = target_pos - curr_pos
        if np.linalg.norm(err) < tol:
            break
        
        J_pos = compute_jacobian(q)[:3, :]
        inv_J = J_pos.T @ np.linalg.inv(J_pos @ J_pos.T + (damping ** 2) * np.eye(3))
        dq = inv_J @ err
        
        max_dq = 0.2
        if np.linalg.norm(dq) > max_dq:
            dq = dq * (max_dq / np.linalg.norm(dq))
            
        q += dq
        q = (q + np.pi) % (2 * np.pi) - np.pi
    return q
