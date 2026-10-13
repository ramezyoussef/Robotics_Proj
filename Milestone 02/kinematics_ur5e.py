#!/usr/bin/env python3
"""UR5e kinematics (standard DH): forward kinematics, numerical Jacobian,
and position-only inverse kinematics (Damped Least Squares)."""
import numpy as np

# Standard DH Parameters for Universal Robots UR5e
# [d_i (m), a_i (m), alpha_i (rad)]  (theta_i is the joint variable)
DH_PARAMS = [
    [0.1625,  0.0,      np.pi / 2],   # Joint 1: Base
    [0.0,    -0.4250,   0.0],         # Joint 2: Shoulder
    [0.0,    -0.3922,   0.0],         # Joint 3: Elbow
    [0.1333,  0.0,      np.pi / 2],   # Joint 4: Wrist 1
    [0.0997,  0.0,     -np.pi / 2],   # Joint 5: Wrist 2
    [0.0996,  0.0,      0.0]          # Joint 6: Wrist 3
]


def dh_transform(theta, d, a, alpha):
    """
    Computes the 4x4 homogeneous transformation matrix i-1_T_i
    from standard Denavit-Hartenberg parameters.
    """
    ct = np.cos(theta)
    st = np.sin(theta)
    ca = np.cos(alpha)
    sa = np.sin(alpha)

    T = np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0.0,     sa,       ca,      d],
        [0.0,    0.0,      0.0,    1.0]
    ])
    return T


def forward_kinematics(joint_angles):
    """
    Computes Forward Kinematics for UR5e.
    Input: list or array of 6 joint angles in radians.
    Output:
        T_06: 4x4 Homogeneous Transformation Matrix of the end-effector.
        positions: list of 3D origins [x, y, z] for each frame (for visualization).
    """
    T = np.eye(4)
    positions = [T[:3, 3].copy()]

    for i in range(6):
        theta = joint_angles[i]
        d, a, alpha = DH_PARAMS[i]
        T_i = dh_transform(theta, d, a, alpha)
        T = np.dot(T, T_i)
        positions.append(T[:3, 3].copy())

    return T, positions


def forward_kinematics_ur5e(joint_angles):
    """
    Returns (T_06, [T_01, T_02, ..., T_06]) - the final transform and the
    cumulative transform of every frame. Used by gui_test_report.py and the
    frame-plotting script.
    """
    T = np.eye(4)
    Ts = []
    for i in range(6):
        d, a, alpha = DH_PARAMS[i]
        T = T @ dh_transform(joint_angles[i], d, a, alpha)
        Ts.append(T.copy())
    return T, Ts


def get_end_effector_position(joint_angles):
    """Returns only the (x, y, z) cartesian coordinates."""
    T, _ = forward_kinematics(joint_angles)
    return T[:3, 3]


def compute_numerical_jacobian(joint_angles, eps=1e-6):
    """Computes the 3x6 position Jacobian matrix using central differences."""
    J = np.zeros((3, 6))
    for i in range(6):
        q_plus = np.array(joint_angles, dtype=float)
        q_minus = np.array(joint_angles, dtype=float)
        q_plus[i] += eps
        q_minus[i] -= eps

        pos_plus = get_end_effector_position(q_plus)
        pos_minus = get_end_effector_position(q_minus)

        J[:, i] = (pos_plus - pos_minus) / (2.0 * eps)
    return J


def inverse_kinematics_position(target_pos, q_init=None, max_iter=200, tol=1e-4, damping=0.01):
    """
    Solves for joint angles given a desired target position [x, y, z]
    using Damped Least Squares (Levenberg-Marquardt).
    Position only: orientation is not constrained, and theta6 stays at its
    initial value because it does not move the flange origin.
    """
    if q_init is None:
        q = np.array([0.0, -np.pi / 2, np.pi / 2, 0.0, 0.0, 0.0])
    else:
        q = np.array(q_init, dtype=float)

    target_pos = np.asarray(target_pos, dtype=float)

    for step in range(max_iter):
        current_pos = get_end_effector_position(q)
        error = target_pos - current_pos

        # Check convergence
        if np.linalg.norm(error) < tol:
            # Wrap angles to [-pi, pi]
            q = (q + np.pi) % (2 * np.pi) - np.pi
            return q, True

        J = compute_numerical_jacobian(q)
        # Damped Least Squares: dq = J.T * (J * J.T + lambda^2 * I)^(-1) * error
        J_damped = np.dot(J.T, np.linalg.inv(np.dot(J, J.T) + (damping ** 2) * np.eye(3)))
        dq = np.dot(J_damped, error)

        q += dq

    return q, False


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)

    print("==================================================")
    print("      UR5e KINEMATICS VERIFICATION (DH MODEL)     ")
    print("==================================================")

    # 1. Forward Kinematics at Zero Configuration
    zero_angles = np.zeros(6)
    T_zero, _ = forward_kinematics(zero_angles)
    print("\n1. Forward Kinematics at Zero Configuration (q = [0,0,0,0,0,0]):")
    print("Homogeneous Transformation Matrix (T_0_6):")
    print(T_zero)
    print(f"End-Effector Position: X={T_zero[0,3]:.4f}m, Y={T_zero[1,3]:.4f}m, Z={T_zero[2,3]:.4f}m")

    # 2. Forward Kinematics at an Arbitrary Test Configuration
    test_angles = np.array([0.0, -np.pi/4, np.pi/3, -np.pi/6, np.pi/2, 0.0])
    T_test, _ = forward_kinematics(test_angles)
    target_pos = T_test[:3, 3]
    print("\n2. Forward Kinematics at Test Configuration:")
    print(f"Joint Angles (rad): {test_angles}")
    print(f"Resulting Tool Position: X={target_pos[0]:.4f}m, Y={target_pos[1]:.4f}m, Z={target_pos[2]:.4f}m")

    # 3. Inverse Kinematics Position Solution
    print("\n3. Testing Inverse Kinematics Solver:")
    print(f"Goal Cartesian Target: {target_pos}")
    q_sol, success = inverse_kinematics_position(target_pos)

    if success:
        print("IK Solved Successfully!")
        print(f"Calculated Joint Angles: {q_sol}")
        reconstructed_pos = get_end_effector_position(q_sol)
        print(f"Verified FK from Solved IK: {reconstructed_pos}")
        print(f"Euclidean Error: {np.linalg.norm(target_pos - reconstructed_pos):.6e} m")
    else:
        print("IK Failed to converge within tolerance.")
