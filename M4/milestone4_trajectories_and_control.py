#!/usr/bin/env python3
"""
MCTR911 - Milestone 4: Trajectory Planning & Closed-Loop Motor Control for UR5e
Aligned with Tutorials 7 (trajectories), 8 (PID / Mamdani FLC) and 9 (Lagrangian dynamics, G(q)).

Key points
- Torch-tip kinematics (position + torch-axis orientation) at MuJoCo's 'torch_tip' site.
- Damped-least-squares IK on a 6x6 task [tip position ; torch axis] + null-space posture term.
- Quintic joint-space and task-space (line / circle) trajectories (Tut 7).
- SISO PID per joint + Lagrangian gravity compensation G(q) = dU/dq (Tut 8 + Tut 9).
- Mamdani FLC: 5 triangular MFs (LN, MN, S, MP, LP), 5x5 rule table on (error, change-of-error),
  min-AND, max-aggregation, centroid defuzzification (Tut 8). Output Cartesian velocity is mapped to
  joint rates with the Jacobian pseudo-inverse.
"""

import numpy as np
from kinematics_derivation_ur5e import forward_kinematics

# Canonical elbow-up welding ready pose (ur5e_mujoco_welding.py)
READY_Q = np.array([2.839, -1.467, 0.711, -0.815, -1.571, 0.023])

# torch_tip site in Standard DH frame 6: tool axis is +Z of T06 (pointing straight down at READY_Q)
T_FLANGE_TO_TIP = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.185],
    [0.0, 0.0, 0.0, 1.0]
])

# Approximate MuJoCo servo stiffness of ur5e.xml position actuators [Nm/rad].
SERVO_KP = np.array([2000.0, 2000.0, 2000.0, 500.0, 500.0, 500.0])

LAST_IK_REPORT = {"max_pos_err": 0.0, "max_tilt_deg": 0.0}


# ----------------------------------------------------------------------------------------
# Torch-tip kinematics
# ----------------------------------------------------------------------------------------
def _tip_and_axis(q):
    """Torch-tip world position and unit torch-axis (DH flange +Z) from ONE forward-kinematics call."""
    T06 = forward_kinematics(q)[6]
    p = (T06 @ T_FLANGE_TO_TIP)[:3, 3]
    a = T06[:3, 2]
    return p, a / np.linalg.norm(a)


def get_torch_tip_position(q):
    """Exact world (x, y, z) of MuJoCo's 'torch_tip' site."""
    return _tip_and_axis(q)[0]


def torch_axis(q):
    return _tip_and_axis(q)[1]


# Orientation we hold during welding = the torch axis of the canonical ready pose
TORCH_AXIS_REF = torch_axis(READY_Q)


def torch_tilt_deg(q, ref=None):
    """Angle [deg] between the current torch axis and the reference (ready-pose) torch axis."""
    ref = TORCH_AXIS_REF if ref is None else ref
    c = np.clip(np.dot(torch_axis(q), ref), -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def torch_tilt_from_vertical_deg(q):
    """Angle [deg] between the torch axis and straight down (0, 0, -1)."""
    c = np.clip(np.dot(torch_axis(q), np.array([0.0, 0.0, -1.0])), -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def compute_torch_tip_jacobian(q):
    """3x6 linear-velocity Jacobian of torch_tip in the world frame (numerical)."""
    eps = 1e-6
    q = np.asarray(q, dtype=float)
    J = np.zeros((3, 6))
    p0 = get_torch_tip_position(q)
    for j in range(6):
        qp = q.copy()
        qp[j] += eps
        J[:, j] = (get_torch_tip_position(qp) - p0) / eps
    return J


def _tip_axis_jacobians(q):
    """Returns tip position, torch axis, 3x6 position Jacobian, 3x6 axis Jacobian (d axis / d q)."""
    eps = 1e-6
    q = np.asarray(q, dtype=float)
    p0, a0 = _tip_and_axis(q)
    Jp = np.zeros((3, 6))
    Ja = np.zeros((3, 6))
    for j in range(6):
        qp = q.copy()
        qp[j] += eps
        p1, a1 = _tip_and_axis(qp)
        Jp[:, j] = (p1 - p0) / eps
        Ja[:, j] = (a1 - a0) / eps
    return p0, a0, Jp, Ja


def solve_torch_tip_ik(target_tip_xyz, q_seed=None, max_steps=150, tol=1e-4, damping=0.01,
                       w_orient=0.3, w_posture=0.05, return_info=False):
    """
    Damped-least-squares IK for the torch_tip position while keeping the torch axis at
    TORCH_AXIS_REF (ready-pose orientation, pointing down). Task = [dp ; w*(a_ref - a)] (6x6),
    null-space posture term pulls toward READY_Q unwrapped to the seed's 2*pi branch.
    """
    target_tip_xyz = np.asarray(target_tip_xyz, dtype=float)
    q = READY_Q.copy() if q_seed is None else np.array(q_seed, dtype=float).copy()
    q_ref = READY_Q + 2.0 * np.pi * np.round((q - READY_Q) / (2.0 * np.pi))

    pos_err = np.inf
    for it in range(max_steps):
        p, a, Jp, Ja = _tip_axis_jacobians(q)
        err_p = target_tip_xyz - p
        err_a = TORCH_AXIS_REF - a
        pos_err = np.linalg.norm(err_p)
        if pos_err < tol and np.linalg.norm(err_a) < 1e-3:
            break

        J = np.vstack([Jp, w_orient * Ja])
        e = np.concatenate([err_p, w_orient * err_a])
        J_pinv = J.T @ np.linalg.inv(J @ J.T + (damping ** 2) * np.eye(6))
        N_null = np.eye(6) - J_pinv @ J
        dq = J_pinv @ e + w_posture * (N_null @ (q_ref - q))

        n = np.linalg.norm(dq)
        if n > 0.15:
            dq *= 0.15 / n
        q += dq

    if return_info:
        info = {"pos_err": float(np.linalg.norm(target_tip_xyz - get_torch_tip_position(q))),
                "tilt_deg": torch_tilt_deg(q), "iters": it + 1}
        return q, info
    return q


# ----------------------------------------------------------------------------------------
# Trajectories (Tutorial 7)
# ----------------------------------------------------------------------------------------
def solve_quintic_coefficients(q0, qf, v0=0.0, vf=0.0, a0=0.0, af=0.0, tf=3.0):
    """5th-order polynomial coefficients [C0..C5]."""
    C0, C1, C2 = q0, v0, 0.5 * a0
    T = float(tf)
    A = np.array([
        [T**3,     T**4,      T**5],
        [3.0*T**2, 4.0*T**3,  5.0*T**4],
        [6.0*T,    12.0*T**2, 20.0*T**3]
    ], dtype=float)
    b = np.array([
        qf - (C0 + C1*T + C2*T**2),
        vf - (C1 + 2.0*C2*T),
        af - (2.0*C2)
    ], dtype=float)
    C3, C4, C5 = np.linalg.solve(A, b)
    return np.array([C0, C1, C2, C3, C4, C5])


def _quintic_eval(C, t):
    pos = C[0] + C[1]*t + C[2]*t**2 + C[3]*t**3 + C[4]*t**4 + C[5]*t**5
    vel = C[1] + 2*C[2]*t + 3*C[3]*t**2 + 4*C[4]*t**3 + 5*C[5]*t**4
    acc = 2*C[2] + 6*C[3]*t + 12*C[4]*t**2 + 20*C[5]*t**3
    return pos, vel, acc


def generate_joint_space_trajectory(q_start, q_end, tf=3.0, dt=0.03):
    """Rest-to-rest joint-space quintic taking the shortest angular path per joint."""
    q_start = np.asarray(q_start, dtype=float)
    q_end = np.asarray(q_end, dtype=float).copy()

    for j in range(6):
        diff = (q_end[j] - q_start[j] + np.pi) % (2.0 * np.pi) - np.pi
        candidate = q_start[j] + diff
        if abs(candidate) <= 6.2:
            q_end[j] = candidate

    t_arr = np.arange(0.0, tf + 1e-9, dt)
    N = len(t_arr)
    q_d, qdot_d, qddot_d = np.zeros((N, 6)), np.zeros((N, 6)), np.zeros((N, 6))
    for j in range(6):
        C = solve_quintic_coefficients(q_start[j], q_end[j], tf=tf)
        q_d[:, j], qdot_d[:, j], qddot_d[:, j] = _quintic_eval(C, t_arr)
    return t_arr, q_d, qdot_d, qddot_d


def _update_ik_report(q_d, X_d):
    errs = [np.linalg.norm(X_d[k] - get_torch_tip_position(q_d[k])) for k in range(len(q_d))]
    tilts = [torch_tilt_deg(q_d[k]) for k in range(len(q_d))]
    LAST_IK_REPORT["max_pos_err"] = float(np.max(errs))
    LAST_IK_REPORT["max_tilt_deg"] = float(np.max(tilts))
    if LAST_IK_REPORT["max_pos_err"] > 1e-3:
        print(f"[IK WARNING] max IK position error {LAST_IK_REPORT['max_pos_err']*1000:.2f} mm "
              f"(target may be near a limit / unreachable)")


def generate_task_space_linear(p_start, p_end, q_seed=None, tf=4.0, dt=0.03):
    """Straight line for torch_tip with quintic time scaling s(t)."""
    p_start, p_end = np.asarray(p_start, float), np.asarray(p_end, float)
    t_arr = np.arange(0.0, tf + 1e-9, dt)
    N = len(t_arr)
    X_d, Xdot_d, q_d = np.zeros((N, 3)), np.zeros((N, 3)), np.zeros((N, 6))

    C = solve_quintic_coefficients(0.0, 1.0, tf=tf)
    s, s_dot, _ = _quintic_eval(C, t_arr)
    q_curr = READY_Q.copy() if q_seed is None else np.asarray(q_seed, float).copy()

    for k in range(N):
        X_d[k] = p_start + s[k] * (p_end - p_start)
        Xdot_d[k] = s_dot[k] * (p_end - p_start)
        q_curr = solve_torch_tip_ik(X_d[k], q_curr, max_steps=100, tol=1e-4)
        q_d[k] = q_curr
    _update_ik_report(q_d, X_d)
    return t_arr, X_d, Xdot_d, q_d


def generate_task_space_circle(center, radius, q_seed=None, tf=5.0, dt=0.03, laps=1.0):
    """Circle (XY plane) for torch_tip with quintic angle scheduling theta(t)."""
    center = np.asarray(center, dtype=float)
    t_arr = np.arange(0.0, tf + 1e-9, dt)
    N = len(t_arr)
    X_d, Xdot_d, q_d = np.zeros((N, 3)), np.zeros((N, 3)), np.zeros((N, 6))

    C = solve_quintic_coefficients(0.0, 2.0 * np.pi * laps, tf=tf)
    theta, omega, _ = _quintic_eval(C, t_arr)
    q_curr = READY_Q.copy() if q_seed is None else np.asarray(q_seed, float).copy()

    for k in range(N):
        X_d[k] = [center[0] + radius*np.cos(theta[k]), center[1] + radius*np.sin(theta[k]), center[2]]
        Xdot_d[k] = [-radius*omega[k]*np.sin(theta[k]), radius*omega[k]*np.cos(theta[k]), 0.0]
        q_curr = solve_torch_tip_ik(X_d[k], q_curr, max_steps=100, tol=1e-4)
        q_d[k] = q_curr
    _update_ik_report(q_d, X_d)
    return t_arr, X_d, Xdot_d, q_d


def interp_traj(t_arr, arr, t):
    """Linear interpolation of a sampled trajectory at wall-clock time t (clamped)."""
    t = float(np.clip(t, t_arr[0], t_arr[-1]))
    k = int(np.clip(np.searchsorted(t_arr, t, side='right') - 1, 0, len(t_arr) - 2))
    a = (t - t_arr[k]) / (t_arr[k + 1] - t_arr[k])
    return (1.0 - a) * arr[k] + a * arr[k + 1]


# ----------------------------------------------------------------------------------------
# Dynamics: gravity vector via Lagrangian (Tutorial 9):  U = sum m_i g^T p_cm_i ,  G = dU/dq
# ----------------------------------------------------------------------------------------
UR5E_MASSES = np.array([3.761, 8.058, 2.846, 1.370, 1.300, 0.365])
GRAVITY_VEC = np.array([0.0, 0.0, -9.81])


def compute_gravity_torques(q):
    """G(q) = dU/dq (central difference). COM of link i approximated at the midpoint of its frames."""
    eps = 1e-5
    q = np.asarray(q, dtype=float)

    def potential_energy(q_vec):
        T_list = forward_kinematics(q_vec)
        U = 0.0
        for i in range(6):
            p_cm = 0.5 * (T_list[i][:3, 3] + T_list[i + 1][:3, 3])
            U -= UR5E_MASSES[i] * np.dot(GRAVITY_VEC, p_cm)
        return U

    G = np.zeros(6)
    for j in range(6):
        qp, qm = q.copy(), q.copy()
        qp[j] += eps
        qm[j] -= eps
        G[j] = (potential_energy(qp) - potential_energy(qm)) / (2.0 * eps)
    return G


# ----------------------------------------------------------------------------------------
# Controllers
# ----------------------------------------------------------------------------------------
class SISOJointPIDController:
    """
    SISO PID per joint (Tutorial 8) + gravity compensation (Tutorial 9).
    MuJoCo uses position servos, so the PID correction is a joint-angle offset and the gravity
    torque is converted to the equivalent servo offset G(q)/kp_servo (the servo error needed to
    hold that torque). Trajectory feed-forward: q_d + lead * qdot_d.
    """
    def __init__(self, kp=0.80, ki=0.35, kd=0.05, integral_limit=0.25, lead=0.20,
                 servo_kp=SERVO_KP, use_gravity_comp=True):
        self.kp = np.full(6, kp)
        self.ki = np.full(6, ki)
        self.kd = np.full(6, kd)
        self.integral_limit = integral_limit
        self.lead = lead
        self.servo_kp = np.asarray(servo_kp, float)
        self.use_gravity_comp = use_gravity_comp
        self.integral_err = np.zeros(6)
        self.name = "SISO PID + G(q)"

    def reset(self):
        self.integral_err = np.zeros(6)

    def step(self, q_desired, q_actual, qdot_desired, qdot_actual, dt):
        dt = max(dt, 1e-3)
        q_desired, q_actual = np.asarray(q_desired, float), np.asarray(q_actual, float)
        err = (q_desired - q_actual + np.pi) % (2.0 * np.pi) - np.pi

        self.integral_err = np.clip(self.integral_err + err * dt, -self.integral_limit, self.integral_limit)
        d_err = np.asarray(qdot_desired) - np.asarray(qdot_actual)
        u_pid = self.kp * err + self.ki * self.integral_err + self.kd * d_err

        tau_grav = compute_gravity_torques(q_actual) if self.use_gravity_comp else np.zeros(6)
        q_cmd = q_desired + self.lead * np.asarray(qdot_desired) + u_pid + tau_grav / self.servo_kp
        tau_equiv = self.servo_kp * (q_cmd - q_actual)
        return q_cmd, tau_equiv, err


class MIMOFuzzyLogicController:
    """
    Mamdani FLC (Tutorial 8).
      Inputs (per Cartesian axis): error e = x_d - x_a  and change of error de/dt
      Sets: LN, MN, S, MP, LP  -> triangular MFs centred at linspace(-max, max, 5)
      Rule base: 5x5 table (rows = e, cols = de), AND = min, aggregation = max
      Defuzzification: centroid of the aggregated output set (centre of gravity)
      Output: Cartesian tip velocity (vx, vy, vz) -> joint rates qdot = J^+ v (DLS pseudo-inverse)
    """
    LABELS = ["LN", "MN", "S", "MP", "LP"]
    RULE_BASE = np.array([
        [0, 0, 0, 1, 2],
        [0, 0, 1, 2, 3],
        [0, 1, 2, 3, 4],
        [1, 2, 3, 4, 4],
        [2, 3, 4, 4, 4],
    ])

    def __init__(self, max_err=0.04, max_de=0.30, max_vel=0.25, fb_horizon=0.08, lead=0.20,
                 servo_kp=SERVO_KP, use_gravity_comp=True, de_filter=0.5):
        self.max_err, self.max_de, self.max_vel = max_err, max_de, max_vel
        self.fb_horizon, self.lead = fb_horizon, lead
        self.servo_kp = np.asarray(servo_kp, float)
        self.use_gravity_comp = use_gravity_comp
        self.de_filter = de_filter

        self.in_e_centers = np.linspace(-max_err, max_err, 5)
        self.in_de_centers = np.linspace(-max_de, max_de, 5)
        self.out_centers = np.linspace(-max_vel, max_vel, 5)
        self.z_universe = np.linspace(-max_vel, max_vel, 201)
        step_out = max_vel / 2.0
        self.out_mfs = np.clip(1.0 - np.abs(self.z_universe[None, :] - self.out_centers[:, None]) / step_out, 0, 1)
        self.prev_err = None
        self.prev_de = np.zeros(3)
        self.name = "MIMO Mamdani FLC"

    def reset(self):
        self.prev_err = None
        self.prev_de = np.zeros(3)

    @staticmethod
    def _fuzzify(x, centers, max_x):
        x = float(np.clip(x, -max_x, max_x))
        return np.clip(1.0 - np.abs(x - centers) / (max_x / 2.0), 0.0, 1.0)

    def infer_axis(self, e, de):
        """Fuzzify -> rule base (min) -> max aggregation -> centroid."""
        mu_e = self._fuzzify(e, self.in_e_centers, self.max_err)
        mu_de = self._fuzzify(de, self.in_de_centers, self.max_de)
        act = np.zeros(5)
        for i in range(5):
            if mu_e[i] <= 0.0:
                continue
            for j in range(5):
                f = min(mu_e[i], mu_de[j])
                if f > 0.0:
                    k = self.RULE_BASE[i, j]
                    act[k] = max(act[k], f)
        agg = np.max(np.minimum(act[:, None], self.out_mfs), axis=0)
        area = np.sum(agg)
        return 0.0 if area < 1e-9 else float(np.sum(self.z_universe * agg) / area)

    def step(self, pos_desired, pos_actual, vel_feedforward, q_desired, q_actual, dt, qdot_desired=None):
        dt = max(dt, 1e-3)
        err = np.asarray(pos_desired, float) - np.asarray(pos_actual, float)
        if self.prev_err is None:
            d_err = np.zeros(3)
        else:
            d_raw = (err - self.prev_err) / dt
            d_err = self.de_filter * d_raw + (1.0 - self.de_filter) * self.prev_de
        self.prev_err, self.prev_de = err.copy(), d_err.copy()

        v_flc = np.array([self.infer_axis(err[a], d_err[a]) for a in range(3)])

        J = compute_torch_tip_jacobian(q_actual)
        J_pinv = J.T @ np.linalg.inv(J @ J.T + (0.01 ** 2) * np.eye(3))
        qdot_fb = J_pinv @ v_flc
        m = np.max(np.abs(qdot_fb))
        if m > 1.5:
            qdot_fb *= 1.5 / m

        qd_ff = np.zeros(6) if qdot_desired is None else np.asarray(qdot_desired, float)
        tau_grav = compute_gravity_torques(q_actual) if self.use_gravity_comp else np.zeros(6)
        q_cmd = (np.asarray(q_desired, float) + self.lead * qd_ff
                 + self.fb_horizon * qdot_fb + tau_grav / self.servo_kp)

        v_cmd = v_flc + np.asarray(vel_feedforward, float)
        return q_cmd, qdot_fb, v_cmd, err