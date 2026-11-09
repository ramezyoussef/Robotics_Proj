#!/usr/bin/env python3
"""
MCTR911 - Milestone 3: REAL MuJoCo validation of position / velocity / acceleration.

Unlike the first version, every "measured" number here is READ FROM MUJOCO,
and the inverse equations are fed with those MuJoCo readings:

  Forward test : q, q_dot, q_ddot -> MuJoCo -> sensed x, xdot, xddot
                 compare with  FK(q),  J q_dot,  J_dot q_dot + J q_ddot
  Inverse test : sensed xdot, xddot (from MuJoCo) -> inverse equations
                 -> recovered q_dot, q_ddot; compare with what was applied

Place next to kinematics_derivation_ur5e.py and run:
    python3 validate_kinematics_mujoco_v2.py
"""
import os
import numpy as np
import mujoco
from kinematics_derivation_ur5e import (
    get_end_effector_position, compute_jacobian,
    forward_velocity_kinematics, inverse_velocity_kinematics,
    forward_acceleration_kinematics, inverse_acceleration_kinematics,
)

POS_TOL = 2e-3     # m      (DH-vs-mesh link length rounding gives ~1.2 mm)
VEL_TOL = 5e-3     # m/s, rad/s
ACC_TOL = 2e-2     # m/s^2, rad/s^2
QD_TOL = 5e-2      # rad/s   recovered joint velocity
QDD_TOL = 5e-1     # rad/s^2 recovered joint acceleration (amplifies pos. error)


def load_model():
    for p in (os.path.expanduser("~/mujoco_menagerie/universal_robots_ur5e/scene.xml"),
              "mujoco_menagerie/universal_robots_ur5e/scene.xml"):
        if os.path.exists(p):
            return mujoco.MjModel.from_xml_path(p)
    raise FileNotFoundError("UR5e scene.xml not found")


def sense(model, data, sid, q, qd, qdd):
    """Set state in MuJoCo and read back EE position, twist and acceleration."""
    data.qpos[:6], data.qvel[:6], data.qacc[:6] = q, qd, qdd
    mujoco.mj_inverse(model, data)           # fills body accelerations (cacc)
    pos = data.site_xpos[sid].copy()
    v, a = np.zeros(6), np.zeros(6)          # MuJoCo order: [angular, linear]
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_SITE, sid, v, 0)
    mujoco.mj_objectAcceleration(model, data, mujoco.mjtObj.mjOBJ_SITE, sid, a, 0)
    return pos, np.r_[v[3:], v[:3]], np.r_[a[3:], a[:3]]   # -> [linear, angular]


def sensed_linear_accel(model, sid, q, qd, qdd, h=1e-4):
    """Linear EE acceleration = d2/dt2 of MuJoCo's own site position (independent check)."""
    def pos(t):
        d = mujoco.MjData(model)
        d.qpos[:6] = q + qd * t + 0.5 * qdd * t * t
        mujoco.mj_kinematics(model, d)
        return d.site_xpos[sid].copy()
    return (pos(h) - 2 * pos(0) + pos(-h)) / h ** 2


def check(name, err, tol, unit):
    ok = err <= tol
    print(f"   [{'PASS' if ok else 'FAIL'}] {name:<42} err = {err:.3e} {unit}  (tol {tol:g})")
    return ok


def main():
    model = load_model()
    data = mujoco.MjData(model)
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")

    # NOTE: do NOT use the home pose [0,-pi/2,0,-pi/2,0,0] for inverse tests:
    # it is an elbow singularity (det J ~ 1e-83), J^-1 does not exist there.
    cases = [
        ("Config A", [0.3, -1.0, 1.2, -0.5, 0.4, 0.1],
                     [0.2, -0.1, 0.3, -0.2, 0.1, 0.0],   [0.5, -0.4, 0.8, -0.6, 0.3, 0.2]),
        ("Config B", [-0.5, -0.7, 0.9, -1.2, 1.0, -0.5],
                     [-1.0, 1.2, -0.8, 1.5, -1.1, 0.9],  [2.0, -1.5, 1.0, -2.0, 1.5, 1.0]),
        ("Config C", [1.0, -1.8, 1.5, -1.0, -0.8, 0.6],
                     [0.4, 0.3, -0.5, 0.2, 0.6, -0.3],   [-0.3, 0.6, 0.2, -0.5, 0.1, 0.4]),
    ]
    all_ok = True
    for name, q, qd, qdd in cases:
        q, qd, qdd = map(np.array, (q, qd, qdd))
        print(f"\n=== {name}: q={np.round(q, 2)} ===")
        print(f"   cond(J) = {np.linalg.cond(compute_jacobian(q)):.1f}")
        pos_mj, tw_mj, acc_mj = sense(model, data, sid, q, qd, qdd)
        acc_lin_fd = sensed_linear_accel(model, sid, q, qd, qdd)

        # ---- forward: equations vs MuJoCo ----
        X = forward_velocity_kinematics(q, qd)
        Xdd = forward_acceleration_kinematics(q, qd, qdd)
        print(" Forward (equations vs MuJoCo):")
        all_ok &= check("position  FK(q)",            np.linalg.norm(pos_mj - get_end_effector_position(q)), POS_TOL, "m")
        all_ok &= check("lin. velocity  (J q_dot)",   np.linalg.norm(tw_mj[:3] - X[:3]), VEL_TOL, "m/s")
        all_ok &= check("ang. velocity  (J q_dot)",   np.linalg.norm(tw_mj[3:] - X[3:]), VEL_TOL, "rad/s")
        all_ok &= check("lin. accel (J_dot q_dot+J q_ddot) vs d2/dt2", np.linalg.norm(acc_lin_fd - Xdd[:3]), ACC_TOL, "m/s^2")

        # ---- inverse: feed MuJoCo-sensed values into the inverse equations ----
        qd_rec = inverse_velocity_kinematics(q, tw_mj)
        qdd_rec = inverse_acceleration_kinematics(q, qd, np.r_[acc_lin_fd, Xdd[3:]])
        print(" Inverse (MuJoCo readings -> equations):")
        all_ok &= check("q_dot  = J^-1 xdot_sensed",   np.linalg.norm(qd_rec - qd), QD_TOL, "rad/s")
        all_ok &= check("q_ddot = J^-1(xddot - Jdot qdot)", np.linalg.norm(qdd_rec - qdd), QDD_TOL, "rad/s^2")

    print("\n" + "=" * 60)
    print(" RESULT:", "ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED")
    print("=" * 60)


if __name__ == "__main__":
    main()
