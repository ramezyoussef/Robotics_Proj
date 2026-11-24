#!/usr/bin/env python3
"""
MCTR911 - Milestone 4: Unified Industrial UR5e Welding Cell GUI & Closed-Loop Controller
- Wall-clock trajectory execution (real elapsed time, not a nominal 30 ms tick)
- Controller selector (FLC / PID) for the task-space seam and circle -> same-trajectory comparison
- Per-run metrics (RMS, max error, min Z, max torch tilt) saved to m4_results_summary.csv
"""

import csv
import os
import re
import time
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, Bool, Empty, String
from sensor_msgs.msg import JointState

from milestone4_trajectories_and_control import (
    READY_Q,
    get_torch_tip_position,
    compute_torch_tip_jacobian,
    solve_torch_tip_ik,
    torch_tilt_deg,
    torch_tilt_from_vertical_deg,
    interp_traj,
    generate_joint_space_trajectory,
    generate_task_space_linear,
    generate_task_space_circle,
    SISOJointPIDController,
    MIMOFuzzyLogicController,
)

# Torch-tip targets (TOP_Z = 0.470 m, BEAD_MARGIN = 0.015 m -> z = 0.480 draws the bead without penetration)
WELD_SAFE_ABOVE = np.array([0.45, 0.00, 0.600])
WELD_SEAM_START = np.array([0.38, 0.00, 0.480])
WELD_SEAM_END   = np.array([0.52, 0.00, 0.480])
WELD_CIRCLE_CTR = np.array([0.45, 0.05, 0.480])
WELD_CIRCLE_RAD = 0.045
BOX_TOP_Z = 0.470

SUMMARY_CSV = "m4_results_summary.csv"


class UR5eM4Node(Node):
    def __init__(self):
        super().__init__('ur5e_m4_closed_loop_node')
        self.pub_joints = self.create_publisher(Float64MultiArray, '/joint_commands', 10)
        self.pub_weld = self.create_publisher(Bool, '/welding_active', 10)
        self.pub_clear = self.create_publisher(Empty, '/weld_clear', 10)
        self.pub_conv = self.create_publisher(Bool, '/conveyor_active', 10)
        self.pub_auto = self.create_publisher(Empty, '/auto_weld', 10)

        self.sub_states = self.create_subscription(JointState, '/joint_states', self.joint_state_cb, 10)
        self.sub_auto = self.create_subscription(String, '/auto_weld_status', self.auto_status_cb, 10)

        self.auto_status = "IDLE"
        self.current_q = READY_Q.copy()
        self.current_qdot = np.zeros(6)

    def joint_state_cb(self, msg):
        if len(msg.position) >= 6:
            self.current_q = np.array(msg.position[:6], dtype=float)
        if len(msg.velocity) >= 6:
            self.current_qdot = np.array(msg.velocity[:6], dtype=float)

    def auto_status_cb(self, msg):
        self.auto_status = msg.data

    def send_joints(self, q):
        msg = Float64MultiArray()
        msg.data = [float(x) for x in q]
        self.pub_joints.publish(msg)

    def set_welding(self, active: bool):
        msg = Bool()
        msg.data = bool(active)
        self.pub_weld.publish(msg)

    def clear_weld_bead(self):
        self.pub_clear.publish(Empty())

    def set_conveyor(self, active: bool):
        msg = Bool()
        msg.data = bool(active)
        self.pub_conv.publish(msg)


class Milestone4WeldingGUI:
    SETTLE_TIME = 0.20

    def __init__(self, root, node):
        self.root = root
        self.node = node
        self.root.title("UR5e Milestone 4 - Trajectory Planning & Closed-Loop Control")
        self.root.geometry("720x1000")

        self.updating_from_code = True
        self.active_q = list(self.node.current_q)
        self.welding_active = False
        self.record_active = True

        self.pid_controller = SISOJointPIDController(kp=0.80, ki=0.35, kd=0.05)
        self.flc_controller = MIMOFuzzyLogicController(max_err=0.04, max_de=0.30, max_vel=0.25)
        self.ctrl_choice = tk.StringVar(value="FLC")

        self.exec_mode = None
        self.traj_t = None
        self.traj_q_d = None
        self.traj_qdot_d = None
        self.traj_X_d = None
        self.traj_Xdot_d = None
        self.traj_dt = 0.03
        self.traj_t0 = 0.0
        self.last_tick = time.monotonic()
        self.traj_label = ""
        self.ctrl_name = ""
        self.on_traj_complete = None
        self.min_z_all = np.inf

        self.log_t, self.log_q_d, self.log_q_a = [], [], []
        self.log_X_d, self.log_X_a, self.log_ctrl, self.log_tilt = [], [], [], []

        self._build_ui()
        self.sync_cart_from_q(self.active_q)
        self.updating_from_code = False
        self.node.set_welding(False)
        self.node.set_conveyor(True)
        self.update_loop()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        mon_frame = ttk.LabelFrame(self.root, text="1. Live Torch-Tip Kinematic & Closed-Loop Error Telemetry", padding=8)
        mon_frame.pack(fill="x", padx=10, pady=4)
        grid = ttk.Frame(mon_frame)
        grid.pack(fill="x")

        ttk.Label(grid, text="Torch-Tip Pose (m):", font=("Helvetica", 9, "bold")).grid(row=0, column=0, sticky="w")
        self.lbl_pos = ttk.Label(grid, text="X: +0.4493 | Y: -0.0006 | Z: +0.5697", font=("Courier", 9, "bold"), foreground="#0066cc")
        self.lbl_pos.grid(row=0, column=1, sticky="w", padx=6)

        ttk.Label(grid, text="Tip Velocity (m/s):", font=("Helvetica", 9, "bold")).grid(row=1, column=0, sticky="w")
        self.lbl_vel = ttk.Label(grid, text="Vx: +0.0000 | Vy: +0.0000 | Vz: +0.0000", font=("Courier", 9, "bold"), foreground="#009933")
        self.lbl_vel.grid(row=1, column=1, sticky="w", padx=6)

        ttk.Label(grid, text="Closed-Loop Error:", font=("Helvetica", 9, "bold")).grid(row=2, column=0, sticky="w")
        self.lbl_err = ttk.Label(grid, text="IDLE (No active trajectory)", font=("Courier", 9, "bold"), foreground="#cc3300")
        self.lbl_err.grid(row=2, column=1, sticky="w", padx=6)

        ttk.Label(grid, text="Torch Tilt (deg):", font=("Helvetica", 9, "bold")).grid(row=3, column=0, sticky="w")
        self.lbl_tilt = ttk.Label(grid, text="--", font=("Courier", 9, "bold"), foreground="#7a3db8")
        self.lbl_tilt.grid(row=3, column=1, sticky="w", padx=6)

        m4_frame = ttk.LabelFrame(self.root, text="2. Milestone 4: Trajectory Planning & Closed-Loop Motor Control", padding=8)
        m4_frame.pack(fill="x", padx=10, pady=4)

        sel = ttk.Frame(m4_frame)
        sel.pack(fill="x", pady=3)
        ttk.Label(sel, text="Task-space controller:").pack(side="left")
        ttk.Combobox(sel, textvariable=self.ctrl_choice, values=["FLC", "PID"], width=6, state="readonly").pack(side="left", padx=6)
        ttk.Label(sel, text="(run seam/circle once with each for the comparison table)", foreground="#666").pack(side="left")

        row1 = ttk.Frame(m4_frame)
        row1.pack(fill="x", pady=3)
        ttk.Button(row1, text="1) Joint-Space Quintic + SISO PID (Hover Pose)",
                   command=self.run_joint_space_demo).pack(side="left", expand=True, fill="x", padx=2)
        ttk.Button(row1, text="2) Task-Space Linear Seam",
                   command=self.run_task_space_linear_demo).pack(side="right", expand=True, fill="x", padx=2)

        row2 = ttk.Frame(m4_frame)
        row2.pack(fill="x", pady=3)
        ttk.Button(row2, text="3) Task-Space Circular Weld",
                   command=self.run_task_space_circle_demo).pack(side="left", expand=True, fill="x", padx=2)
        ttk.Button(row2, text="★ Run Full Industrial Welding Cycle ★",
                   command=self.run_full_industrial_cycle).pack(side="right", expand=True, fill="x", padx=2)

        row3 = ttk.Frame(m4_frame)
        row3.pack(fill="x", pady=3)
        ttk.Button(row3, text="■ Stop / Abort Trajectory", command=self.abort_trajectory).pack(side="left", expand=True, fill="x", padx=2)
        ttk.Button(row3, text="📊 Save Validation Report Plots", command=self.save_validation_plots).pack(side="right", expand=True, fill="x", padx=2)

        self.lbl_m4_status = ttk.Label(m4_frame, text="Controller Status: READY", font=("Courier", 9, "bold"), foreground="#005580", wraplength=660, justify="left")
        self.lbl_m4_status.pack(anchor="w", pady=2)

        cart_frame = ttk.LabelFrame(self.root, text="3. Manual Torch-Tip Cartesian Control (IK)", padding=8)
        cart_frame.pack(fill="x", padx=10, pady=4)
        self.cart_sliders, self.cart_labels = {}, {}
        for axis_name, min_v, max_v, init_v in [("X (m)", 0.25, 0.65, 0.45), ("Y (m)", -0.35, 0.35, 0.00), ("Z (m)", 0.475, 0.80, 0.600)]:
            row = ttk.Frame(cart_frame)
            row.pack(fill="x", pady=1)
            lbl = ttk.Label(row, text=f"{axis_name}: {init_v:6.3f}", width=16)
            lbl.pack(side="left")
            self.cart_labels[axis_name] = lbl
            slider = ttk.Scale(row, from_=min_v, to=max_v, orient="horizontal",
                               command=lambda val, a=axis_name: self.on_cart_slider(a, val))
            self.cart_sliders[axis_name] = slider
            slider.set(init_v)
            slider.pack(side="right", fill="x", expand=True)

        joint_frame = ttk.LabelFrame(self.root, text="4. Manual Joint Position Control (Direct / FK)", padding=8)
        joint_frame.pack(fill="x", padx=10, pady=4)
        self.joint_sliders, self.joint_labels = [], []
        for i, name in enumerate(["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]):
            row = ttk.Frame(joint_frame)
            row.pack(fill="x", pady=1)
            lbl = ttk.Label(row, text=f"{name[:12]}: 0.00", width=18)
            lbl.pack(side="left")
            self.joint_labels.append(lbl)
            slider = ttk.Scale(row, from_=-6.2832, to=6.2832, orient="horizontal",
                               command=lambda val, idx=i: self.on_joint_slider(idx, val))
            self.joint_sliders.append(slider)
            slider.set(self.active_q[i])
            slider.pack(side="right", fill="x", expand=True)

        ttk.Button(self.root, text="Move to Welding Ready Pose (Elbow-Up)", command=self.preset_ready).pack(fill="x", padx=12, pady=2)

        env_frame = ttk.LabelFrame(self.root, text="5. Welding Torch & Conveyor Workcell Controls", padding=8)
        env_frame.pack(fill="x", padx=10, pady=4)
        weld_row = ttk.Frame(env_frame)
        weld_row.pack(fill="x", pady=2)
        self.weld_btn = ttk.Button(weld_row, text="⚪ Welding Torch: OFF", command=self.toggle_welding)
        self.weld_btn.pack(side="left", expand=True, fill="x", padx=2)
        ttk.Button(weld_row, text="Clear Weld Bead", command=self.node.clear_weld_bead).pack(side="right", expand=True, fill="x", padx=2)

        self.conveyor_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(env_frame, text="Infeed Conveyor Active", variable=self.conveyor_var,
                        command=lambda: self.node.set_conveyor(self.conveyor_var.get())).pack(anchor="w")

    # ------------------------------------------------------------------ helpers
    def _reset_logs(self, label):
        self.traj_label = label
        for lst in (self.log_t, self.log_q_d, self.log_q_a, self.log_X_d, self.log_X_a, self.log_ctrl, self.log_tilt):
            lst.clear()

    def _ik(self, target, seed):
        """IK wrapper: warns in the status bar if the solution is not accurate / not upright."""
        q, info = solve_torch_tip_ik(target, seed, return_info=True)
        if info["pos_err"] > 1e-3 or info["tilt_deg"] > 5.0:
            msg = f"IK WARNING: err {info['pos_err']*1000:.2f} mm, tilt {info['tilt_deg']:.1f} deg for target {np.round(target, 3)}"
            print(msg)
            self.lbl_m4_status.config(text=msg)
        return q

    def _begin(self, mode, label, disp_ctrl, on_done):
        self.exec_mode = mode
        self.on_traj_complete = on_done
        self.traj_t0 = time.monotonic()
        self.last_tick = self.traj_t0
        self.lbl_m4_status.config(text=f"RUNNING: {label} [{disp_ctrl}] ({self.traj_t[-1]:.1f} s)")

    def _reset_run_metrics(self):
        self.min_z_all = np.inf

    # ------------------------------------------------------------------ trajectory starters
    def start_joint_space_trajectory(self, q_target, tf=3.0, label="Joint-Space Quintic + SISO PID", on_done=None, record_log=True):
        q_start = self.node.current_q.copy()
        t_arr, q_d, qdot_d, _ = generate_joint_space_trajectory(q_start, q_target, tf=tf, dt=self.traj_dt)
        self.pid_controller.reset()
        self.record_active = record_log
        if record_log:
            self._reset_logs(label)
            self.ctrl_name = self.pid_controller.name
        self.traj_t, self.traj_q_d, self.traj_qdot_d = t_arr, q_d, qdot_d
        self.traj_X_d = np.array([get_torch_tip_position(q) for q in q_d])
        self.traj_Xdot_d = None
        self._begin("JOINT_PID", label, self.pid_controller.name, on_done)

    def _start_task(self, t_arr, X_d, Xdot_d, q_d, label, on_done):
        self.record_active = True
        self._reset_logs(label)
        self.traj_t, self.traj_X_d, self.traj_Xdot_d, self.traj_q_d = t_arr, X_d, Xdot_d, q_d
        self.traj_qdot_d = np.gradient(q_d, self.traj_dt, axis=0)
        if self.ctrl_choice.get() == "PID":
            self.pid_controller.reset()
            self.ctrl_name = self.pid_controller.name
            mode = "TASK_PID"
        else:
            self.flc_controller.reset()
            self.ctrl_name = self.flc_controller.name
            mode = "TASK_FLC"
        self._begin(mode, label, self.ctrl_name, on_done)

    def start_task_space_linear(self, p_start, p_end, tf=4.0, label="Task-Space Linear Seam", on_done=None):
        t_arr, X_d, Xdot_d, q_d = generate_task_space_linear(p_start, p_end, self.node.current_q.copy(), tf=tf, dt=self.traj_dt)
        self._start_task(t_arr, X_d, Xdot_d, q_d, label, on_done)

    def start_task_space_circle(self, center, radius, tf=5.0, label="Task-Space Circular Patch", on_done=None):
        t_arr, X_d, Xdot_d, q_d = generate_task_space_circle(center, radius, self.node.current_q.copy(), tf=tf, dt=self.traj_dt, laps=1.0)
        self._start_task(t_arr, X_d, Xdot_d, q_d, label, on_done)

    # ------------------------------------------------------------------ demos
    def run_joint_space_demo(self):
        self._reset_run_metrics()
        q_hover = self._ik(WELD_SAFE_ABOVE, self.node.current_q.copy())
        self.start_joint_space_trajectory(q_hover, tf=3.0, label="Joint-Space Quintic + SISO PID", record_log=True)

    def run_task_space_linear_demo(self):
        self._reset_run_metrics()
        q_hover_start = self._ik(np.array([0.38, 0.0, 0.600]), self.node.current_q.copy())
        q_seam_start = self._ik(WELD_SEAM_START, q_hover_start)

        def plunge_to_start():
            self.start_joint_space_trajectory(q_seam_start, tf=1.5, label="Lower Torch to Seam", on_done=begin_linear, record_log=False)

        def begin_linear():
            self.set_weld_state(True)
            self.start_task_space_linear(WELD_SEAM_START, WELD_SEAM_END, tf=3.5,
                                         label="Task-Space Linear Seam", on_done=lift_after_weld)

        def lift_after_weld():
            self.set_weld_state(False)
            self.start_joint_space_trajectory(q_hover_start, tf=1.5, label="Safe Lift After Seam", record_log=False)

        self.start_joint_space_trajectory(q_hover_start, tf=2.0, label="Move Above Seam Start", on_done=plunge_to_start, record_log=False)

    def run_task_space_circle_demo(self):
        self._reset_run_metrics()
        p_circle_start = WELD_CIRCLE_CTR + np.array([WELD_CIRCLE_RAD, 0.0, 0.0])
        q_hover_c = self._ik(p_circle_start + np.array([0.0, 0.0, 0.12]), self.node.current_q.copy())
        q_circle_start = self._ik(p_circle_start, q_hover_c)

        def plunge_circle():
            self.start_joint_space_trajectory(q_circle_start, tf=1.5, label="Lower Torch to Circle", on_done=begin_circle, record_log=False)

        def begin_circle():
            self.set_weld_state(True)
            self.start_task_space_circle(WELD_CIRCLE_CTR, WELD_CIRCLE_RAD, tf=5.0,
                                         label="Task-Space Circular Patch", on_done=lift_circle)

        def lift_circle():
            self.set_weld_state(False)
            self.start_joint_space_trajectory(q_hover_c, tf=1.5, label="Safe Lift After Circle", record_log=False)

        self.start_joint_space_trajectory(q_hover_c, tf=2.0, label="Move Above Circle Start", on_done=plunge_circle, record_log=False)

    def run_full_industrial_cycle(self):
        self._reset_run_metrics()
        self.node.clear_weld_bead()
        self.conveyor_var.set(False)
        self.node.set_conveyor(False)
        self.set_weld_state(False)

        q_seed = self.node.current_q.copy()
        q_hover_s = self._ik(np.array([0.38, 0.0, 0.600]), q_seed)
        q_seam_s = self._ik(WELD_SEAM_START, q_hover_s)
        q_hover_e = self._ik(np.array([0.52, 0.0, 0.600]), q_seam_s)
        q_ready_unwrapped = READY_Q + 2.0 * np.pi * np.round((q_hover_e - READY_Q) / (2.0 * np.pi))

        def step2_plunge():
            self.start_joint_space_trajectory(q_seam_s, tf=1.5, label="Step 2: Seam Plunge (Quintic + PID)", on_done=step3_weld, record_log=False)

        def step3_weld():
            self.set_weld_state(True)
            self.start_task_space_linear(WELD_SEAM_START, WELD_SEAM_END, tf=3.5, label="Full Welding Cycle: Seam Weld", on_done=step4_lift)

        def step4_lift():
            self.set_weld_state(False)
            self.start_joint_space_trajectory(q_hover_e, tf=1.5, label="Step 4: Safe Lift Above Box", on_done=step5_retract, record_log=False)

        def step5_retract():
            self.start_joint_space_trajectory(q_ready_unwrapped, tf=2.0, label="Step 5: Return Welding Ready (Quintic + PID)", on_done=step6_resume_conv, record_log=False)

        def step6_resume_conv():
            self.conveyor_var.set(True)
            self.node.set_conveyor(True)
            self.lbl_m4_status.config(text="COMPLETED: Full Industrial Welding Cycle!")
            self.save_validation_plots(silent=True)

        self.start_joint_space_trajectory(q_hover_s, tf=2.0, label="Step 1: Approach Hover (Quintic + PID)", on_done=step2_plunge, record_log=False)

    def abort_trajectory(self):
        self.exec_mode = None
        self.on_traj_complete = None
        self.set_weld_state(False)
        self.lbl_m4_status.config(text="Controller Status: ABORTED")

    def set_weld_state(self, active: bool):
        self.welding_active = active
        self.weld_btn.config(text="⚡ Welding Torch: ACTIVE (ON)" if active else "⚪ Welding Torch: OFF")
        self.node.set_welding(active)

    def toggle_welding(self):
        self.set_weld_state(not self.welding_active)

    # ------------------------------------------------------------------ metrics / plots
    def _compute_metrics(self):
        X_d, X_a = np.array(self.log_X_d), np.array(self.log_X_a)
        err_mm = np.linalg.norm(X_d - X_a, axis=1) * 1000.0
        return {
            "run": self.traj_label,
            "controller": self.ctrl_name,
            "samples": len(err_mm),
            "duration_s": round(float(self.log_t[-1] - self.log_t[0]), 3),
            "rms_err_mm": round(float(np.sqrt(np.mean(err_mm ** 2))), 3),
            "max_err_mm": round(float(np.max(err_mm)), 3),
            "final_err_mm": round(float(err_mm[-1]), 3),
            "min_tip_z_logged_m": round(float(np.min(X_a[:, 2])), 4),
            "min_tip_z_all_legs_m": round(float(self.min_z_all), 4),
            "clearance_to_box_mm": round((float(self.min_z_all) - BOX_TOP_Z) * 1000.0, 1),
            "max_tilt_deg": round(float(np.max(self.log_tilt)), 3),
        }

    def _append_summary(self, m):
        new = not os.path.exists(SUMMARY_CSV)
        with open(SUMMARY_CSV, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(m.keys()))
            if new:
                w.writeheader()
            w.writerow(m)

    def save_validation_plots(self, silent=False):
        if len(self.log_t) < 5:
            if not silent:
                messagebox.showwarning("No Data", "Run a Joint-Space or Task-Space trajectory first to record telemetry.")
            return

        t = np.array(self.log_t)
        q_d, q_a = np.array(self.log_q_d), np.array(self.log_q_a)
        X_d, X_a = np.array(self.log_X_d), np.array(self.log_X_a)
        tilt = np.array(self.log_tilt)
        met = self._compute_metrics()

        fig, axes = plt.subplots(4, 1, figsize=(10, 14), sharex=True)
        fig.suptitle(f"MCTR911 Milestone 4 Validation: {self.traj_label}  [{self.ctrl_name}]", fontsize=13, fontweight="bold")
        colors, labels = ['r', 'g', 'b'], ['X', 'Y', 'Z']

        for i in range(3):
            axes[0].plot(t, X_d[:, i], '--', color=colors[i], label=f"{labels[i]}_desired")
            axes[0].plot(t, X_a[:, i], '-', color=colors[i], linewidth=1.8, label=f"{labels[i]}_actual")
        axes[0].set_ylabel("Torch-Tip Position (m)")
        axes[0].set_title("Task-Space Torch-Tip Tracking (Desired vs. Actual)")
        axes[0].legend(ncol=3, fontsize=8)
        axes[0].grid(True, alpha=0.3)

        for j in range(3):
            axes[1].plot(t, q_d[:, j], '--', color=colors[j], label=f"q{j+1}_desired")
            axes[1].plot(t, q_a[:, j], '-', color=colors[j], linewidth=1.8, label=f"q{j+1}_actual")
        axes[1].set_ylabel("Joint Angle (rad)")
        axes[1].set_title("Joint-Space Tracking (Shoulder & Elbow Joints)")
        axes[1].legend(ncol=3, fontsize=8)
        axes[1].grid(True, alpha=0.3)

        err_mm = np.linalg.norm(X_d - X_a, axis=1) * 1000.0
        axes[2].plot(t, err_mm, color='purple', linewidth=2.0, label="Torch-Tip Error Norm (mm)")
        axes[2].set_ylabel("Error (mm)")
        axes[2].set_title(f"Closed-Loop Tracking Error (RMS = {met['rms_err_mm']:.2f} mm, max = {met['max_err_mm']:.2f} mm)")
        axes[2].legend(fontsize=9)
        axes[2].grid(True, alpha=0.3)

        axes[3].plot(t, tilt, color='darkorange', linewidth=2.0, label="Torch tilt vs. ready orientation (deg)")
        axes[3].set_xlabel("Time (s)")
        axes[3].set_ylabel("Tilt (deg)")
        axes[3].set_title(f"Torch Orientation Hold (max = {met['max_tilt_deg']:.2f} deg)")
        axes[3].legend(fontsize=9)
        axes[3].grid(True, alpha=0.3)

        plt.tight_layout()
        slug = re.sub(r'[^A-Za-z0-9]+', '_', f"{self.traj_label}_{self.ctrl_name}").strip('_')[:60]
        filename = f"m4_{slug}.png"
        plt.savefig(filename, dpi=150)
        plt.close(fig)

        self.lbl_m4_status.config(text=f"Saved {filename} | RMS {met['rms_err_mm']:.2f} mm, max {met['max_err_mm']:.2f} mm, "
                                       f"tilt {met['max_tilt_deg']:.2f} deg, min Z {met['min_tip_z_all_legs_m']:.4f} m")
        if not silent:
            messagebox.showinfo("Validation Plot Saved", f"Saved:\n{filename}\nMetrics appended to {SUMMARY_CSV}")

    # ------------------------------------------------------------------ manual control
    def on_cart_slider(self, axis_name, val):
        if self.updating_from_code or self.exec_mode is not None:
            return
        self.cart_labels[axis_name].config(text=f"{axis_name}: {float(val):6.3f}")
        target_xyz = np.array([self.cart_sliders["X (m)"].get(), self.cart_sliders["Y (m)"].get(), self.cart_sliders["Z (m)"].get()])
        new_q = solve_torch_tip_ik(target_xyz, self.active_q)
        self.active_q = list(new_q)
        self.node.send_joints(self.active_q)
        self.refresh_sliders_only()

    def on_joint_slider(self, idx, val):
        if self.updating_from_code or self.exec_mode is not None:
            return
        val = float(val)
        self.active_q[idx] = val
        self.joint_labels[idx].config(text=f"{self.joint_labels[idx].cget('text').split(':')[0]}: {val:6.2f}")
        self.node.send_joints(self.active_q)
        self.sync_cart_from_q(self.active_q)

    def sync_cart_from_q(self, q):
        pos = get_torch_tip_position(q)
        self.updating_from_code = True
        for (axis_name, v) in zip(["X (m)", "Y (m)", "Z (m)"], pos):
            self.cart_sliders[axis_name].set(v)
            self.cart_labels[axis_name].config(text=f"{axis_name}: {v:6.3f}")
        self.updating_from_code = False

    def refresh_sliders_only(self):
        self.updating_from_code = True
        for i, v in enumerate(self.active_q):
            self.joint_sliders[i].set(v)
            self.joint_labels[i].config(text=f"{self.joint_labels[i].cget('text').split(':')[0]}: {v:6.2f}")
        self.updating_from_code = False

    def preset_ready(self):
        self.abort_trajectory()
        self._reset_run_metrics()
        self.start_joint_space_trajectory(READY_Q, tf=2.5, label="Move to Welding Ready Pose", record_log=False)

    # ------------------------------------------------------------------ main loop (wall-clock based)
    def update_loop(self):
        rclpy.spin_once(self.node, timeout_sec=0.005)

        now = time.monotonic()
        dt = max(now - self.last_tick, 1e-3)
        self.last_tick = now

        q_actual = self.node.current_q.copy()
        qdot_actual = self.node.current_qdot.copy()
        pos_actual = get_torch_tip_position(q_actual)

        if self.exec_mode is not None:
            t_el = now - self.traj_t0
            t_end = float(self.traj_t[-1])

            if t_el <= t_end + self.SETTLE_TIME:
                in_traj = t_el <= t_end
                t_s = min(t_el, t_end)
                q_d = interp_traj(self.traj_t, self.traj_q_d, t_s)
                qdot_d = interp_traj(self.traj_t, self.traj_qdot_d, t_s) if in_traj else np.zeros(6)
                X_d = interp_traj(self.traj_t, self.traj_X_d, t_s)

                if self.exec_mode in ("JOINT_PID", "TASK_PID"):
                    q_cmd, tau_equiv, q_err = self.pid_controller.step(q_d, q_actual, qdot_d, qdot_actual, dt)
                    self.node.send_joints(q_cmd)
                    cart_err = (X_d - pos_actual) * 1000.0
                    self.lbl_err.config(text=f"PID q-err {np.linalg.norm(q_err):.4f} rad | tip err {np.linalg.norm(cart_err):.1f} mm")
                    ctrl_rec = tau_equiv
                else:
                    Xdot_d = interp_traj(self.traj_t, self.traj_Xdot_d, t_s) if in_traj else np.zeros(3)
                    q_cmd, qdot_fb, v_cmd, xyz_err = self.flc_controller.step(
                        X_d, pos_actual, Xdot_d, q_d, q_actual, dt, qdot_desired=qdot_d)
                    self.node.send_joints(q_cmd)
                    self.lbl_err.config(text=f"FLC XYZ Err: [{xyz_err[0]*1000:+.1f}, {xyz_err[1]*1000:+.1f}, {xyz_err[2]*1000:+.1f}] mm")
                    ctrl_rec = v_cmd

                self.min_z_all = min(self.min_z_all, float(pos_actual[2]))

                if self.record_active and in_traj:
                    self.log_t.append(t_s)
                    self.log_q_d.append(q_d.copy())
                    self.log_q_a.append(q_actual.copy())
                    self.log_X_d.append(X_d.copy())
                    self.log_X_a.append(pos_actual.copy())
                    self.log_ctrl.append(np.array(ctrl_rec).copy())
                    self.log_tilt.append(torch_tilt_deg(q_actual))

                self.active_q = list(q_actual)
                self.refresh_sliders_only()
                self.sync_cart_from_q(self.active_q)
            else:
                done_cb = self.on_traj_complete
                self.exec_mode = None
                self.on_traj_complete = None
                msg = f"FINISHED: {self.traj_label}"
                if self.record_active and len(self.log_t) > 5:
                    met = self._compute_metrics()
                    self._append_summary(met)
                    msg += (f" | [{self.ctrl_name}] RMS {met['rms_err_mm']:.2f} mm, max {met['max_err_mm']:.2f} mm, "
                            f"tilt {met['max_tilt_deg']:.2f} deg, clearance {met['clearance_to_box_mm']:.1f} mm")
                self.lbl_m4_status.config(text=msg)
                if done_cb is not None:
                    done_cb()

        self.lbl_pos.config(text=f"X: {pos_actual[0]:+7.4f} | Y: {pos_actual[1]:+7.4f} | Z: {pos_actual[2]:+7.4f}")
        v_tip = compute_torch_tip_jacobian(q_actual) @ qdot_actual
        self.lbl_vel.config(text=f"Vx: {v_tip[0]:+7.4f} | Vy: {v_tip[1]:+7.4f} | Vz: {v_tip[2]:+7.4f}")
        self.lbl_tilt.config(text=f"vs ready: {torch_tilt_deg(q_actual):5.2f} | vs vertical: {torch_tilt_from_vertical_deg(q_actual):5.2f}")

        self.root.after(int(self.traj_dt * 1000), self.update_loop)


def main():
    rclpy.init()
    node = UR5eM4Node()
    root = tk.Tk()
    app = Milestone4WeldingGUI(root, node)

    def on_closing():
        root.destroy()
        node.destroy_node()
        rclpy.shutdown()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()


if __name__ == '__main__':
    main()