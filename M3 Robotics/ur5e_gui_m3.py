#!/usr/bin/env python3
"""
MCTR911 - Milestone 3 Unified Industrial UR5e GUI
Features:
  - Real-Time Kinematics & Differential Telemetry Monitor
  - Cartesian Position (IK) & Joint Position (FK) Sliders
  - Cartesian Velocity Stepping via Pseudo-Inverse (J^+)
  - Simulated Welding Torch Arc Trigger (/welding_active)
  - Weld bead clear (/weld_clear) and conveyor on/off (/conveyor_active)
  - Automatic weld test button (/auto_weld, status on /auto_weld_status)
"""

import time
import tkinter as tk
from tkinter import ttk
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, Bool, Empty, String
from sensor_msgs.msg import JointState

from kinematics_derivation_ur5e import (
    forward_kinematics,
    get_end_effector_position,
    compute_jacobian,
    forward_velocity_kinematics,
    forward_acceleration_kinematics,
    inverse_velocity_kinematics,
    inverse_kinematics_position
)

class UR5eM3Node(Node):
    def __init__(self):
        super().__init__('ur5e_m3_unified_gui_node')
        self.pub_joints = self.create_publisher(Float64MultiArray, '/joint_commands', 10)
        self.pub_weld = self.create_publisher(Bool, '/welding_active', 10)
        self.pub_clear = self.create_publisher(Empty, '/weld_clear', 10)
        self.pub_conv = self.create_publisher(Bool, '/conveyor_active', 10)
        self.pub_auto = self.create_publisher(Empty, '/auto_weld', 10)
        self.sub_states = self.create_subscription(JointState, '/joint_states', self.joint_state_cb, 10)
        self.sub_auto = self.create_subscription(String, '/auto_weld_status', self.auto_status_cb, 10)
        self.auto_status = "IDLE"
        
        self.current_q = [0.0, -np.pi/2, 0.0, -np.pi/2, 0.0, 0.0]
        self.current_qdot = np.zeros(6)
        self.prev_qdot = np.zeros(6)
        self.current_qddot = np.zeros(6)
        self.last_time = time.time()

    def joint_state_cb(self, msg):
        now = time.time()
        dt = max(now - self.last_time, 1e-3)
        self.last_time = now

        if len(msg.position) >= 6:
            self.current_q = list(msg.position[:6])
            
        if len(msg.velocity) >= 6:
            new_qdot = np.array(msg.velocity[:6])
            self.current_qddot = (new_qdot - self.prev_qdot) / dt
            self.prev_qdot = new_qdot.copy()
            self.current_qdot = new_qdot

    def auto_status_cb(self, msg):
        self.auto_status = msg.data

    @property
    def auto_running(self):
        return not self.auto_status.startswith(("IDLE", "DONE", "ERROR"))

    def trigger_auto_weld(self):
        self.pub_auto.publish(Empty())

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


class Milestone3UnifiedGUI:
    def __init__(self, root, node):
        self.root = root
        self.node = node
        self.root.title("UR5e M3 - Kinematics, Velocity & Welding Monitor")
        self.root.geometry("660x960")

        self.updating_from_code = True
        self.active_q = list(self.node.current_q)
        self.welding_active = True

        # =============================================================
        # Section 1: LIVE KINEMATIC & DIFFERENTIAL TELEMETRY
        # =============================================================
        mon_frame = ttk.LabelFrame(root, text="Kinematic Tool Estimation (Differential Telemetry)", padding=8)
        mon_frame.pack(fill="x", padx=10, pady=4)

        grid_frame = ttk.Frame(mon_frame)
        grid_frame.pack(fill="x", pady=2)

        ttk.Label(grid_frame, text="End-Effector Pose (m):", font=("Helvetica", 9, "bold")).grid(row=0, column=0, sticky="w", pady=1)
        self.lbl_pos = ttk.Label(grid_frame, text="X: +0.0000 | Y: -0.2329 | Z: +1.0794", font=("Courier", 9, "bold"), foreground="#0066cc")
        self.lbl_pos.grid(row=0, column=1, sticky="w", padx=5)

        ttk.Label(grid_frame, text="Linear Velocity (m/s):", font=("Helvetica", 9, "bold")).grid(row=1, column=0, sticky="w", pady=1)
        self.lbl_vel = ttk.Label(grid_frame, text="Vx: +0.0000 | Vy: +0.0000 | Vz: +0.0000", font=("Courier", 9, "bold"), foreground="#009933")
        self.lbl_vel.grid(row=1, column=1, sticky="w", padx=5)

        ttk.Label(grid_frame, text="Angular Velocity (rad/s):", font=("Helvetica", 9, "bold")).grid(row=2, column=0, sticky="w", pady=1)
        self.lbl_ang_vel = ttk.Label(grid_frame, text="Wx: +0.0000 | Wy: +0.0000 | Wz: +0.0000", font=("Courier", 9), foreground="#227722")
        self.lbl_ang_vel.grid(row=2, column=1, sticky="w", padx=5)

        ttk.Label(grid_frame, text="Linear Accel (m/s²):", font=("Helvetica", 9, "bold")).grid(row=3, column=0, sticky="w", pady=1)
        self.lbl_acc = ttk.Label(grid_frame, text="Ax: +0.0000 | Ay: +0.0000 | Az: +0.0000", font=("Courier", 9, "bold"), foreground="#cc3300")
        self.lbl_acc.grid(row=3, column=1, sticky="w", padx=5)

        ttk.Separator(mon_frame, orient="horizontal").pack(fill="x", pady=4)

        ttk.Label(mon_frame, text="Live Joint Velocities [q_dot 1..6] (rad/s):", font=("Helvetica", 9, "bold")).pack(anchor="w")
        self.lbl_qdot = ttk.Label(mon_frame, text="[+0.000, +0.000, +0.000, +0.000, +0.000, +0.000]", font=("Courier", 9), foreground="#551a8b")
        self.lbl_qdot.pack(anchor="w", padx=5)

        # =============================================================
        # Section 2: Cartesian Position Control (IK)
        # =============================================================
        cart_frame = ttk.LabelFrame(root, text="Cartesian Position Control (Inverse Kinematics)", padding=8)
        cart_frame.pack(fill="x", padx=10, pady=4)

        self.cart_sliders = {}
        self.cart_labels = {}
        axes = [
            ("X (m)", -0.85, 0.85, 0.0),
            ("Y (m)", -0.85, 0.85, -0.45),
            ("Z (m)",  0.05, 1.10, 0.45)
        ]

        for axis_name, min_v, max_v, init_v in axes:
            row = ttk.Frame(cart_frame)
            row.pack(fill="x", pady=2)
            lbl = ttk.Label(row, text=f"{axis_name}: {init_v:6.3f}", width=16)
            lbl.pack(side="left")
            self.cart_labels[axis_name] = lbl

            slider = ttk.Scale(row, from_=min_v, to=max_v, orient="horizontal",
                               command=lambda val, a=axis_name: self.on_cart_slider(a, val))
            self.cart_sliders[axis_name] = slider
            slider.set(init_v)
            slider.pack(side="right", fill="x", expand=True)

        # =============================================================
        # Section 3: Joint Position Control (Direct / FK)
        # =============================================================
        joint_frame = ttk.LabelFrame(root, text="Joint Position Control (Direct / FK)", padding=8)
        joint_frame.pack(fill="x", padx=10, pady=4)

        self.joint_sliders = []
        self.joint_labels = []
        joint_names = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]

        for i, name in enumerate(joint_names):
            row = ttk.Frame(joint_frame)
            row.pack(fill="x", pady=2)
            lbl = ttk.Label(row, text=f"{name[:12]}: 0.00", width=18)
            lbl.pack(side="left")
            self.joint_labels.append(lbl)

            slider = ttk.Scale(row, from_=-3.1416, to=3.1416, orient="horizontal",
                               command=lambda val, idx=i: self.on_joint_slider(idx, val))
            self.joint_sliders.append(slider)
            slider.set(self.active_q[i])
            slider.pack(side="right", fill="x", expand=True)

        btn_frame = ttk.Frame(root, padding=4)
        btn_frame.pack(fill="x", padx=10, pady=2)
        ttk.Button(btn_frame, text="Reset Upright", command=self.preset_upright).pack(side="left", expand=True, fill="x", padx=2)

        # =============================================================
        # Section 4: Milestone 3 Cartesian Velocity Stepping
        # =============================================================
        vel_frame = ttk.LabelFrame(root, text="Milestone 3: Cartesian Velocity Input (dx/dt, dy/dt, dz/dt)", padding=8)
        vel_frame.pack(fill="x", padx=10, pady=4)

        self.vx_slider = self._make_vel_slider(vel_frame, "Vx (m/s)", -0.25, 0.25, 0.0)
        self.vy_slider = self._make_vel_slider(vel_frame, "Vy (m/s)", -0.25, 0.25, 0.0)
        self.vz_slider = self._make_vel_slider(vel_frame, "Vz (m/s)", -0.25, 0.25, 0.0)

        jog_row = ttk.Frame(vel_frame)
        jog_row.pack(fill="x", pady=4)
        ttk.Button(jog_row, text="Step Velocity (0.1s)", command=self.apply_velocity_step).pack(side="left", expand=True, fill="x", padx=2)
        ttk.Button(jog_row, text="Zero Velocity", command=self.stop_velocity).pack(side="right", expand=True, fill="x", padx=2)

        # =============================================================
        # Section 5: Welding Process & Workcell Module Controls
        # =============================================================
        env_frame = ttk.LabelFrame(root, text="Welding Simulation Process & Safety Modules", padding=8)
        env_frame.pack(fill="x", padx=10, pady=4)

        weld_row = ttk.Frame(env_frame)
        weld_row.pack(fill="x", pady=4)
        self.weld_btn = ttk.Button(weld_row, text="⚡ Welding Torch: ACTIVE (ON)", command=self.toggle_welding)
        self.weld_btn.pack(side="left", expand=True, fill="x", padx=2)
        ttk.Button(weld_row, text="Clear Weld Bead Line", command=self.clear_weld).pack(side="right", expand=True, fill="x", padx=2)

        auto_row = ttk.Frame(env_frame)
        auto_row.pack(fill="x", pady=4)
        self.auto_btn = ttk.Button(auto_row, text="▶ Run Auto Weld Test", command=self.node.trigger_auto_weld)
        self.auto_btn.pack(side="left", expand=True, fill="x", padx=2)
        self.lbl_auto = ttk.Label(auto_row, text="Auto weld: IDLE", font=("Courier", 9, "bold"), foreground="#b36b00", width=36)
        self.lbl_auto.pack(side="right", padx=4)
        self.was_auto = False

        self.conveyor_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(env_frame, text="Infeed Conveyor Active", variable=self.conveyor_var,
                        command=lambda: self.node.set_conveyor(self.conveyor_var.get())).pack(anchor="w")

        self.sync_cart_from_q(self.active_q)
        self.updating_from_code = False
        self.node.set_welding(self.welding_active)
        self.node.set_conveyor(self.conveyor_var.get())
        self.update_loop()

    def _make_vel_slider(self, parent, text, f, t, init):
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=2)
        lbl = ttk.Label(row, text=f"{text}: {init:6.3f}", width=16)
        lbl.pack(side="left")
        slider = ttk.Scale(row, from_=f, to=t, orient="horizontal",
                           command=lambda val, l=lbl, t_name=text: l.config(text=f"{t_name}: {float(val):6.3f}"))
        slider.set(init)
        slider.pack(side="right", fill="x", expand=True)
        return slider

    def toggle_welding(self):
        self.welding_active = not self.welding_active
        if self.welding_active:
            self.weld_btn.config(text="⚡ Welding Torch: ACTIVE (ON)")
        else:
            self.weld_btn.config(text="⚪ Welding Torch: DISABLED (OFF)")
        self.node.set_welding(self.welding_active)

    def clear_weld(self):
        self.node.clear_weld_bead()

    def on_cart_slider(self, axis_name, val):
        if self.updating_from_code:
            return
        if not all(k in self.cart_sliders for k in ["X (m)", "Y (m)", "Z (m)"]):
            return

        self.cart_labels[axis_name].config(text=f"{axis_name}: {float(val):6.3f}")
        target_xyz = np.array([
            self.cart_sliders["X (m)"].get(),
            self.cart_sliders["Y (m)"].get(),
            self.cart_sliders["Z (m)"].get()
        ])
        
        new_q = inverse_kinematics_position(target_xyz, self.active_q)
        self.active_q = list(new_q)
        self.node.send_joints(self.active_q)

        self.updating_from_code = True
        for i, v in enumerate(self.active_q):
            self.joint_sliders[i].set(v)
            self.joint_labels[i].config(text=f"{self.joint_labels[i].cget('text').split(':')[0]}: {v:6.2f}")
        self.updating_from_code = False

    def on_joint_slider(self, idx, val):
        if self.updating_from_code:
            return
        val = float(val)
        self.active_q[idx] = val
        self.joint_labels[idx].config(text=f"{self.joint_labels[idx].cget('text').split(':')[0]}: {val:6.2f}")
        self.node.send_joints(self.active_q)
        self.sync_cart_from_q(self.active_q)

    def apply_velocity_step(self):
        """Integrates Cartesian velocity into joint displacement using over-actuated pseudo-inverse."""
        v_target = np.array([self.vx_slider.get(), self.vy_slider.get(), self.vz_slider.get()])
        dt = 0.1
        q_dot = inverse_velocity_kinematics(np.array(self.active_q), v_target, position_only=True)
        self.active_q = list(np.array(self.active_q) + q_dot * dt)
        self.node.send_joints(self.active_q)

        self.updating_from_code = True
        for i, v in enumerate(self.active_q):
            self.joint_sliders[i].set(v)
            self.joint_labels[i].config(text=f"{self.joint_labels[i].cget('text').split(':')[0]}: {v:6.2f}")
        self.updating_from_code = False
        self.sync_cart_from_q(self.active_q)

    def stop_velocity(self):
        self.vx_slider.set(0.0)
        self.vy_slider.set(0.0)
        self.vz_slider.set(0.0)

    def sync_cart_from_q(self, q):
        pos = get_end_effector_position(q)
        self.updating_from_code = True
        for (axis_name, val) in zip(["X (m)", "Y (m)", "Z (m)"], pos):
            if axis_name in self.cart_sliders:
                self.cart_sliders[axis_name].set(val)
                self.cart_labels[axis_name].config(text=f"{axis_name}: {val:6.3f}")
        self.updating_from_code = False

    def preset_upright(self):
        self.set_joints([0.0, -np.pi/2, 0.0, -np.pi/2, 0.0, 0.0])

    def set_joints(self, q):
        self.active_q = list(q)
        self.updating_from_code = True
        for i, val in enumerate(self.active_q):
            self.joint_sliders[i].set(val)
            self.joint_labels[i].config(text=f"{self.joint_labels[i].cget('text').split(':')[0]}: {val:6.2f}")
        self.updating_from_code = False
        self.sync_cart_from_q(self.active_q)
        self.node.send_joints(self.active_q)

    def refresh_from_state(self):
        """Copy the real arm pose (from /joint_states) into the sliders."""
        self.active_q = list(self.node.current_q)
        self.updating_from_code = True
        for i, v in enumerate(self.active_q):
            self.joint_sliders[i].set(v)
            self.joint_labels[i].config(text=f"{self.joint_labels[i].cget('text').split(':')[0]}: {v:6.2f}")
        self.updating_from_code = False
        self.sync_cart_from_q(self.active_q)

    def update_loop(self):
        rclpy.spin_once(self.node, timeout_sec=0.01)

        # automatic weld test: follow the arm while the sim drives it
        running = self.node.auto_running
        self.lbl_auto.config(text=f"Auto weld: {self.node.auto_status}")
        self.auto_btn.config(text="■ Abort Auto Weld" if running else "▶ Run Auto Weld Test")
        if running or self.was_auto:
            self.refresh_from_state()
        self.was_auto = running

        q = np.array(self.active_q)
        qdot = self.node.current_qdot
        qddot = self.node.current_qddot

        pos = get_end_effector_position(q)
        self.lbl_pos.config(text=f"X: {pos[0]:+7.4f} | Y: {pos[1]:+7.4f} | Z: {pos[2]:+7.4f}")

        X_dot = forward_velocity_kinematics(q, qdot)
        self.lbl_vel.config(text=f"Vx: {X_dot[0]:+7.4f} | Vy: {X_dot[1]:+7.4f} | Vz: {X_dot[2]:+7.4f}")
        self.lbl_ang_vel.config(text=f"Wx: {X_dot[3]:+7.4f} | Wy: {X_dot[4]:+7.4f} | Wz: {X_dot[5]:+7.4f}")

        X_ddot = forward_acceleration_kinematics(q, qdot, qddot)
        self.lbl_acc.config(text=f"Ax: {X_ddot[0]:+7.4f} | Ay: {X_ddot[1]:+7.4f} | Az: {X_ddot[2]:+7.4f}")

        qdot_str = ", ".join([f"{val:+6.3f}" for val in qdot])
        self.lbl_qdot.config(text=f"[{qdot_str}]")

        self.root.after(30, self.update_loop)

def main():
    rclpy.init()
    node = UR5eM3Node()
    root = tk.Tk()
    app = Milestone3UnifiedGUI(root, node)

    def on_closing():
        root.destroy()
        node.destroy_node()
        rclpy.shutdown()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == '__main__':
    main()
