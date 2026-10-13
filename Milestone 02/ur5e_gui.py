#!/usr/bin/env python3
import tkinter as tk
from tkinter import ttk
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState

# UR5e Standard DH Parameters
D = [0.1625, 0.0, 0.0, 0.1333, 0.0997, 0.0996]
A = [0.0, -0.425, -0.3922, 0.0, 0.0, 0.0]
ALPHA = [np.pi / 2, 0.0, 0.0, np.pi / 2, -np.pi / 2, 0.0]

def dh_matrix(theta, d, a, alpha):
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0.0, sa,       ca,      d],
        [0.0, 0.0,      0.0,     1.0]
    ])

def forward_kinematics(q):
    """Computes end-effector position (x, y, z) from 6 joint angles."""
    T = np.identity(4)
    for i in range(6):
        T = np.matmul(T, dh_matrix(q[i], D[i], A[i], ALPHA[i]))
    return T[:3, 3]

def inverse_kinematics_xyz(target_pos, q_seed, max_steps=40, tol=1e-3, damping=0.01):
    """Numerical IK solving for Cartesian XYZ target position."""
    q = np.array(q_seed, dtype=float)
    for _ in range(max_steps):
        curr_pos = forward_kinematics(q)
        err = target_pos - curr_pos
        if np.linalg.norm(err) < tol:
            break
        
        # Numerical Jacobian for position
        J = np.zeros((3, 6))
        eps = 1e-6
        for i in range(6):
            q_perturbed = q.copy()
            q_perturbed[i] += eps
            pos_perturbed = forward_kinematics(q_perturbed)
            J[:, i] = (pos_perturbed - curr_pos) / eps
            
        # Damped Least Squares inverse
        inv_J = J.T @ np.linalg.inv(J @ J.T + damping**2 * np.eye(3))
        dq = inv_J @ err
        
        # Clamp step size to prevent sudden jumps
        max_dq = 0.2
        if np.linalg.norm(dq) > max_dq:
            dq = dq * (max_dq / np.linalg.norm(dq))
            
        q += dq
        # Normalize within [-pi, pi]
        q = (q + np.pi) % (2 * np.pi) - np.pi
    return q


class UR5eNode(Node):
    def __init__(self):
        super().__init__('ur5e_ik_gui_node')
        self.pub = self.create_publisher(Float64MultiArray, '/joint_commands', 10)
        self.sub = self.create_subscription(JointState, '/joint_states', self.cb, 10)
        self.current_q = [0.0, -1.5708, 0.0, -1.5708, 0.0, 0.0]

    def cb(self, msg):
        if len(msg.position) >= 6:
            self.current_q = list(msg.position[:6])

    def send_joints(self, q):
        msg = Float64MultiArray()
        msg.data = [float(x) for x in q]
        self.pub.publish(msg)


class IKApp:
    def __init__(self, root, node):
        self.root = root
        self.node = node
        self.root.title("UR5e IK Cartesian & Joint Controller")
        self.root.geometry("520x650")

        self.updating_from_code = False
        self.active_q = list(self.node.current_q)

        # Frame 1: Cartesian Sliders (Inverse Kinematics)
        cart_frame = ttk.LabelFrame(root, text="Cartesian Position Control (Inverse Kinematics)", padding=10)
        cart_frame.pack(fill="x", padx=10, pady=5)

        self.cart_sliders = {}
        self.cart_labels = {}
        axes = [
            ("X (m)", -0.85, 0.85, 0.0),
            ("Y (m)", -0.85, 0.85, -0.45),
            ("Z (m)",  0.05, 1.10, 0.45)
        ]

        for axis_name, min_v, max_v, init_v in axes:
            row = ttk.Frame(cart_frame)
            row.pack(fill="x", pady=4)
            lbl = ttk.Label(row, text=f"{axis_name}: {init_v:6.3f}", width=16)
            lbl.pack(side="left")
            self.cart_labels[axis_name] = lbl

            slider = ttk.Scale(row, from_=min_v, to=max_v, orient="horizontal",
                               command=lambda val, a=axis_name: self.on_cart_slider(a, val))
            slider.set(init_v)
            slider.pack(side="right", fill="x", expand=True)
            self.cart_sliders[axis_name] = slider

        # Frame 2: Joint Sliders
        joint_frame = ttk.LabelFrame(root, text="Joint Position Control (Direct / FK)", padding=10)
        joint_frame.pack(fill="both", expand=True, padx=10, pady=5)

        self.joint_sliders = []
        self.joint_labels = []
        joint_names = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]

        for i, name in enumerate(joint_names):
            row = ttk.Frame(joint_frame)
            row.pack(fill="x", pady=3)
            lbl = ttk.Label(row, text=f"{name[:12]}: 0.00", width=18)
            lbl.pack(side="left")
            self.joint_labels.append(lbl)

            slider = ttk.Scale(row, from_=-3.1416, to=3.1416, orient="horizontal",
                               command=lambda val, idx=i: self.on_joint_slider(idx, val))
            slider.set(self.active_q[i])
            slider.pack(side="right", fill="x", expand=True)
            self.joint_sliders.append(slider)

        # Presets
        btn_frame = ttk.Frame(root, padding=5)
        btn_frame.pack(fill="x", padx=10, pady=5)
        ttk.Button(btn_frame, text="Reset Upright", command=self.preset_upright).pack(side="left", expand=True, fill="x", padx=2)
        ttk.Button(btn_frame, text="Home / Ready", command=self.preset_ready).pack(side="left", expand=True, fill="x", padx=2)

        # Sync UI to initial forward kinematics
        self.sync_cart_from_q(self.active_q)
        self.update_loop()

    def on_cart_slider(self, axis_name, val):
        if self.updating_from_code:
            return
        self.cart_labels[axis_name].config(text=f"{axis_name}: {float(val):6.3f}")
        target_xyz = np.array([
            self.cart_sliders["X (m)"].get(),
            self.cart_sliders["Y (m)"].get(),
            self.cart_sliders["Z (m)"].get()
        ])
        
        # Solve IK using current joint angles as seed
        new_q = inverse_kinematics_xyz(target_xyz, self.active_q)
        self.active_q = list(new_q)
        self.node.send_joints(self.active_q)

        # Update joint sliders visually
        self.updating_from_code = True
        for i, val in enumerate(self.active_q):
            self.joint_sliders[i].set(val)
            self.joint_labels[i].config(text=f"{self.joint_labels[i].cget('text').split(':')[0]}: {val:6.2f}")
        self.updating_from_code = False

    def on_joint_slider(self, idx, val):
        if self.updating_from_code:
            return
        val = float(val)
        self.active_q[idx] = val
        self.joint_labels[idx].config(text=f"{self.joint_labels[idx].cget('text').split(':')[0]}: {val:6.2f}")
        self.node.send_joints(self.active_q)
        self.sync_cart_from_q(self.active_q)

    def sync_cart_from_q(self, q):
        pos = forward_kinematics(q)
        self.updating_from_code = True
        for (axis_name, val) in zip(["X (m)", "Y (m)", "Z (m)"], pos):
            self.cart_sliders[axis_name].set(val)
            self.cart_labels[axis_name].config(text=f"{axis_name}: {val:6.3f}")
        self.updating_from_code = False

    def preset_upright(self):
        self.set_joints([0.0, -1.5708, 0.0, -1.5708, 0.0, 0.0])

    def preset_ready(self):
        self.set_joints([0.0, -1.0, 1.2, -0.5, 1.57, 0.0])

    def set_joints(self, q):
        self.active_q = list(q)
        self.updating_from_code = True
        for i, val in enumerate(self.active_q):
            self.joint_sliders[i].set(val)
            self.joint_labels[i].config(text=f"{self.joint_labels[i].cget('text').split(':')[0]}: {val:6.2f}")
        self.updating_from_code = False
        self.sync_cart_from_q(self.active_q)
        self.node.send_joints(self.active_q)

    def update_loop(self):
        rclpy.spin_once(self.node, timeout_sec=0.01)
        self.root.after(20, self.update_loop)


def main():
    rclpy.init()
    node = UR5eNode()
    root = tk.Tk()
    app = IKApp(root, node)

    def on_closing():
        root.destroy()
        node.destroy_node()
        rclpy.shutdown()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()