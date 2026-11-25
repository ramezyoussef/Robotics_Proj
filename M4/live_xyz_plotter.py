#!/usr/bin/env python3
"""
MCTR911 - Milestone 4: Live X, Y, Z Tracking Monitor (Standalone Window)
Runs alongside your existing `ur5e_mujoco_welding.py` and `milestone4_welding_gui.py`
without modifying any existing files!

How it works:
- Subscribes to `/joint_states` (actual robot joints) and `/joint_commands` (commanded target joints).
- Computes forward kinematics to get Actual Torch-Tip (X, Y, Z) and Desired Target (X, Y, Z).
- Plots three live graphs (X, Y, Z vs. Time) with the desired target as a dashed line
  and the actual torch tip approaching/tracking it as a solid line.
"""

import time
import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState

from milestone4_trajectories_and_control import (
    READY_Q,
    get_torch_tip_position,
)


class LiveXYZMonitorNode(Node):
    def __init__(self):
        super().__init__('ur5e_live_xyz_monitor_node')
        self.sub_states = self.create_subscription(
            JointState, '/joint_states', self.joint_state_cb, 10
        )
        self.sub_cmds = self.create_subscription(
            Float64MultiArray, '/joint_commands', self.joint_cmd_cb, 10
        )

        self.actual_q = READY_Q.copy()
        self.desired_q = READY_Q.copy()
        self.has_cmd = False

    def joint_state_cb(self, msg):
        if len(msg.position) >= 6:
            self.actual_q = np.array(msg.position[:6], dtype=float)
            if not self.has_cmd:
                self.desired_q = self.actual_q.copy()

    def joint_cmd_cb(self, msg):
        if len(msg.data) >= 6:
            self.desired_q = np.array(msg.data[:6], dtype=float)
            self.has_cmd = True


class LiveXYZPlotterGUI:
    def __init__(self, root, node):
        self.root = root
        self.node = node
        self.root.title("UR5e M4 - Live X, Y, Z Approach & Tracking Monitor")
        self.root.geometry("820x860")

        self.t0 = time.monotonic()
        self.paused = False
        self.window_sec = tk.DoubleVar(value=15.0)

        self.t_hist = []
        self.xd_hist = []
        self.xa_hist = []

        self._build_ui()
        self.update_loop()

    def _build_ui(self):
        # Top status & controls bar
        top_bar = ttk.Frame(self.root, padding=8)
        top_bar.pack(fill="x")

        self.lbl_status = ttk.Label(
            top_bar,
            text="Actual: [0.000, 0.000, 0.000] m  |  Desired: [0.000, 0.000, 0.000] m  |  Error: 0.0 mm",
            font=("Courier", 10, "bold"),
            foreground="#0f4c81",
        )
        self.lbl_status.pack(side="left", padx=4)

        btn_bar = ttk.Frame(self.root, padding=(8, 0, 8, 6))
        btn_bar.pack(fill="x")

        ttk.Button(btn_bar, text="Clear / Reset Graphs", command=self.clear_plot).pack(side="left", padx=3)
        self.pause_btn = ttk.Button(btn_bar, text="Pause Plot", command=self.toggle_pause)
        self.pause_btn.pack(side="left", padx=3)

        ttk.Label(btn_bar, text="Time Window (s):").pack(side="left", padx=(14, 4))
        ttk.Combobox(
            btn_bar,
            textvariable=self.window_sec,
            values=[5.0, 10.0, 15.0, 30.0, 60.0],
            width=6,
            state="readonly",
        ).pack(side="left")

        ttk.Button(btn_bar, text="Save Snapshot (.png)", command=self.save_snapshot).pack(side="right", padx=3)

        # Matplotlib 3-subplot canvas (X, Y, Z)
        plot_frame = ttk.LabelFrame(
            self.root,
            text="Live Torch-Tip Position vs. Desired Destination (Dashed = Target, Solid = Actual)",
            padding=6,
        )
        plot_frame.pack(fill="both", expand=True, padx=8, pady=6)

        self.fig, self.axes = plt.subplots(3, 1, figsize=(7.6, 7.2), sharex=True)
        self.fig.subplots_adjust(left=0.12, right=0.96, top=0.94, bottom=0.08, hspace=0.25)

        specs = [
            ("X Position (m)", "#d62728"),
            ("Y Position (m)", "#2ca02c"),
            ("Z Position (m)", "#1f77b4"),
        ]
        self.des_lines = []
        self.act_lines = []
        self.target_hlines = []

        for idx, (ylabel, color) in enumerate(specs):
            ax = self.axes[idx]
            # Horizontal dashed marker showing current instantaneous destination setpoint
            hline = ax.axhline(0.0, linestyle=":", color=color, alpha=0.55, linewidth=1.2)
            # Dashed desired trajectory curve
            d_line, = ax.plot([], [], "--", color=color, linewidth=1.8, label="Desired Destination")
            # Solid actual torch-tip curve
            a_line, = ax.plot([], [], "-", color=color, linewidth=2.2, label="Actual Torch Tip")

            ax.set_ylabel(ylabel, fontsize=10, fontweight="bold")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right", fontsize=8.5)

            self.target_hlines.append(hline)
            self.des_lines.append(d_line)
            self.act_lines.append(a_line)

        self.axes[0].set_title("Real-Time Torch-Tip Approach & Tracking (X, Y, Z)", fontsize=11, fontweight="bold")
        self.axes[2].set_xlabel("Elapsed Time (s)", fontsize=10, fontweight="bold")

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def clear_plot(self):
        self.t0 = time.monotonic()
        self.t_hist.clear()
        self.xd_hist.clear()
        self.xa_hist.clear()
        for i in range(3):
            self.des_lines[i].set_data([], [])
            self.act_lines[i].set_data([], [])
        self.canvas.draw_idle()

    def toggle_pause(self):
        self.paused = not self.paused
        self.pause_btn.config(text="Resume Plot" if self.paused else "Pause Plot")

    def save_snapshot(self):
        fname = f"m4_live_xyz_{int(time.time())}.png"
        self.fig.savefig(fname, dpi=150)
        self.lbl_status.config(text=f"Saved plot snapshot to {fname}")

    def update_loop(self):
        rclpy.spin_once(self.node, timeout_sec=0.005)

        if not self.paused:
            t_now = time.monotonic() - self.t0
            pos_actual = get_torch_tip_position(self.node.actual_q)
            pos_desired = get_torch_tip_position(self.node.desired_q)

            self.t_hist.append(t_now)
            self.xd_hist.append(pos_desired)
            self.xa_hist.append(pos_actual)

            # Trim history to the selected rolling time window
            win = float(self.window_sec.get())
            while len(self.t_hist) > 2 and (t_now - self.t_hist[0]) > win:
                self.t_hist.pop(0)
                self.xd_hist.pop(0)
                self.xa_hist.pop(0)

            t_arr = np.array(self.t_hist)
            xd_arr = np.array(self.xd_hist)
            xa_arr = np.array(self.xa_hist)

            for i in range(3):
                self.des_lines[i].set_data(t_arr, xd_arr[:, i])
                self.act_lines[i].set_data(t_arr, xa_arr[:, i])
                self.target_hlines[i].set_ydata([pos_desired[i], pos_desired[i]])

                ax = self.axes[i]
                ax.relim()
                ax.autoscale_view()

            err_mm = np.linalg.norm(pos_desired - pos_actual) * 1000.0
            self.lbl_status.config(
                text=(
                    f"Actual: [{pos_actual[0]:+.3f}, {pos_actual[1]:+.3f}, {pos_actual[2]:+.3f}] m  |  "
                    f"Desired: [{pos_desired[0]:+.3f}, {pos_desired[1]:+.3f}, {pos_desired[2]:+.3f}] m  |  "
                    f"Error: {err_mm:.1f} mm"
                )
            )
            self.canvas.draw_idle()

        self.root.after(30, self.update_loop)


def main():
    rclpy.init()
    node = LiveXYZMonitorNode()
    root = tk.Tk()
    app = LiveXYZPlotterGUI(root, node)

    def on_closing():
        plt.close('all')
        root.destroy()
        node.destroy_node()
        rclpy.shutdown()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()


if __name__ == '__main__':
    main()
