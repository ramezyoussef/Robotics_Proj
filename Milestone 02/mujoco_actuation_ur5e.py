#!/usr/bin/env python3
import time
import numpy as np
import mujoco
import mujoco.viewer
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

class UR5eActuationController(Node):
    def __init__(self, model, data):
        super().__init__('ur5e_actuation_controller')
        self.model = model
        self.data = data
        
        # ROS 2 Publisher: streams live joint telemetry
        self.joint_pub = self.create_publisher(JointState, '/joint_states', 10)
        
        # ROS 2 Subscriber: listens for joint commands (constants or trajectories)
        self.cmd_sub = self.create_subscription(
            Float64MultiArray, '/joint_commands', self.command_callback, 10
        )
        
        self.joint_names = [
            'shoulder_pan_joint',
            'shoulder_lift_joint',
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint'
        ]
        
        # Modes: 'ros_cmd', 'signal_builder', or 'constant'
        self.mode = 'ros_cmd'
        self.sim_time = 0.0
        
        # Cached target positions initialized to current positions
        self.target_qpos = np.copy(self.data.qpos[:6])
        
        # Control timer running at 100 Hz (0.01s)
        self.timer = self.create_timer(0.01, self.control_loop)
        self.get_logger().info(f'UR5e Actuation Controller initialized in [{self.mode}] mode!')

    def command_callback(self, msg):
        """Processes constant or external trajectory vectors received via ROS 2"""
        if len(msg.data) == 6:
            self.target_qpos = np.array(msg.data, dtype=float)
            self.get_logger().info(f'Received new joint target: {np.round(self.target_qpos, 3)}')

    def generate_signal(self, t):
        """Signal Builder: Smooth multi-joint harmonic wave functions"""
        cmd = np.zeros(6)
        cmd[0] = 0.6 * np.sin(0.5 * t)                   # Shoulder Pan
        cmd[1] = -np.pi/4 + 0.3 * np.sin(0.4 * t)        # Shoulder Lift
        cmd[2] = np.pi/3 + 0.4 * np.cos(0.4 * t)         # Elbow
        cmd[3] = -np.pi/4 + 0.2 * np.sin(0.6 * t)        # Wrist 1
        cmd[4] = 0.3 * np.sin(0.8 * t)                   # Wrist 2
        cmd[5] = 0.5 * np.sin(1.0 * t)                   # Wrist 3
        return cmd

    def control_loop(self):
        self.sim_time += 0.01
        
        if self.mode == 'signal_builder':
            cmd = self.generate_signal(self.sim_time)
        elif self.mode == 'constant':
            # Predefined static step posture
            cmd = np.array([0.0, -np.pi/3, np.pi/4, -np.pi/6, np.pi/2, 0.0])
        else: # 'ros_cmd'
            cmd = self.target_qpos

        # Apply target to actuators (ctrl) or kinematic joint state (qpos)
        for i in range(6):
            if self.model.nu >= 6:
                self.data.ctrl[i] = cmd[i]
            else:
                self.data.qpos[i] = cmd[i]

        # Publish live state back to ROS 2
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = [float(self.data.qpos[i]) for i in range(6)]
        msg.velocity = [float(self.data.qvel[i]) for i in range(6)]
        self.joint_pub.publish(msg)

def main():
    rclpy.init()
    
    # Path to UR5e scene XML inside project
    model_path = "mujoco_menagerie/universal_robots_ur5e/scene.xml"
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    
    node = UR5eActuationController(model, data)
    
    # Launch viewer and advance physics synchronized with the ROS 2 loop
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and rclpy.ok():
            step_start = time.time()
            
            mujoco.mj_step(model, data)
            rclpy.spin_once(node, timeout_sec=0.0)
            viewer.sync()
            
            dt = model.opt.timestep - (time.time() - step_start)
            if dt > 0:
                time.sleep(dt)
                
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()