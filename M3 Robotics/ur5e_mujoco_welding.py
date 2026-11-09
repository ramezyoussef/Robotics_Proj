#!/usr/bin/env python3
"""
UR5e welding cell (MuJoCo + ROS 2)

Replaces the old actuation file. Keeps the same topics:
  sub  /joint_commands   Float64MultiArray   (from your GUI)
  pub  /joint_states     JointState
New:
  sub  /welding_active   Bool    torch on/off (already published by your GUI)
  sub  /weld_clear       Empty   wipe the white bead
  sub  /conveyor_active  Bool    belt on/off
  sub  /auto_weld        Empty   run (or abort) the automatic weld test
  pub  /auto_weld_status String  IDLE / WAITING / MOVING / WELDING / DONE / ERROR

The bead is stored in each box's local frame, so it rides along with the box on the conveyor.
Requires mujoco >= 3.2 (MjSpec is used to bolt the torch onto wrist_3_link).
"""
import os
import time
import numpy as np
import mujoco
import mujoco.viewer
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, Bool, Empty, String

CELL_XML = "mujoco_menagerie/universal_robots_ur5e/welding_cell.xml"

# --- tunables --------------------------------------------------------------
CONVEYOR_SPEED = 0.04     # m/s
CONVEYOR_LIMIT = 0.35     # boxes wrap back after travelling this far (m)
BEAD_RADIUS = 0.004       # m
BEAD_STEP = 0.004         # min spacing between bead points (m)
BEAD_MARGIN = 0.015       # tip within this distance of a box surface -> it welds (m)
MAX_BEAD_PTS = 1800       # viewer's user_scn holds ~2000 geoms
BEAD_RGBA = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
ARC_RGBA = np.array([0.6, 0.8, 1.0, 0.9], dtype=np.float32)


def build_model():
    """Load the cell and attach a welding torch to the UR5e flange."""
    spec = mujoco.MjSpec.from_file(CELL_XML)
    wrist = spec.body("wrist_3_link")
    # same frame as Menagerie's attachment_site; torch points along local +Z
    torch = wrist.add_body(name="torch", pos=[0, 0.1, 0], quat=[-0.70710678, 0.70710678, 0, 0])
    torch.add_geom(name="torch_neck", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                   size=[0.02, 0.04, 0], pos=[0, 0, 0.04], rgba=[0.15, 0.15, 0.17, 1],
                   contype=0, conaffinity=0, mass=0.2)
    torch.add_geom(name="torch_nozzle", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                   size=[0.008, 0.05, 0], pos=[0, 0, 0.13], rgba=[0.85, 0.45, 0.15, 1],
                   contype=0, conaffinity=0, mass=0.05)
    torch.add_site(name="torch_tip", pos=[0, 0, 0.185], size=[0.004, 0, 0])
    model = spec.compile()
    model.body_gravcomp[:] = 1.0     # arm is gravity-compensated -> no sag, accurate tip tracking
    return model


class WeldRecorder:
    """Records bead points in box-local coordinates and draws them each frame."""

    def __init__(self, model, data, box_names=("box_a", "box_b")):
        self.m, self.d = model, data
        self.body_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in box_names]
        self.half = []
        for n in box_names:
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n + "_geom")
            self.half.append(model.geom_size[gid].copy())
        self.strokes = []      # each: {"box": i, "pts": [local xyz, ...]}
        self.cur = None
        self.npts = 0

    def clear(self):
        self.strokes, self.cur, self.npts = [], None, 0

    def _surface_point(self, tip):
        """Return (box_index, local_point_on_surface) if the tip is on/near a box, else None."""
        for i, b in enumerate(self.body_ids):
            R = self.d.xmat[b].reshape(3, 3)
            loc = R.T @ (tip - self.d.xpos[b])
            h = self.half[i]
            if np.all(np.abs(loc) <= h + BEAD_MARGIN):
                if loc[2] > h[2] - 0.02:                  # tip at / just under / just above the top face
                    loc[:2] = np.clip(loc[:2], -h[:2], h[:2])
                    loc[2] = h[2] + 0.002                 # bead sits on top of the surface
                else:
                    pen = h - np.abs(loc)
                    if np.all(pen > 0):                  # tip inside the box -> push to nearest face
                        k = int(np.argmin(pen))
                        loc[k] = np.sign(loc[k]) * h[k]
                    else:
                        loc = np.clip(loc, -h, h)
                return i, loc
        return None

    def update(self, tip, torch_on):
        """Call every physics step. Returns True while an arc is burning on a workpiece."""
        hit = self._surface_point(tip) if torch_on else None
        if hit is None:
            self.cur = None
            return False
        i, loc = hit
        if self.cur is None or self.cur["box"] != i:
            self.cur = {"box": i, "pts": [loc]}
            self.strokes.append(self.cur)
            self.npts += 1
        elif np.linalg.norm(loc - self.cur["pts"][-1]) >= BEAD_STEP:
            self.cur["pts"].append(loc)
            self.npts += 1
        while self.npts > MAX_BEAD_PTS and self.strokes:   # drop oldest
            old = self.strokes[0]
            old["pts"].pop(0)
            self.npts -= 1
            if len(old["pts"]) == 0:
                if old is self.cur:
                    self.cur = None
                self.strokes.pop(0)
        return True

    def clear_box(self, i):
        self.strokes = [s for s in self.strokes if s["box"] != i]
        self.npts = sum(len(s["pts"]) for s in self.strokes)
        if self.cur not in self.strokes:
            self.cur = None

    def draw(self, scn, tip, arcing):
        scn.ngeom = 0
        for s in self.strokes:
            b = self.body_ids[s["box"]]
            R, p = self.d.xmat[b].reshape(3, 3), self.d.xpos[b]
            w = [p + R @ q for q in s["pts"]]
            if len(w) == 1:
                self._sphere(scn, w[0], BEAD_RADIUS, BEAD_RGBA)
            for a, c in zip(w[:-1], w[1:]):
                self._capsule(scn, a, c, BEAD_RADIUS, BEAD_RGBA)
        if arcing:
            self._sphere(scn, tip, 0.009, ARC_RGBA)

    @staticmethod
    def _capsule(scn, a, b, r, rgba):
        if scn.ngeom >= scn.maxgeom:
            return
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.zeros(9), rgba)
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, r, a, b)
        scn.ngeom += 1

    @staticmethod
    def _sphere(scn, pos, r, rgba):
        if scn.ngeom >= scn.maxgeom:
            return
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([r, 0, 0]), pos,
                            np.eye(3).flatten(), rgba)
        scn.ngeom += 1


class AutoWelder:
    """
    Automatic weld test:
      1. wait until the seam between the two boxes reaches the belt centre
      2. stop the belt
      3. move above the seam (from ANY starting pose), lower the torch, weld a straight line
         across the seam (white bead), lift, return to the welding-ready pose
      4. resume the belt
    Uses MuJoCo's own Jacobian IK (tip position + torch pointing down).
    """
    X_START, X_END = 0.38, 0.52        # weld line along X, across the seam (boxes span x 0.37..0.53)
    TOP_Z = 0.47                       # box top surface (world z)
    WELD_Z = TOP_Z + 0.002
    HOVER = 0.10
    WELD_SPEED = 0.05                  # m/s
    JOINT_SPEED = 0.6                  # rad/s for free moves
    KV_OVER_KP = 0.2                   # Menagerie actuators: 400/2000 -> velocity feed-forward
    ARRIVE_TOL = 0.003                 # m
    # Canonical arm configuration (elbow up, facing the table, torch down). The IK is always seeded
    # from this, so the weld pose is the same no matter where the arm starts.
    READY_Q = np.array([2.839, -1.467, 0.711, -0.815, -1.571, 0.023])

    def __init__(self, model, node, tip_id, conv_qadr):
        self.m, self.node, self.tip = model, node, tip_id
        self.conv_qadr = conv_qadr
        self.ik_data = mujoco.MjData(model)
        self.home_q = model.key("home").qpos[:6].copy() if model.nkey > 0 else np.zeros(6)
        self.running = False
        self.weld_on = False
        self.conveyor_cmd = True
        self.segs, self.i, self.t = [], 0, 0.0
        self.status = ""

    # -- helpers -----------------------------------------------------------
    def _set_status(self, text):
        if text != self.status:
            self.status = text
            self.node.publish_status(text)

    def _ik(self, target, seeds):
        d = self.ik_data
        for q0 in seeds:
            q = np.asarray(q0, float).copy()
            for _ in range(300):
                d.qpos[:6] = q
                mujoco.mj_forward(self.m, d)
                R = d.site_xmat[self.tip].reshape(3, 3)
                e = np.r_[target - d.site_xpos[self.tip], 0.5 * np.cross(R[:, 2], [0, 0, -1])]
                if np.linalg.norm(e) < 5e-4:
                    return q
                jp, jr = np.zeros((3, self.m.nv)), np.zeros((3, self.m.nv))
                mujoco.mj_jacSite(self.m, d, jp, jr, self.tip)
                J = np.vstack([jp, 0.5 * jr])[:, :6]
                q = q + 0.5 * J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), e)
        return None

    @staticmethod
    def _unwrap_offset(q, ref):
        """Per-joint multiple of 2*pi that moves q closest to ref while staying inside +-2*pi."""
        off = 2 * np.pi * np.round((ref - q) / (2 * np.pi))
        bad = np.abs(q + off) > 6.2
        off[bad] = 0.0
        return off

    def _free_move(self, name, q_a, q_b, weld=False):
        dur = max(1.0, float(np.max(np.abs(q_b - q_a))) / self.JOINT_SPEED)
        return {"name": name, "qs": np.array([q_a, q_b]), "dur": dur, "weld": weld, "smooth": True}

    # -- control -----------------------------------------------------------
    def start(self, data):
        self.running, self.weld_on, self.conveyor_cmd = True, False, True
        self.state = "WAIT"
        self.q_start = data.qpos[:6].copy()
        self._set_status("WAITING FOR BOX")

    def abort(self, data, text="IDLE (aborted)"):
        self.running, self.weld_on, self.conveyor_cmd = False, False, True
        self.node.target_qpos = data.qpos[:6].copy()
        self._set_status(text)

    def _plan(self, data):
        y = float(data.qpos[self.conv_qadr])          # seam world-Y = conveyor offset
        pts = {
            "pre_s": np.array([self.X_START, y, self.TOP_Z + self.HOVER]),
            "start": np.array([self.X_START, y, self.WELD_Z]),
            "end":   np.array([self.X_END,   y, self.WELD_Z]),
            "pre_e": np.array([self.X_END,   y, self.TOP_Z + self.HOVER]),
        }
        q_now = data.qpos[:6].copy()
        sol = {}
        seed = self.READY_Q
        sol["pre_s"] = self._ik(pts["pre_s"], [self.READY_Q, self.home_q])
        if sol["pre_s"] is None:
            return None
        seed = sol["pre_s"]
        sol["start"] = self._ik(pts["start"], [seed, self.READY_Q])
        if sol["start"] is None:
            return None
        seed = sol["start"]
        n = int(np.ceil((self.X_END - self.X_START) / 0.004)) + 1
        path = []
        for x in np.linspace(self.X_START, self.X_END, n):
            seed = self._ik(np.array([x, y, self.WELD_Z]), [seed])
            if seed is None:
                return None
            path.append(seed)
        sol["end"] = path[-1]
        sol["pre_e"] = self._ik(pts["pre_e"], [sol["end"]])
        if sol["pre_e"] is None:
            return None
        # pick the 2*pi-equivalent angles closest to where the arm is now (shortest way there)
        off = self._unwrap_offset(sol["pre_s"], q_now)
        for k in sol:
            sol[k] = sol[k] + off
        path = [q + off for q in path]
        ready = self.READY_Q + self._unwrap_offset(self.READY_Q, sol["pre_e"])
        weld_dur = (self.X_END - self.X_START) / self.WELD_SPEED
        return [
            self._free_move("MOVING TO WELD START", q_now, sol["pre_s"]),
            self._free_move("LOWERING TORCH", sol["pre_s"], sol["start"]),
            {"name": "SETTLING", "qs": np.array([sol["start"], sol["start"]]), "dur": 0.7, "weld": False, "smooth": False},
            {"name": "WELDING", "qs": np.array(path), "dur": weld_dur, "weld": True, "smooth": False},
            self._free_move("LIFTING TORCH", sol["end"], sol["pre_e"]),
            self._free_move("RETURNING", sol["pre_e"], ready),
        ]

    @staticmethod
    def _sample(seg, t):
        s = min(max(t / seg["dur"], 0.0), 1.0)
        if seg["smooth"]:
            s = s * s * (3 - 2 * s)
        qs = seg["qs"]
        f = s * (len(qs) - 1)
        k = min(int(f), len(qs) - 2)
        return qs[k] + (f - k) * (qs[k + 1] - qs[k])

    def update(self, data, dt):
        if not self.running:
            return
        if self.state == "WAIT":
            self.weld_on, self.conveyor_cmd = False, True
            if abs(data.qpos[self.conv_qadr]) < self.ARRIVE_TOL:
                self.conveyor_cmd = False
                data.qvel[self.conv_qadr] = 0.0
                segs = self._plan(data)
                if segs is None:
                    self.abort(data, "ERROR: weld line unreachable")
                    return
                self.segs, self.i, self.t, self.state = segs, 0, 0.0, "RUN"
            return

        seg = self.segs[self.i]
        self.conveyor_cmd = False
        self.weld_on = seg["weld"]
        self._set_status(seg["name"])
        self.t += dt
        q_ref = self._sample(seg, self.t)
        q_ahead = self._sample(seg, self.t + 0.01)
        self.node.target_qpos = q_ref + self.KV_OVER_KP * (q_ahead - q_ref) / 0.01
        if self.t >= seg["dur"]:
            self.i, self.t = self.i + 1, 0.0
            if self.i >= len(self.segs):
                self.running, self.weld_on, self.conveyor_cmd = False, False, True
                self._set_status("DONE")


class UR5eActuationController(Node):
    def __init__(self, model, data):
        super().__init__('ur5e_actuation_controller')
        self.model, self.data = model, data

        self.joint_pub = self.create_publisher(JointState, '/joint_states', 10)
        self.create_subscription(Float64MultiArray, '/joint_commands', self.command_callback, 10)
        self.create_subscription(Bool, '/welding_active', lambda m: setattr(self, 'welding_active', m.data), 10)
        self.create_subscription(Bool, '/conveyor_active', lambda m: setattr(self, 'conveyor_active', m.data), 10)
        self.create_subscription(Empty, '/weld_clear', lambda m: setattr(self, 'clear_requested', True), 10)
        self.create_subscription(Empty, '/auto_weld', lambda m: setattr(self, 'auto_toggle', True), 10)
        self.status_pub = self.create_publisher(String, '/auto_weld_status', 10)
        self.auto = None
        self.auto_toggle = False

        self.joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
                            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        self.welding_active = True      # matches GUI default
        self.conveyor_active = True     # matches GUI default
        self.clear_requested = False
        self.target_qpos = np.copy(self.data.qpos[:6])
        self.create_timer(0.01, self.control_loop)

    def publish_status(self, text):
        self.status_pub.publish(String(data=text))

    def command_callback(self, msg):
        if self.auto is not None and self.auto.running:
            return                      # auto weld owns the arm while it runs
        if len(msg.data) == 6:
            self.target_qpos = np.array(msg.data, dtype=float)

    def control_loop(self):
        self.data.ctrl[:6] = self.target_qpos
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = [float(self.data.qpos[i]) for i in range(6)]
        msg.velocity = [float(self.data.qvel[i]) for i in range(6)]
        self.joint_pub.publish(msg)


def main():
    rclpy.init()
    model = build_model()
    data = mujoco.MjData(model)
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    node = UR5eActuationController(model, data)
    weld = WeldRecorder(model, data)

    tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "torch_tip")
    conv = [model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in ("conv_a", "conv_b")]
    auto = AutoWelder(model, node, tip_id, conv[0])
    node.auto = auto

    with mujoco.viewer.launch_passive(model, data) as viewer:
        step = 0
        while viewer.is_running() and rclpy.ok():
            t0 = time.time()

            if node.auto_toggle:
                node.auto_toggle = False
                if auto.running:
                    auto.abort(data)
                else:
                    auto.start(data)
            auto.update(data, model.opt.timestep)
            belt_on = auto.conveyor_cmd if auto.running else node.conveyor_active
            torch_on = auto.weld_on if auto.running else node.welding_active

            # conveyor: both boxes move together, wrap back and wipe bead on wrap
            for adr in conv:
                data.qvel[adr] = CONVEYOR_SPEED if belt_on else 0.0
            if data.qpos[conv[0]] > CONVEYOR_LIMIT:
                for adr in conv:
                    data.qpos[adr] -= 2 * CONVEYOR_LIMIT
                weld.clear()

            mujoco.mj_step(model, data)
            rclpy.spin_once(node, timeout_sec=0.0)

            if node.clear_requested:
                weld.clear()
                node.clear_requested = False

            tip = data.site_xpos[tip_id].copy()
            arcing = weld.update(tip, torch_on)

            step += 1
            if step % 8 == 0:                      # ~60 Hz rendering
                with viewer.lock():
                    weld.draw(viewer.user_scn, tip, arcing)
                viewer.sync()

            dt = model.opt.timestep - (time.time() - t0)
            if dt > 0:
                time.sleep(dt)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
