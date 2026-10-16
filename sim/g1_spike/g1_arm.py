"""IK for one G1 arm (5 DoF, optionally + the torso yaw), along dense Cartesian paths.

The generalisation of sim/solve_microwave.py's Arm: SAPIEN's pinocchio model of the whole
G1 (robot.create_pinocchio_model()), only one arm's joints (and optionally the torso joint)
move, everything else is held. The G1 arm has no wrist: shoulder pitch / roll / yaw, elbow
pitch, elbow roll (= about the forearm). So a full 6-D hand pose is generally NOT reachable
with the arm alone; the IK is a bounded least-squares on a *task* instead of a pose:

    position of a point on the hand (default: the tcp link, between thumb and fingers)
  + any number of axis constraints  (hand-frame axis a  ->  world direction d, weight w)
  + a small pull towards the seed   (continuation: each path point seeded with the last)

Hand frame (palm link, right hand): +x along the forearm to the finger tips, +z along
the finger stack (the fingers curl ABOUT z), +y the palm face (the thumb sits there). The
left hand is mirrored: its palm face is -y.
"""
import numpy as np
import sapien
from scipy.optimize import least_squares
from transforms3d.quaternions import quat2mat

VERBOSE = False
HOLD_TOL = 0.012     # m: how far a holding hand (on the door handle) may be dragged by a torso move
TORSO_LIMIT_DEG = None  # cap on the waist yaw the IK may use, each way (None: the robot's own limit).
                        # The recorded run swings the waist -107..+38 deg to reach what legs would turn
                        # towards; with the pelvis fixed and 5-DoF arms no cap below ~45 deg leaves all
                        # four hand tasks reachable (stance_scan.py), and at 45 the unhook path fails.
from sim.g1_spike.g1_env import LEFT_ARM, RIGHT_ARM, TORSO, LEFT_FINGERS, RIGHT_FINGERS


class G1Kin:
    """Forward kinematics + task IK for the G1 (world frame; the root pose is fixed)."""

    def __init__(self, robot, root_pose):
        self.robot = robot
        self.pm = robot.create_pinocchio_model()
        self.names = [l.name for l in robot.links]
        self.lim = np.array(robot.get_qlimits(), dtype=np.float64).reshape(-1, 2)
        if TORSO_LIMIT_DEG is not None:
            lim = np.deg2rad(TORSO_LIMIT_DEG)
            self.lim[TORSO] = [max(self.lim[TORSO, 0], -lim), min(self.lim[TORSO, 1], lim)]
        self.T_root = root_pose.to_transformation_matrix()

    def link(self, suffix):
        return [i for i, n in enumerate(self.names) if n.endswith(suffix)][0]

    def fk(self, q, link):
        """4x4 world pose of link at full qpos q."""
        self.pm.compute_forward_kinematics(np.asarray(q, np.float64))
        return self.T_root @ self.pm.get_link_pose(link).to_transformation_matrix()

    def solve(self, q_full, joints, link, p=None, axes=(), offset=(0, 0, 0), w_pos=1.0,
              w_reg=0.02, q_ref=None, max_nfev=200, extra=None):
        """Bounded least squares over `joints` (indices into the 25 qpos) from q_full.
        p: target position of `offset` (in the link frame); axes: [(a_link, d_world, w)];
        extra: a callable T_link -> (residuals, error in m) for other hard constraints.
        Returns (q_full_new, pos_err, max_axis_err_deg)."""
        q0 = np.asarray(q_full, np.float64).copy()
        ref = q0[joints] if q_ref is None else np.asarray(q_ref, np.float64)
        lo, hi = self.lim[joints, 0] + 1e-4, self.lim[joints, 1] - 1e-4
        off = np.r_[offset, 1.0]
        axes = [(np.asarray(a, float), np.asarray(d, float) / np.linalg.norm(d), w) for a, d, w in axes]

        def res(x):
            q = q0.copy()
            q[joints] = x
            T = self.fk(q, link)
            r = []
            if p is not None:
                r.append(w_pos * ((T @ off)[:3] - p))
            for a, d, w in axes:
                r.append(w * (T[:3, :3] @ a - d))
            if extra is not None:
                r.append(extra(T)[0])
            r.append(reg[0] * (x - ref))
            return np.concatenate(r)

        reg = [w_reg]
        x0 = np.clip(q0[joints], lo, hi)
        sol = least_squares(res, x0, bounds=(lo, hi), method="trf", diff_step=1e-3, max_nfev=max_nfev,
                            xtol=1e-8, ftol=1e-8)
        def hard_err(x):
            q = q0.copy()
            q[joints] = x
            T = self.fk(q, link)
            pe = 0.0 if p is None else float(np.linalg.norm((T @ off)[:3] - p))
            if extra is not None:
                pe = max(pe, extra(T)[1])
            ae = max([0.0] + [np.degrees(np.arccos(np.clip(T[:3, :3] @ a @ d, -1, 1))) for a, d, w in axes if w >= 0.5])
            return pe, ae

        pe1, ae1 = hard_err(sol.x)
        if pe1 > 0.002 or ae1 > 2.0:
            # the soft terms (axis weights < 0.5, the pull to the seed) choose the basin; then
            # polish the hard task with them down to 10 %
            axes = [(a, d, w if w >= 0.5 else 0.1 * w) for a, d, w in axes]
            reg[0] = 0.1 * w_reg
            sol = least_squares(res, sol.x, bounds=(lo, hi), method="trf", diff_step=1e-3, max_nfev=max_nfev,
                                xtol=1e-8, ftol=1e-8)
        q = q0.copy()
        q[joints] = sol.x
        # axis error: over the hard constraints only (weight >= 0.5); softer ones are preferences
        pe, ae = hard_err(sol.x)
        return q, pe, ae


class G1Arm:
    """Executes joint paths on the live env with pd_joint_pos (all 25 joints commanded:
    the arm being moved, the other arm and the torso held, the fingers as set)."""

    def __init__(self, env, kin):
        self.env, self.u, self.kin = env, env.unwrapped, kin
        self.q_cmd = self.u.agent.robot.get_qpos()[0].cpu().numpy().astype(np.float64)
        self.frames = []
        self.hold_offset = {}
        self.hold_fn = {}        # side -> q -> solve() kwargs: a custom hold (the hook riding on the door's arc)

    def hold_task(self, side, q):
        """Keep `side`'s hand where it is at q: its hold point (hold_offset[side], palm frame;
        default the tcp) in place and its finger axis as it is -- the hand may turn about that
        axis (a hook on a vertical bar allows exactly that)."""
        if side in self.hold_fn:
            return self.hold_fn[side](q)
        off = np.asarray(self.hold_offset.get(side, (0.07, 0.035 if side == "right" else -0.035, 0.0)))
        T = self.kin.fk(q, self.palm_link(side))
        return dict(p=(T @ np.r_[off, 1.0])[:3], offset=off, axes=[((0, 0, 1), T[:3, 2], 1.0)])

    def joints(self, side, torso=False):
        j = list(LEFT_ARM if side == "left" else RIGHT_ARM)
        return ([TORSO] + j) if torso else j

    def tcp_link(self, side):
        return self.kin.link(f"{side}_tcp_link")

    def palm_link(self, side):
        return self.kin.link(f"{side}_palm_link")

    def act(self, n=1):
        for _ in range(n):
            self.env.step(self.q_cmd.astype(np.float32))

    def path(self, side, targets, torso=False, link=None, tol_p=0.004, tol_a=4.0, max_jump=0.25, q_end=None,
             w_end=0.05, **kw):
        """Joint configurations along `targets` (list of dicts of solve() kwargs: p, axes,
        offset), each seeded with the previous. Stops where IK fails; returns the list."""
        link = self.tcp_link(side) if link is None else link
        js = self.joints(side, torso)
        q, out = self.q_cmd.copy(), []
        if torso:
            # the torso turns both arms: the other hand is held where it is (position + finger axis)
            other = "right" if side == "left" else "left"
            o_link, o_js = self.palm_link(other), self.joints(other)
            hold = self.hold_task(other, q)
        q_start = q.copy()
        for i, t in enumerate(targets):
            if q_end is not None:
                # a soft pull towards the joint-space interpolation between the start and a known
                # solution at the end: keeps the redundant torso / arm on a sensible course
                f = (i + 1) / len(targets)
                kw = {**kw, "q_ref": (q_start[js] * (1 - f) + np.asarray(q_end)[js] * f), "w_reg": w_end}
            qn, pe, ae = self.kin.solve(q, js, link, **{**kw, **t})
            if pe > tol_p or ae > tol_a or np.abs(qn[js] - q[js]).max() > max_jump:
                self.fail = dict(at=len(out), pos_mm=round(pe * 1000, 1), axis_deg=round(ae, 1),
                                 jump=round(float(np.abs(qn[js] - q[js]).max()), 3),
                                 dq=np.round(qn[js] - q[js], 3).tolist(), q=np.round(q[js], 2).tolist())
                print("    path stops:", self.fail, flush=True)
                break
            if torso:
                qn, pe_o, ae_o = self.kin.solve(qn, o_js, o_link, **hold)
                if pe_o > HOLD_TOL and other in self.hold_offset:
                    # the other hand is holding something (the door handle) and would be dragged
                    self.fail = dict(at=len(out), other_hand_mm=round(pe_o * 1000, 1))
                    print("    path stops:", self.fail, flush=True)
                    break
            out.append(qn)
            q = qn
        return out

    def first(self, side, target, torso=False, link=None, n_seeds=16, tol_p=0.004, tol_a=4.0, **kw):
        """IK for one target from several seeds (the current arm, the init pose, random);
        the feasible solution closest to the current arm, or None. Prints the best error."""
        from sim.g1_spike.g1_env import INIT_QPOS
        link = self.tcp_link(side) if link is None else link
        js = self.joints(side, torso)
        rng = np.random.default_rng(len(self.frames))
        lim = self.kin.lim[js]
        seeds = [self.q_cmd.copy()]
        s = self.q_cmd.copy(); s[js] = INIT_QPOS[js]; seeds.append(s)
        for _ in range(n_seeds):
            s = self.q_cmd.copy(); s[js] = rng.uniform(lim[:, 0], lim[:, 1]) * 0.8; seeds.append(s)
        # a natural arm: torso straight, upper arm forward-down, elbow a little bent
        nom = self.q_cmd.copy()
        nom[TORSO] = 0.0
        arm_j = LEFT_ARM if side == "left" else RIGHT_ARM
        sg = 1.0 if side == "left" else -1.0
        nom[arm_j] = [-0.6, sg * 0.2, 0.0, 0.4, 0.0]
        seeds.insert(0, nom.copy())
        best, err = None, None
        for s in seeds:
            q, pe, ae = self.kin.solve(s, js, link, w_reg=0.01, q_ref=nom[js], **{**kw, **target})
            if err is None or pe + 0.001 * ae < err[0] + 0.001 * err[1]:
                err = (pe, ae)
            if pe < tol_p and ae < tol_a:
                if torso:
                    # the torso moved the other arm: it must still be able to hold its hand
                    other = "right" if side == "left" else "left"
                    q, pe_o, ae_o = self.kin.solve(q, self.joints(other), self.palm_link(other),
                                                   **self.hold_task(other, self.q_cmd))
                    if (pe_o > HOLD_TOL or ae_o > 4) and other in self.hold_offset:
                        continue
                # prefer solutions without an over-the-head shoulder pitch, closest to the natural arm
                d = np.abs(q[js] - nom[js]).sum() + (100.0 if abs(q[js[-5]]) > 1.8 else 0.0)
                if best is None or d < best[0]:
                    best = (d, q)
        if best is None:
            print(f"    IK failed, best pos err {err[0]*1000:.1f} mm, axis err {err[1]:.1f} deg", flush=True)
        return None if best is None else best[1]

    def execute(self, qs, settle=8, keep_fingers=True, max_step=0.04):
        for q in qs:
            q = np.asarray(q).copy()
            if keep_fingers:
                q[LEFT_FINGERS + RIGHT_FINGERS] = self.q_cmd[LEFT_FINGERS + RIGHT_FINGERS]
            # big joint steps (near the stretched-arm singularity) are split into sub-steps
            q0 = self.q_cmd.copy()
            n = int(max(1, np.ceil(np.abs(q - q0).max() / max_step)))
            for k in range(1, n + 1):
                self.q_cmd = q0 + (q - q0) * k / n
                self.act()
        self.act(settle)

    def move_joint(self, q_target, step=0.03, settle=8, hold=None):
        """Straight line in joint space. hold='left'/'right': that hand is kept where it is
        (hold_task) at every step, its arm re-solved, while the torso and the other arm move."""
        q0 = self.q_cmd.copy()
        qt = np.asarray(q_target).copy()
        qt[LEFT_FINGERS + RIGHT_FINGERS] = q0[LEFT_FINGERS + RIGHT_FINGERS]
        n = int(max(np.ceil(np.abs(qt - q0).max() / step), 1))
        task = None if hold is None else self.hold_task(hold, q0)
        hj = None if hold is None else self.joints(hold)
        q_h = q0.copy()
        for k in range(1, n + 1):
            q = q0 + (qt - q0) * k / n
            if hold is not None:
                q[hj] = q_h[hj]
                q, pe, _ = self.kin.solve(q, hj, self.palm_link(hold), **task)
                q_h = q
            self.q_cmd = q
            self.act()
        self.act(settle)

    def set_hand(self, side, three, four, one, two, n=20):
        """Finger joints by magnitude (the sign per hand is applied here): three/five = finger
        knuckles, four/six = distal joints, one = thumb swing (+ = open, back towards the wrist),
        two = thumb curl (magnitude <= 0 ... use negative for curling towards the palm face)."""
        fj = LEFT_FINGERS if side == "left" else RIGHT_FINGERS
        sg = 1.0 if side == "right" else -1.0
        # order: zero, three, five, one, four, six, two
        target = np.array([0.0, sg * three, sg * three, sg * one, sg * four, sg * four, sg * two])
        target = np.clip(target, self.kin.lim[fj, 0], self.kin.lim[fj, 1])
        q0 = self.q_cmd[fj].copy()
        for k in range(1, n + 1):
            self.q_cmd[fj] = q0 + (target - q0) * k / n
            self.act()

    def hand(self, side, closure, n=20, thumb=None):
        """closure 0 = open (fingers straight), 1 = fully curled. thumb: (one, two) joint
        values; default curls with the fingers."""
        fj = LEFT_FINGERS if side == "left" else RIGHT_FINGERS
        lo, hi = self.kin.lim[fj, 0], self.kin.lim[fj, 1]
        # finger joints curl towards the palm face: right hand positive, left hand negative
        target = np.zeros(len(fj))
        names = ["zero", "three", "five", "one", "four", "six", "two"]
        for k, nm in enumerate(names):
            if nm in ("three", "four", "five", "six"):
                target[k] = closure * (hi[k] if side == "right" else lo[k])
            if nm == "two":
                target[k] = closure * (lo[k] if side == "right" else hi[k])
        if thumb is not None:
            target[names.index("one")], target[names.index("two")] = thumb
        q0 = self.q_cmd[fj].copy()
        for k in range(1, n + 1):
            self.q_cmd[fj] = q0 + (target - q0) * k / n
            self.act()
