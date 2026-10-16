#!/usr/bin/env python
"""Scripted G1 solution of MicrowavePotG1-v0 (feasibility spike): task IK along dense
Cartesian paths (sim/g1_spike/g1_arm.py), executed with joint-position control in the
physics simulation. No learning, no collision checking (routes are via points).

Task keyframes from the human takes (--take: recon/takes/<take>/pot.json, door.npz): the pot
is picked where the take picked it, placed where the take placed it, the door opened to the
take's maximum angle. Left hand = door, right hand = pot (the roles of take_002).

  door  hook: left hand level, forearm square to the door, fingers pre-curled into an L,
        10 cm to the hinge side of the bar; slide sideways so the distal phalanges go into the
        21.5 mm gap behind the 9 mm bar, curl; pull the bar round the hinge, 1 deg per target
        (only the hook point on the bar and the vertical finger axis are prescribed: the hook
        may turn about the bar). Let go: slide out, back off, round the door's free edge.
  pot   right hand level, palm on the pot's end, the 55 mm body clamped between the thumb and
        the curled fingers; lift 3 cm, check, retry; measure where the pot sits in the hand;
        carry it (torso turning ~70-90 deg) into the cavity onto the take's place point, open,
        back out along the forearm, away from the door.
  shut  hook the bar again (route planned before moving: round the free edge, from beyond it,
        or straight), push it round to 0 with the palm's front, press, let go.

  python sim/g1_spike/g1_solve.py --take take_000 --out <dir> [--video]
"""
import argparse
import json
import os
import sys

import gymnasium as gym
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import sim.microwave_env as M  # noqa: E402
import sim.g1_spike.g1_env as G  # noqa: E402
from sim.g1_spike.g1_arm import G1Kin, G1Arm  # noqa: E402
from mani_skill.utils.wrappers.record import RecordEpisode  # noqa: E402

STANCE = (0.35, -0.45, -0.10, 0.6)          # pelvis x, y (mw), height above counter, yaw (rad, + = to the left); stance_scan.py
DER = json.load(open(M.MW_PARAMS))["derived"]
HANDLE = np.array(DER["handle_bar_centre_closed"])
BAR_Z = 0.11                                 # palm centre on the bar (posts at 0.063 / 0.223; finger stack +-36 mm)
TORSO_DOOR = True
# pot grasp, right hand: palm face (+y_palm) on the pot's +x end (the lid rim, 95 mm long), forearm
# along +y_mw (into the microwave), the 55 mm body clamped between the thumb (near the wrist)
# and the curled fingers; palm centre GRIP_Z above the pot's bottom (palm bottom 3 mm clear of
# the counter). POT_OFF = the pot's origin (bottom centre) in the right palm frame.
GRIP_Z = 0.044
POT_OFF_LEVEL = np.array([0.055, 0.016 + 0.0475 + 0.002, -GRIP_Z])
GRIP_PITCH = 0.0                             # forearm pitched down by this (rad) while holding the pot: the
                                             # grasp is made pitched, so the pot itself stays upright


def pot_grasp(beta=None):
    """(offset of the pot's bottom centre in the right palm frame, palm-frame axis that is the
    pot's vertical) for a hand pitched down by beta about its palm normal."""
    b = GRIP_PITCH if beta is None else beta
    c, s_ = np.cos(b), np.sin(b)
    R = np.array([[c, 0, s_], [0, 1, 0], [-s_, 0, c]])      # palm axes in the level frame
    return R.T @ POT_OFF_LEVEL, R.T @ np.array([0, 0, 1.0])


POT_OFF = POT_OFF_LEVEL
POT_YAW = np.deg2rad(21)                     # pot yaw in mw (the takes do not give it): its 55 mm axis along the
                                             # forearm of the natural pick pose (stance_scan.py: forearm at 111 deg)
PICK_PHI, PLACE_PHI = np.deg2rad(111), np.deg2rad(112)   # forearm yaw (from +x_mw) at pick / place
THUMB_OPEN = (0.8, 0.0)                      # right_one swings the thumb back towards the wrist

HOOK_OFF = np.array([0.1125, -0.050, 0.0])  # bar centre in the LEFT palm frame when hooked: knuckles 29 deg + distal 61 deg,
                                             # so the distal phalanges (14 mm) stand square behind the 9 mm bar,
                                             # the proximal ones 1.5 mm off the door face (finger.py geometry)
HOOK_CURL = (0.50, 1.07)


def door_rot(q):
    c, s = np.cos(q), np.sin(q)
    return np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])


def W(v):
    """mw direction -> world"""
    return M.R_W_MW @ np.asarray(v, float)


class Solver:
    def __init__(self, env, arm):
        self.env, self.u, self.arm = env, env.unwrapped, arm
        self.log = []
        self.phi = PICK_PHI

    def step(self, name, ok, **extra):
        info = self.u.evaluate()
        rec = dict(phase=name, ok=bool(ok), door_deg=round(float(info["door_deg"][0]), 1),
                   pot_mw=np.round(self.u.pot_in_mw().cpu().numpy()[0], 3).tolist(), **extra)
        self.log.append(rec)
        print(rec, flush=True)
        if SNAP_DIR:
            from PIL import Image
            for k, cam in enumerate(SNAP_CAMS):
                self.u._human_render_cameras["render_camera"].camera.set_local_pose(cam)
                img = self.u.render_rgb_array()
                img = np.asarray(img.cpu() if hasattr(img, "cpu") else img)[0]
                Image.fromarray(img).save(os.path.join(SNAP_DIR, f"{len(self.log):02d}_{k}_{''.join(ch if ch.isalnum() else '_' for ch in name[:30])}.png"))
        return ok

    def contacts(self, min_imp=1e-3):
        out = {}
        for c in self.u.scene.sub_scenes[0].get_contacts():
            imp = sum(np.linalg.norm(p.impulse) for p in c.points)
            if imp > min_imp:
                k = " | ".join(sorted(b.entity.name.split("_body_")[-1].replace("scene-0_", "") for b in c.bodies))
                out[k] = round(out.get(k, 0) + imp, 4)
        return out

    def door_deg(self):
        return float(self.u.door_angle()[0])

    def hand_mw(self, side, link="tcp_link"):
        L = self.u.agent.robot.links_map[f"{side}_{link}"]
        return M.world_to_mw(L.pose.p.cpu().numpy()[0])

    # ---------------------------------------------------------------- door
    def bar(self, q, z=None):
        return door_rot(q) @ np.array([HANDLE[0], HANDLE[1], BAR_Z if z is None else z])

    def hook_task(self, p_mw, q_door, w_face=0.3):
        """Left palm: HOOK_OFF at p_mw, finger axis (palm z) vertical, forearm (palm x)
        along the door's inward normal (weight w_face)."""
        n = door_rot(q_door) @ [0, 1.0, 0]
        return dict(p=M.mw_to_world(p_mw), offset=HOOK_OFF,
                    axes=[((0, 0, 1), W([0, 0, 1]), 1.0), ((1, 0, 0), W(n), w_face)])

    # ---------------------------------------------------------------- pot
    def pot_task(self, p_mw, phi=None, w_face=0.3):
        """Right palm: the grasp offset at p_mw (the pot's bottom centre), the pot's vertical
        vertical (it stays upright), forearm yawed phi from +x_mw (soft: the hand may turn
        about z, and the pot with it)."""
        off, up = pot_grasp()
        if getattr(self, "grip_off", None) is not None:
            off, up = self.grip_off, self.grip_up
        phi = self.phi if phi is None else phi
        c, s_ = np.cos(GRIP_PITCH), np.sin(GRIP_PITCH)
        fwd = np.array([c, 0, s_])                       # the level frame's x axis in palm coordinates
        return dict(p=M.mw_to_world(p_mw), offset=off,
                    axes=[(up, W([0, 0, 1]), 1.0), (fwd, W([np.cos(phi), np.sin(phi), 0]), w_face)])

    def lines(self, pts, step_len=0.005):
        out = []
        for a_, b_ in zip(pts[:-1], pts[1:]):
            n = int(np.ceil(np.linalg.norm(b_ - a_) / step_len))
            out += [a_ + (b_ - a_) * t for t in np.linspace(0, 1, n + 1)[1:]]
        return out

    def run_path(self, side, pts, name, torso=True, settle=6, phis=None, **kw):
        task = self.pot_task if side == "right" else None
        phis = [None] * len(pts) if phis is None else phis
        qs = self.arm.path(side, [task(p, ph) for p, ph in zip(pts, phis)], torso=torso,
                           link=self.arm.palm_link(side), **kw)
        ok = self.step(f"{name} ({len(qs)}/{len(pts)})", len(qs) == len(pts))
        if ok:
            self.arm.execute(qs, settle=settle)
        return ok

    def held(self):
        """The left hand while it is hooked on the door handle (it must stay on the bar)."""
        return "left" if "left" in self.arm.hold_offset else None

    def pot_pose_mw(self):
        """(bottom-centre position, yaw of the long side) of the pot in mw."""
        from transforms3d.quaternions import quat2mat
        p = self.u.pot_in_mw().cpu().numpy()[0]
        Rw = quat2mat(self.u.pot.pose.q.cpu().numpy()[0])
        x_mw = M.R_W_MW.T @ Rw[:, 0]
        return p, float(np.arctan2(x_mw[1], x_mw[0]))

    def measure_grip(self):
        """Where the pot actually sits in the right palm (its origin and its vertical axis, palm
        frame): the carry and the place then put the POT where it should go, upright, rather
        than the nominal grasp point."""
        from transforms3d.quaternions import quat2mat
        L = self.u.agent.robot.links_map["right_palm_link"]
        Tp = L.pose.sp.to_transformation_matrix()
        pw = self.u.pot.pose.p.cpu().numpy()[0]
        Rw = quat2mat(self.u.pot.pose.q.cpu().numpy()[0])
        self.grip_off = (np.linalg.inv(Tp) @ np.r_[pw, 1.0])[:3]
        self.grip_up = Tp[:3, :3].T @ Rw[:, 2]
        return self.grip_off, self.grip_up

    def pick(self, close=(0.75, 0.75), thumb_close=(-0.35, -0.6), tries=3):
        arm, step, u = self.arm, self.step, self.u
        self.grip_off = self.grip_up = None
        for k in range(tries):
            p, yaw = self.pot_pose_mw()
            p = np.array([p[0], p[1], 0.0])
            # forearm along the pot's 55 mm axis, of the two directions the one nearer the planned one
            phis = [yaw + np.pi / 2, yaw - np.pi / 2]
            self.phi = min(phis, key=lambda f: abs(np.angle(np.exp(1j * (f - PICK_PHI)))))
            arm.set_hand("right", 0.0, 0.0, *THUMB_OPEN)
            above = p + [0, 0, 0.10]
            q0 = arm.first("right", self.pot_task(above), torso=True, link=arm.palm_link("right"))
            if not step(f"pot: above IK (try {k + 1})", q0 is not None):
                return False
            arm.move_joint(q0, step=0.02, hold=self.held())
            if not self.run_path("right", self.lines([above, p + [0, 0, 0.002]]), "pot: down beside it"):
                return False
            arm.set_hand("right", close[0], close[1], *thumb_close, n=30)
            step("pot: closed", True, contacts=self.contacts())
            z0 = self.u.pot_in_mw()[0, 2].item()
            ok = self.run_path("right", self.lines([p + [0, 0, 0.002], p + [0, 0, 0.03]]), "pot: lift 3 cm")
            if ok and self.u.pot_in_mw()[0, 2].item() > z0 + 0.015:
                break
            # not holding it: open, back up, look again
            step("pot: not lifted, retry", False, pot_z=round(self.u.pot_in_mw()[0, 2].item(), 3))
            arm.set_hand("right", 0.0, 0.0, *THUMB_OPEN, n=20)
            here = self.hand_pot_origin()
            self.run_path("right", self.lines([here, here + [0, 0, 0.10]]), "pot: hand up")
        else:
            return False
        here = self.hand_pot_origin()
        ok = self.run_path("right", self.lines([here, here + [0, 0, 0.07]]), "pot: lift")
        off, up = self.measure_grip()
        return step("pot: lifted", ok and self.u.pot_in_mw()[0, 2].item() > 0.06,
                    pot_z=round(self.u.pot_in_mw()[0, 2].item(), 3),
                    grip_off_mm=np.round(1000 * (off - pot_grasp()[0]), 1).tolist(),
                    pot_tilt_deg=round(float(np.degrees(np.arccos(np.clip(up @ pot_grasp()[1], -1, 1)))), 1))

    def place(self, xy):
        arm, step, der = self.arm, self.step, DER
        x, y = xy
        z_tt = der["turntable_top_z"]
        start = self.u.pot_in_mw().cpu().numpy()[0]
        pts = [start, np.array([x, -0.15, 0.10]), np.array([x, -0.15, z_tt + 0.02]),
               np.array([x, y, z_tt + 0.02]), np.array([x, y, z_tt + 0.002])]
        line = self.lines(pts)
        phis = list(np.linspace(self.phi, PLACE_PHI, len(line)))
        q_end = self.arm.first("right", self.pot_task(pts[-1], PLACE_PHI), torso=True, link=self.arm.palm_link("right"))
        if not step("pot: place IK", q_end is not None):
            return False
        ok = self.run_path("right", line, "pot: carry, in, down", settle=10, phis=phis, q_end=q_end, max_jump=0.8)
        if not ok:
            return False
        arm.set_hand("right", 0.0, 0.0, *THUMB_OPEN, n=30)
        step("pot: released", True, contacts=self.contacts())
        here = self.hand_pot_origin()
        f = np.array([np.cos(PLACE_PHI), np.sin(PLACE_PHI), 0.0])     # forearm direction (horizontal)
        side = np.array([np.sin(PLACE_PHI), -np.cos(PLACE_PHI), 0.0])  # away from the palm face (-y_palm)
        # 1 cm off the pot sideways, then straight back out along the forearm, then up
        # out of the cavity along the forearm to y = -0.06 (just outside the opening), then away
        # from the open door (+x) and back, then up
        k_out = max(0.0, (here[1] + 0.06) / max(1e-3, f[1]))
        p_out = here + 0.01 * side - k_out * f
        back = [here, here + 0.01 * side, p_out, p_out + [0.12, -0.12, 0.0], p_out + [0.12, -0.12, 0.08]]
        q_end = arm.first("right", self.pot_task(back[-1], PLACE_PHI), torso=True, link=arm.palm_link("right"))
        ok = q_end is not None and self.run_path("right", self.lines(back), "pot: hand out", max_jump=0.6,
                                                 phis=None, q_end=q_end)
        if ok and PARK_RIGHT:
            arm.move_joint(self.park_right(), step=0.03, hold=self.held())
        return step("pot: placed, hand out", ok)

    def park_right(self):
        """The right arm back to its raised start pose (torso and left arm as they are)."""
        q = self.arm.q_cmd.copy()
        q[G.RIGHT_ARM] = G.INIT_QPOS[G.RIGHT_ARM]
        return q

    def hand_pot_origin(self):
        """Where POT_OFF of the right palm is now (commanded), in mw."""
        T = self.arm.kin.fk(self.arm.q_cmd, self.arm.palm_link("right"))
        off = pot_grasp()[0] if getattr(self, "grip_off", None) is None else self.grip_off
        return M.world_to_mw((T @ np.r_[off, 1.0])[:3])

    # ---------------------------------------------------------------- door
    def hook(self, curl0=HOOK_CURL, curl1=(0.6, 1.3)):
        """Hook the left hand's fingers behind the bar, at whatever angle the door is: level hand,
        forearm square to the door, fingers pre-curled into an L, 10 cm to the hinge side of the
        bar; in to the gap depth, slide sideways so the fingers go behind the bar, curl."""
        arm, step = self.arm, self.step
        palm = arm.palm_link("left")
        q = self.door_deg()
        tol = dict(tol_p=0.004, tol_a=4.0)
        arm.set_hand("left", *curl0, 0.0, 0.0)
        h = np.array([HANDLE[0], HANDLE[1], BAR_Z])
        pts_d = [h + [-0.10, -0.03, 0], h + [-0.10, 0, 0]] + [h + [-0.10 + d, 0, 0] for d in np.arange(0.01, 0.1001, 0.01)]
        routes = [self.door_frame_line(q, pts_d)]
        if AROUND and q > np.deg2rad(30):
            # the door open (re-hook to shut it): from the robot's side of the door round its free
            # edge to the outside, or from just beyond the edge, or (last) straight there
            routes = [self.door_frame_line(q, AROUND_PTS + pts_d), self.door_frame_line(q, AROUND_PTS[1:] + pts_d)] + routes
        # plan each route (IK at its start, continuation along it) before moving; take the first
        # that works all the way
        for r, pts in enumerate(routes):
            line = [pts[0]] + self.lines(pts)
            q0 = arm.first("left", self.hook_task(line[0], q), torso=TORSO_DOOR, link=palm, **tol)
            if q0 is None:
                continue
            saved = arm.q_cmd.copy()
            arm.q_cmd = q0.copy()
            qs = arm.path("left", [self.hook_task(p, q) for p in line[1:]], torso=TORSO_DOOR, link=palm,
                          max_jump=0.6, **tol)
            arm.q_cmd = saved
            if len(qs) == len(line) - 1:
                break
        else:
            return step(f"hook at {np.degrees(q):.0f} deg: no route", False)
        step(f"hook at {np.degrees(q):.0f} deg: route {r} planned ({len(line)} points)", True)
        arm.move_joint(q0, step=0.02)
        arm.execute(qs, settle=10)
        arm.set_hand("left", *curl1, 0.0, 0.0)
        arm.hold_offset["left"] = HOOK_OFF        # while hooked, the other arm's moves keep the hook on the bar
        arm.hold_fn["left"] = self.hook_hold
        return step("hook: curled", True, contacts=self.contacts())

    def open_door(self, max_deg=60):
        arm, step = self.arm, self.step
        if not self.hook():
            return False
        # pull round the hinge, 1 deg per target; a 4 mm outward bias keeps the finger on the bar
        arc = []
        for d in np.arange(1, max_deg + 1e-6, 1.0):
            q = np.deg2rad(d)
            n = door_rot(q) @ [0, 1.0, 0]
            arc.append(self.hook_task(self.bar(q) - 0.004 * n, q))
        qs = arm.path("left", arc, torso=TORSO_DOOR, link=arm.palm_link("left"), tol_p=0.004, tol_a=4.0, max_jump=0.6)
        step(f"pull: IK along the arc to {len(qs)} deg", len(qs) > 0)
        for q in qs:
            arm.execute([q], settle=1)
        arm.act(10)
        return step("pulled", len(qs) > 0)

    def hook_hold(self, q_robot):
        """Hold task for the hooked left hand while the torso turns for the right arm: the hook
        point stays on the circle the bar's centre runs on round the hinge, at its height, the
        fingers' axis vertical, the forearm (soft) square to the door at that angle -- i.e. the hand
        keeps hold of the door, which may swing with it (it is not required to stay at one angle)."""
        r_bar = np.hypot(HANDLE[0], HANDLE[1])
        hinge_w = M.mw_to_world([0.0, 0.0, 0.0])
        R = M.R_W_MW

        def extra(T):
            h = (T @ np.r_[HOOK_OFF, 1.0])[:3]
            v = (h - hinge_w) @ R                                   # hook point in mw
            e = np.array([np.hypot(v[0], v[1]) - r_bar, v[2] - BAR_Z])
            return e, float(np.abs(e).max())
        T0 = self.arm.kin.fk(q_robot, self.arm.palm_link("left"))
        return dict(extra=extra, axes=[((0, 0, 1), W([0, 0, 1]), 1.0)])

    def door_frame_line(self, q, pts_door):
        """Hook-point targets given in the door frame (rotated to the door at angle q)."""
        return [door_rot(q) @ p for p in pts_door]

    def unhook(self, park=True):
        """Open the fingers to the insertion curl, slide out sideways the way they came in
        (towards the hinge, along the door face), back off from the door, park the arm."""
        arm = self.arm
        q = self.door_deg()
        arm.set_hand("left", *HOOK_CURL, 0.0, 0.0)
        arm.hold_offset.pop("left", None)
        arm.hold_fn.pop("left", None)
        h = np.array([HANDLE[0], HANDLE[1], BAR_Z])
        out = self.lines(self.door_frame_line(q, [h, h + [-0.10, 0, 0], h + [-0.10, -0.08, 0]]))
        if AROUND and q > np.deg2rad(30):
            # the robot stands on the door's inner side once it is open beyond ~50 deg: bring the hand
            # round the door's free edge back to that side, or the arm stays wrapped round the edge
            up = self.lines(self.door_frame_line(q, AROUND_PTS[::-1]))
        elif FROM_ABOVE:
            up = self.lines(self.door_frame_line(q, [h + [-0.10, -0.08, 0], h + [-0.10, -0.10, 0.20]]))
        else:
            up = []
        pts = out + up
        qs = arm.path("left", [self.hook_task(p, q, w_face=0.05) for p in pts], torso=False,
                      link=arm.palm_link("left"), tol_p=0.006, tol_a=5.0, max_jump=0.6)
        ok = self.step(f"unhook: out sideways, back ({min(len(qs), len(out))}/{len(out)}), round/up "
                       f"({max(0, len(qs) - len(out))}/{len(up)})", len(qs) >= len(out))
        arm.execute(qs, settle=10)
        if park:
            # left arm straight up (elbow and hand above the microwave and the door): a torso turn
            # then only spins it about its own axis, it does not sweep into the open door
            qp = arm.q_cmd.copy()
            qp[G.LEFT_ARM] = LEFT_PARK if (FROM_ABOVE or PARK_UP) else G.INIT_QPOS[G.LEFT_ARM]
            arm.move_joint(qp, step=0.03)
            self.step("left arm parked", True)
        return ok

    def shut_door_push(self):
        """(Alternative, less reliable in the tests) push the door shut with the left fist on the handle bar: fingers curled into a fist
        (its front is flat with the palm's front), forearm square to the door from outside, the
        bar pressed 3.5 mm by the fist's front; carried round to 0 deg, 1 deg per target."""
        arm, step = self.arm, self.step
        palm = arm.palm_link("left")
        q0 = self.door_deg()
        arm.set_hand("left", 1.5, 1.5, 0.0, 0.0)
        bar = lambda q: self.bar(q)
        nrm = lambda q: door_rot(q) @ [0, 1.0, 0]
        fist = lambda p, q, depth: dict(p=M.mw_to_world(p), offset=np.array([0.1035 - depth, -0.02, 0.0]),
                                        axes=[((0, 0, 1), W([0, 0, 1]), 1.0), ((1, 0, 0), W(nrm(q)), 0.3)])
        # out in front of the bar (6 cm), then onto it
        q_app = arm.first("left", fist(bar(q0) - 0.06 * nrm(q0), q0, 0.0), torso=True, link=palm, tol_p=0.005)
        if not step(f"shut from {np.degrees(q0):.0f} deg: approach IK", q_app is not None):
            return False
        arm.move_joint(q_app, step=0.02)
        line = self.lines([bar(q0) - 0.06 * nrm(q0), bar(q0)])
        qs = arm.path("left", [fist(p, q0, 0.0) for p in line], torso=True, link=palm, tol_p=0.005, tol_a=5.0,
                      max_jump=0.6)
        if not step(f"shut: fist onto the bar ({len(qs)}/{len(line)})", len(qs) == len(line)):
            return False
        arm.execute(qs, settle=5)
        arc = [fist(bar(np.deg2rad(max(d, 0.0))), np.deg2rad(max(d, 0.0)), 0.0035 + (0.004 if d < 0 else 0.0))
               for d in np.arange(np.degrees(q0) - 1, -3.0 - 1e-6, -1.0)]
        qs = arm.path("left", arc, torso=True, link=palm, tol_p=0.006, tol_a=5.0, max_jump=0.6)
        step(f"shut: IK along the arc ({len(qs)}/{len(arc)})", len(qs) > 0)
        for q in qs:
            arm.execute([q], settle=1)
        arm.act(15)
        ok = step("shut: pushed", self.door_deg() < np.deg2rad(5))
        # straight back out from the closed door
        qe = self.door_deg()
        line = self.lines([bar(qe), bar(qe) - 0.10 * nrm(qe), bar(qe) - 0.10 * nrm(qe) + [0, 0, 0.10]])
        qs = arm.path("left", [fist(p, qe, 0.0) for p in line], torso=True, link=palm, tol_p=0.006, tol_a=5.0,
                      max_jump=0.6)
        arm.execute(qs, settle=10)
        step(f"shut: hand back ({len(qs)}/{len(line)})", True)
        return ok


    def shut_door(self):
        """Push the bar back round to 0 with the hooked hand (the palm's front pushes the bar);
        hook it again first if the hand had let go."""
        arm, step = self.arm, self.step
        if "left" not in arm.hold_fn and not self.hook():
            return False
        q_now = self.door_deg()
        arc = []
        for d in np.arange(np.degrees(q_now) - 1, -3.0 - 1e-6, -1.0):
            q = np.deg2rad(max(d, 0.0))
            n = door_rot(q) @ [0, 1.0, 0]
            # 12 mm deeper than the hooked bar: the palm's front (13 mm in front of the bar
            # when hooked) presses it; at 0 deg 3 mm more, against the door's stop
            arc.append(self.hook_task(self.bar(q) + 0.012 * n + (0.003 * n if d < 0 else 0), q))
        qs = arm.path("left", arc, torso=TORSO_DOOR, link=arm.palm_link("left"), tol_p=0.006, tol_a=5.0, max_jump=0.6)
        step(f"shut: IK along the arc ({len(qs)}/{len(arc)})", len(qs) > 0)
        for q in qs:
            arm.execute([q], settle=1)
        arm.act(15)
        # and press the closed door against its stop
        n0 = np.array([0, 1.0, 0])
        press = [self.hook_task(self.bar(0.0) + (0.015 + d) * n0, 0.0) for d in np.linspace(0, 0.01, 6)]
        qs = arm.path("left", press, torso=TORSO_DOOR, link=arm.palm_link("left"), tol_p=0.006, tol_a=5.0, max_jump=0.6)
        arm.execute(qs, settle=15)
        ok = step("shut: pushed", self.door_deg() < np.deg2rad(5))
        self.unhook(park=True)
        return ok


LEFT_PARK = [-2.8, 0.2, 0.0, 1.5, 0.0]       # shoulder pitch / roll / yaw, elbow pitch / roll
H_BAR = np.array([HANDLE[0], HANDLE[1], BAR_Z])
# door-frame hook-point waypoints from the door's inner side round its free edge (x 0.356) to the
# outside, 8 cm out from the bar: the hand (palm 5 cm hinge-ward of the hook point, curled
# fingers 4 cm past it) clears the edge at +0.12
AROUND_PTS = [H_BAR + [0.12, 0.12, 0], H_BAR + [0.12, -0.08, 0], H_BAR + [-0.10, -0.08, 0]]
AROUND = True
PARK_UP = os.environ.get("G1_PARK_UP", "0") == "1"   # park the free left arm straight up (else: its start pose)
PARK_RIGHT = False                           # True: raise the right arm again after the place (its joint-space
                                             # swing hit the open door in tests); False: it stays out right-front
FROM_ABOVE = False                           # True: park the left arm straight up and re-hook from over the
                                             # door's top edge (tried; less reliable than the direct way)
SNAP_DIR = None
SNAP_CAMS = []


def make_env(video, out, pot_xy, yaw=np.pi / 2):
    x, y, z, ry = STANCE
    G.set_g1_stance((x, y), z, ry)
    # G1_RENDER_BACKEND=cpu + VK_ICD_FILENAMES=...lvp_icd... renders on the CPU (lavapipe) when the GPU is full
    kw = dict(render_mode="rgb_array", render_backend=os.environ.get("G1_RENDER_BACKEND", "gpu")) if video \
        else dict(render_backend="none")
    env = gym.make("MicrowavePotG1-v0", obs_mode="state", control_mode="pd_joint_pos", sim_backend="cpu",
                   pot_mw=(*pot_xy, yaw), max_episode_steps=6000, **kw)
    if out:
        env = RecordEpisode(env, out, save_video=video, save_trajectory=True, trajectory_name="g1_solve",
                            video_fps=30, max_steps_per_video=6000)
    return env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--pot", type=float, nargs=2, default=(0.480, -0.241))   # take_000 pick
    ap.add_argument("--place", type=float, nargs=2, default=(0.183, 0.095))  # take_000 place
    ap.add_argument("--phases", default="door,pick,place,shut")
    ap.add_argument("--open-deg", type=float, default=75)
    ap.add_argument("--two-handed", type=int, default=0, help="1: the left hand keeps hold of the door handle "
                    "while the right hand does the pot (as take_002), then pushes it shut; 0: let go, re-hook to shut")                    # take_000's max door angle
    ap.add_argument("--pot-yaw", type=float, default=None, help="deg, the pot's long side from +x_mw at the pick")
    ap.add_argument("--take", help="pick xy + yaw, place xy and max door angle from "
                                   "Gitignore-but-important/recon/takes/<take>/{pot.json,door.npz}")
    ap.add_argument("--door0", type=float, default=0.0, help="start with the door open (deg), to test later phases")
    ap.add_argument("--snap", default=None, help="save stills (top + side) at every phase step to this dir")
    ap.add_argument("--torso-deg", type=float, default=None, help="waist yaw the IK may use, each way (default g1_arm.TORSO_LIMIT_DEG)")
    ap.add_argument("--stance", type=float, nargs=4, default=None, metavar=("X", "Y", "Z", "YAW"),
                    help="pelvis x y (mw, m), height above the counter (m), yaw (rad); default STANCE (stance_scan.py)")
    a = ap.parse_args()
    if a.torso_deg is not None:
        import sim.g1_spike.g1_arm as _ga
        _ga.TORSO_LIMIT_DEG = a.torso_deg
    if a.stance is not None:
        global STANCE
        STANCE = tuple(a.stance)
    global SNAP_DIR, POT_YAW, PICK_PHI
    if a.take:
        td = os.path.join(ROOT, "Gitignore-but-important/recon/takes", a.take)
        tk = json.load(open(os.path.join(td, "pot.json")))
        a.pot, a.place = (tk["pick"]["x"], tk["pick"]["y"]), (tk["end"]["x"], tk["end"]["y"])
        if a.pot_yaw is None:
            a.pot_yaw = np.degrees(tk["pick"]["yaw"])
        if a.open_deg is None:
            a.open_deg = float(np.degrees(np.nanmax(np.load(os.path.join(td, "door.npz"))["q"])))
        print(f"{a.take}: pick {np.round(a.pot, 3)} yaw {a.pot_yaw:.0f} deg, place {np.round(a.place, 3)}, "
              f"door to {a.open_deg:.0f} deg", flush=True)
    if a.open_deg is None:
        a.open_deg = 75.0                                        # take_000's max door angle
    if a.pot_yaw is not None:
        POT_YAW = np.deg2rad(a.pot_yaw) % np.pi                  # a box: the long side's direction mod 180
        PICK_PHI = POT_YAW + np.pi / 2                          # the forearm along the pot's 55 mm axis
    if a.snap:
        os.makedirs(a.snap, exist_ok=True)
        SNAP_DIR = a.snap
    if a.out:
        os.makedirs(a.out, exist_ok=True)
    env = make_env(a.video or bool(a.snap), a.out, a.pot, POT_YAW)
    if a.snap:
        from mani_skill.utils import sapien_utils
        for eye, tgt in (([0.32, -0.30, 1.15], [0.32, -0.18, 0.1]), ([-0.35, -0.55, 0.45], [0.3, -0.05, 0.08])):
            SNAP_CAMS.append(sapien_utils.look_at(M.mw_to_world(eye).tolist(), M.mw_to_world(tgt).tolist()).sp)
    env.reset(seed=0)
    u = env.unwrapped
    kin = G1Kin(u.agent.robot._objs[0], G.root_pose())
    arm = G1Arm(env, kin)
    if a.door0:
        import torch
        u.mw.set_qpos(torch.tensor([[np.deg2rad(a.door0)]], dtype=torch.float32))
    arm.act(10)
    S = Solver(env, arm)
    ok = True
    if "door" in a.phases:
        ok = S.open_door(a.open_deg)
        if not a.two_handed:
            ok = ok and S.unhook(park=True)
    if ok and "pick" in a.phases:
        ok = S.pick()
    if ok and "place" in a.phases:
        ok = S.place(a.place)
    if ok and "shut" in a.phases:
        ok = S.shut_door()
    info = u.evaluate()
    res = dict(success=bool(info["success"][0]), door_deg=float(info["door_deg"][0]), phases=S.log)
    if a.out:
        json.dump(res, open(os.path.join(a.out, "result.json"), "w"), indent=1, default=float)
    print("RESULT", {k: v for k, v in res.items() if k != "phases"})
    env.close()


if __name__ == "__main__":
    main()
