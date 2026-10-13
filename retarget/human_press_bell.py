"""RoboTwin task: click_bell driven by recorded human presses.

Drop-in for `RoboTwin/envs/human_press_bell.py`. It subclasses `click_bell`, so
the success check, the contact points and the asset are the task's own; what
changes is **where** the bell goes and **how** the gripper comes down.

Episodes are indexed by `seed`, not by `now_ep_num`. RoboTwin advances
`now_ep_num` only on success and retries a failure with a fresh seed, so
indexing by it would retry an unreachable press forever; and the scene has to be
reproducible between the planning pass and the replay pass, where `seed` is the
only stable handle.

Honest accounting of what the human data contributes here:

  bell position     entirely from the recorded press location, including the
                    13 presses that land where click_bell never samples
  arm choice        follows from that position, by the task's own sign(x) rule
  descent waypoints the shape of the measured displacement -- how far down at
                    each step, whether it is one push or several, how long it
                    dwells at the bottom
  NOT the timing    `self.move()` re-plans each waypoint and picks its own
                    velocity profile, so the human's speed does not survive.
                    What survives is the sequence of depths, not its tempo.
  NOT the approach  the object does not move before contact, so the capture
                    contains no approach trajectory; that comes from the planner

Set `HP_SPECS` to the specs JSON from `retarget/events_to_specs.py`.
`HP_WAYPOINTS` (default 8) subsamples the descent — every waypoint is a separate
motion-planning call, so the full 24 is needlessly slow.
"""
import json
import os

import numpy as np
import sapien

from .click_bell import click_bell
from .utils import *

_SPECS = None


def _specs():
    global _SPECS
    if _SPECS is None:
        path = os.environ.get("HP_SPECS")
        if not path:
            raise RuntimeError("set HP_SPECS to the specs json from events_to_specs.py")
        _SPECS = json.load(open(path))["specs"]
    return _SPECS


class human_press_bell(click_bell):

    def setup_demo(self, **kwags):
        # the seed is only used to re-seed numpy upstream and is never stored,
        # but it is our episode identity, so keep it before delegating
        self._hp_seed = int(kwags.get("seed", 0))
        super()._init_task_env_(**kwags)

    @property
    def spec(self):
        s = _specs()
        if self._hp_seed >= len(s):
            raise IndexError(f"seed {self._hp_seed} past the end of {len(s)} specs")
        return s[self._hp_seed]

    def load_actors(self):
        x, y = self.spec["bell_xy"]
        self.bell_id = int(self.spec.get("bell_id", 0))
        # z and quaternion are click_bell's own defaults (rand_pose zlim=[0.741])
        self.bell = create_actor(
            scene=self,
            pose=sapien.Pose([x, y, 0.741], [0.5, 0.5, 0.5, 0.5]),
            modelname="050_bell",
            convex=True,
            model_id=self.bell_id,
            is_static=True,
        )
        self.add_prohibit_area(self.bell, padding=0.07)
        self.check_arm_function = (self.is_left_gripper_close if x < 0
                                   else self.is_right_gripper_close)

    def _waypoints(self):
        """Descent depths in metres, monotone-ish, subsampled."""
        d = np.asarray(self.spec["descent_m"], dtype=float)
        n = int(os.environ.get("HP_WAYPOINTS", 8))
        if len(d) > n:
            d = d[np.linspace(0, len(d) - 1, n).round().astype(int)]
        return d

    def play_once(self):
        spec = self.spec
        arm_tag = ArmTag("right" if self.bell.get_pose().p[0] > 0 else "left")

        # Approach: the planner's, not ours -- see the module docstring.
        self.move(self.grasp_actor(self.bell, arm_tag=arm_tag, pre_grasp_dis=0.1,
                                   grasp_dis=0.1, contact_point_id=0))

        depths = self._waypoints()
        prev = 0.0
        for dz in depths:
            step = float(dz) - prev
            if abs(step) < 1e-4:
                continue
            self.move(self.move_by_displacement(arm_tag, z=-step))
            if self.plan_success is False:
                return self.info
            prev = float(dz)
            self.check_success()

        if prev > 1e-4:                      # retract to where the approach left off
            self.move(self.move_by_displacement(arm_tag, z=prev))
        self.check_success()

        # `info["info"]` is the placeholder table for the instruction generator,
        # which matches a template only when its placeholder set equals these keys
        # exactly. Provenance therefore has to live beside it, not inside it.
        self.info["info"] = {
            "{A}": f"050_bell/base{self.bell_id}",
            "{a}": str(arm_tag),
        }
        self.info["human_press"] = {
            "source_take": spec["take"],
            "source_event": spec["event"],
            "human_peak_mm": spec["human_peak_mm"],
            "capture_xy_mm": spec["capture_xy_mm"],
            "outside_task_distribution": spec["outside_task_distribution"],
            "n_descent_waypoints": int(len(depths)),
        }
        return self.info
