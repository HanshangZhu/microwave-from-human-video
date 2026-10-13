"""click_bell with the bell on the half of the table the task never samples.

`click_bell.load_actors` draws the bell from `y in [-0.20, 0.00]`. Every episode
the released checkpoint was post-trained on, and every episode of the September
closed-loop runs, therefore has the bell in front of that line. The recorded
human presses spread over `y in [-0.16, +0.11]`, so a third of them land on the
far side -- a region the policy has never been shown.

This task is the stock one with `ylim` moved across that line and **nothing else
changed**: same x range, same exclusion of the ambiguous `|x| < 0.05` band, same
asset, same contact points, same success check. One variable, so a difference in
success rate between `click_bell` and `click_bell_ood` is attributable to the
bell's position and not to a different task.

Drop in as `RoboTwin/envs/click_bell_ood.py`.
"""
import numpy as np
import sapien

from .click_bell import click_bell
from .utils import *

Y_OOD = [0.02, 0.10]          # strictly outside click_bell's [-0.20, 0.00]


class click_bell_ood(click_bell):

    def load_actors(self):
        rand_pos = rand_pose(xlim=[-0.25, 0.25], ylim=Y_OOD, qpos=[0.5, 0.5, 0.5, 0.5])
        while abs(rand_pos.p[0]) < 0.05:
            rand_pos = rand_pose(xlim=[-0.25, 0.25], ylim=Y_OOD, qpos=[0.5, 0.5, 0.5, 0.5])

        self.bell_id = np.random.choice([0, 1], 1)[0]
        self.bell = create_actor(
            scene=self,
            pose=rand_pos,
            modelname="050_bell",
            convex=True,
            model_id=self.bell_id,
            is_static=True,
        )
        self.add_prohibit_area(self.bell, padding=0.07)
        self.check_arm_function = (self.is_left_gripper_close
                                   if self.bell.get_pose().p[0] < 0
                                   else self.is_right_gripper_close)
