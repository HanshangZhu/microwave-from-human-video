"""Cameras for camera-based policies on MicrowavePot-v0.

  front   the RealSense D435i of the human takes, at the pose registered for the take
          (recon/take_anchor.py + register_model.py: camera -> microwave frame) and with
          its real intrinsics -- so a policy sees the scene from where the real
          camera stood, and the take's own frames are a real-image test set
  wrist   on the Panda's hand, looking along the fingers (as ManiSkill's panda_wristcam):
          the last centimetres of the grasp, which the front camera sees from 1 m away

Both are added to the environment's sensors with MicrowavePotEnv(views=True).
"""
import json
import os

import numpy as np
import sapien
from mani_skill.sensors.camera import CameraConfig

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAKES = os.path.join(ROOT, "Gitignore-but-important/recon/takes")
CAPTURES = os.path.join(ROOT, "Gitignore-but-important/laptop/microwave01")

FRONT_W, FRONT_H = 224, 126          # the D435's 16:9, small enough for a ResNet-18 at batch 256
WRIST_W, WRIST_H = 128, 128
WRIST_FOV = np.deg2rad(87)           # D405's vertical-ish field of view, rounded up
# on the hand where ManiSkill's panda_v3 (panda_wristcam) mounts its RealSense: 4.7 cm off
# the hand's centre, behind the fingers, looking along them (camera +x = hand +z)
WRIST_POSE = sapien.Pose(p=[0.0465, -0.02, 0.036], q=[0.0, 0.7071068, 0.0, 0.7071068])


def front_camera(take="take_000", width=None):
    """(pose in world, 3x3 intrinsics at `width`) of the take's camera, at rest (the take's
    last segment). The intrinsics are the real ones, principal point included (it sits
    26 px below the centre of the 720 px image: ignoring it shifts the view by 3.6 %)."""
    import sim.microwave_env as M
    reg = json.load(open(os.path.join(TAKES, take, "run", "registration.json")))
    segs = json.load(open(os.path.join(TAKES, take, "anchor.json")))["segments"]
    ci = json.load(open(os.path.join(CAPTURES, take + ".json")))["intrinsics"]
    C = np.linalg.inv(np.array(reg["T_model_to_world"])) @ np.array(segs[-1]["T_last_from_cam"])   # OpenCV cam -> mw
    G = C[:3, :3] @ np.diag([1, -1, -1])                         # OpenGL camera axes in mw
    T = np.eye(4)
    T[:3, :3] = np.stack([-G[:, 2], -G[:, 0], G[:, 1]], 1)        # SAPIEN: x forward, y left, z up
    T[:3, 3] = C[:3, 3]
    T = M.T_W_MW @ T                                              # mw -> world
    s = (width or FRONT_W) / ci["width"]
    K = np.array([[ci["fx"] * s, 0, ci["ppx"] * s], [0, ci["fy"] * s, ci["ppy"] * s], [0, 0, 1]])
    return sapien.Pose(T), K


SUPERSAMPLE = 2      # "hq": render at 2x and area-downsample (anti-aliasing), default shader


def camera_configs(env, take="take_000", hq=False):
    """hq: the default shader (lighting, shadows-free PBR) instead of "minimal", rendered at
    SUPERSAMPLE x the resolution -- downsample() brings it back to FRONT_W / WRIST_W."""
    k = SUPERSAMPLE if hq else 1
    shader = "default" if hq else "minimal"
    pose, K = front_camera(take, FRONT_W * k)
    wrist_pose = WRIST_POSE
    if getattr(env, "dr_seed", None) is not None and getattr(env, "dr_style", "") == "doorman":
        from transforms3d.euler import euler2quat
        from transforms3d.quaternions import qmult
        from sim.dr import camera_jitter
        (fp, fr), (wp, wr) = camera_jitter(env.dr_seed)
        pose = sapien.Pose(p=pose.p + fp, q=qmult(euler2quat(*fr), pose.q))
        wrist_pose = sapien.Pose(p=WRIST_POSE.p + wp, q=qmult(WRIST_POSE.q, euler2quat(*wr)))
    hand = env.agent.robot.links_map["panda_hand"]
    return [CameraConfig("front", pose, FRONT_W * k, FRONT_H * k, intrinsic=K, near=0.01, far=10, shader_pack=shader),
            CameraConfig("wrist", wrist_pose, WRIST_W * k, WRIST_H * k, WRIST_FOV, 0.005, 5, mount=hand,
                         shader_pack=shader)]


def downsample(img, name):
    """(H, W, 3) uint8 rendered image -> the policy's resolution (area average when supersampled)."""
    import cv2
    w, h = (FRONT_W, FRONT_H) if name == "front" else (WRIST_W, WRIST_H)
    return img if img.shape[1] == w else cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
