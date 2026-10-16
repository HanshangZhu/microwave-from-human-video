# Demonstrations: what was extracted (2026-10-14, 19:30)

Three takes, `captures/microwave01/take_00{0,1,2}` (40 s, 30 fps, RealSense D435i, fixed
camera at the end of the table). The demonstrated task is **put the pot into the
microwave**: open the door, pick the pot up from the counter, place it on the turntable,
close the door. Takes 0 and 1 are one-handed in that order; **take 2 is two-handed**
(left hand opens the door while the right holds the pot), so its order does not transfer
to a single arm as is.

All positions in the microwave's (= simulation) frame: metres, z up, counter z = 0,
origin under the front-left corner, +y into the microwave. Turntable centre (0.175, 0.145).

| take | door opens (s) | max door deg (s) | door shut (s) | pot start (x, y) | picked from | pick (s) | placed at | place (s) | camera knocked |
|---|---|---|---|---|---|---|---|---|---|
| take_000 | 10.7 | 75 (12.4) | 21.5 | (0.355, -0.266) | (0.480, -0.241) | 14.9 | (0.183, 0.095) | 17.5 | 1.0 deg until 10.7 s |
| take_001 | 11.6 | 60 (12.3) | 18.3 | (0.207, -0.336) | (0.441, -0.279) | 14.6 | (0.172, 0.079) | 17.4 | 1.1 deg until 18.5 s |
| take_002 | 2.6 | 57 (3.8) | 6.9 | (0.303, -0.239) | (0.504, -0.173) | 1.4 | (0.173, 0.111) | 5.9 | no |

How each quantity is obtained (scripts in `recon/`):

- **Camera** (`take_anchor.py`, `register_model.py --seed`): the take's last 10 s at rest,
  depth median-filtered, registered to the measured microwave model; a knocked camera is
  detected from the background's image shift and its earlier pose aligned by ICP.
- **Door angle per frame** (`door_track.py`): each frame's depth in the model frame,
  scored against the door panel for every angle (1 deg); median-filtered over 1/4 s, which
  lags by ~10 deg while the door is shut fast.
- **Pot** (`pot_track.py`): on the counter, the teal lid's ray meets the plane at the
  pot's height (the depth agrees to 1 mm); in the cavity the lid is out of sight, so the
  pot's white side, kept only where its depth lies in the cavity above the turntable.
- **Hand** (`hand_track.py`, MediaPipe + depth): reliable while reaching for and holding
  the handle and the pot outside the cavity; lost inside the dark cavity, so the place
  is taken from the pot, not the hand.
- **Check** (`replay_take.py`): recorded frame next to the simulation rendered from the
  same camera with door, pot and hand keypoints; `<take>/replay.mp4`.
