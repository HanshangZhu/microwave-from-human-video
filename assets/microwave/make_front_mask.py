#!/usr/bin/env python
"""Mask of the parts of front_photo.jpg that stand proud of the front plane far enough
to show parallax in a straight-on photo: the handle bar with its posts (2.6 cm) and
the knob (2.2 cm). bake_colors.py --decal-mask inpaints them out of the front decal
(the 3D parts cover that area) and colours those parts from the unmasked photo.
The buttons (4 mm) stay painted. Pixel shapes read off the full-resolution photo.

  python assets/microwave/make_front_mask.py   # -> front_photo_mask.png
"""
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
img = cv2.imread(os.path.join(HERE, "front_photo.jpg"))
m = np.zeros(img.shape[:2], np.uint8)
cv2.rectangle(m, (2725, 760), (2940, 2400), 255, -1)       # handle bar + posts
cv2.circle(m, (3315, 1930), 185, 255, -1)                   # knob, ring and chrome face
cv2.imwrite(os.path.join(HERE, "front_photo_mask.png"), m)
print("front_photo_mask.png:", int((m > 0).sum()), "px")
