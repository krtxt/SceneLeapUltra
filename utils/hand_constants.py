import torch

# Define slice constants for pose components
TRANSLATION_SLICE = slice(0, 3)
QPOS_SLICE = slice(3, 19)
ROTATION_SLICE = slice(19, None)

SELF_PENETRATION_POINT_RADIUS = 0.01
