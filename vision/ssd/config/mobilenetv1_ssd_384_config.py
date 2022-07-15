import numpy as np

from vision.utils.box_utils import SSDSpec, SSDBoxSizes, generate_ssd_priors

num_classes = 2

image_size = 384
image_mean = np.array([0, 0, 0])  # RGB layout
image_std = 255.0
iou_threshold = 0.45
center_variance = 0.1
size_variance = 0.2

specs = [
    SSDSpec(24, 16, SSDBoxSizes(76, 120), [2, 3]),
    SSDSpec(12, 32, SSDBoxSizes(120, 164), [2, 3]),
    SSDSpec(6, 64, SSDBoxSizes(164, 208), [2, 3]),
    SSDSpec(3, 100, SSDBoxSizes(208, 252), [2, 3]),
    SSDSpec(2, 150, SSDBoxSizes(252, 296), [2, 3]),
    SSDSpec(1, 300, SSDBoxSizes(296, 340), [2, 3])
]

# # TODO change to
# specs = [
#     SSDSpec(19, 16, SSDBoxSizes(60, 95), [2, 3]),
#     SSDSpec(10, 32, SSDBoxSizes(95, 130), [2, 3]),
#     SSDSpec(5, 64, SSDBoxSizes(130, 165), [2, 3]),
#     SSDSpec(3, 100, SSDBoxSizes(165, 200), [2, 3]),
#     SSDSpec(2, 150, SSDBoxSizes(200, 235), [2, 3]),
#     SSDSpec(1, 300, SSDBoxSizes(235, 270), [2, 3])
# ]

priors = generate_ssd_priors(specs, image_size)
