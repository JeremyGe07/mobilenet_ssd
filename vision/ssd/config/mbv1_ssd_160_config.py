# -*- coding: utf-8 -*-
"""
Created on Sat May  7 15:41:14 2022

@author: GJZ
"""

import numpy as np

from vision.utils.box_utils import SSDSpec, SSDBoxSizes, generate_ssd_priors

num_classes = 2

image_size = 160
image_mean = np.array([0, 0, 0])  # RGB layout
image_std = 255.0
iou_threshold = 0.45
center_variance = 0.1
size_variance = 0.2

specs = [
    SSDSpec(20, 8, SSDBoxSizes(32, 54), [2, 3]),
    SSDSpec(10, 16, SSDBoxSizes(54, 76), [2, 3]),
    SSDSpec(5, 32, SSDBoxSizes(76, 98), [2, 3]),
    SSDSpec(2, 80, SSDBoxSizes(98, 120), [2, 3]),
    SSDSpec(1, 160, SSDBoxSizes(120, 142), [2, 3]),
]


priors = generate_ssd_priors(specs, image_size)