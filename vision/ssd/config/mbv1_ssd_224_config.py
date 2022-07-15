# -*- coding: utf-8 -*-
"""
Created on Sat May  7 15:41:14 2022

@author: GJZ
"""

import numpy as np

from vision.utils.box_utils import SSDSpec, SSDBoxSizes, generate_ssd_priors

num_classes = 2

image_size = 224
image_mean = np.array([0, 0, 0])  # RGB layout
image_std = 255.0
iou_threshold = 0.45
center_variance = 0.1
size_variance = 0.2

specs = [
    SSDSpec(14, 16, SSDBoxSizes(44, 76), [2, 3]),
    SSDSpec(7, 32, SSDBoxSizes(76, 108), [2, 3]),
    SSDSpec(4, 56, SSDBoxSizes(108, 140), [2, 3]),
    SSDSpec(2, 112, SSDBoxSizes(140, 172), [2, 3]),
    SSDSpec(1, 224, SSDBoxSizes(172, 204), [2, 3]),
]


priors = generate_ssd_priors(specs, image_size)