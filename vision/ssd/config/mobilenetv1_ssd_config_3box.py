import numpy as np

from vision.utils.box_utils import SSDSpec, SSDBoxSizes, generate_ssd_priors3

num_classes = 2

image_size = 300
image_mean = np.array([0, 0, 0])  # RGB layout
image_std = 255.0
iou_threshold = 0.2 #0.45s
center_variance = 0.1
size_variance = 0.2


# TODO change to  , 而且去掉正方形框换成1.4的比例,header 输出个数也要改  参考4box
specs = [
    # SSDSpec(19, 16, SSDBoxSizes(16,16), [2,3, 4]),
    # SSDSpec(10, 32, SSDBoxSizes(46,46), [1.6,2.4, 3]),
    # SSDSpec(5, 64, SSDBoxSizes(76,76), [1.4,2.2, 3]),
    # SSDSpec(3, 100, SSDBoxSizes(106,106), [1.4,2.2, 3]),
    # SSDSpec(2, 150, SSDBoxSizes(136,136), [1.2,1.7, 2.2]),
    # SSDSpec(1, 300, SSDBoxSizes(166,166), [1,1.5, 2])
    SSDSpec(19, 16, SSDBoxSizes(40,40), [1.2,2.1, 3]), 
    SSDSpec(10, 32, SSDBoxSizes(60,60), [1.2,2, 2.8]),
    SSDSpec(5, 64, SSDBoxSizes(80,80), [1.2,1.9, 2.6]),
    SSDSpec(3, 100, SSDBoxSizes(120,120), [1.2,1.8, 2.4]),
    SSDSpec(2, 150, SSDBoxSizes(140,140), [1.2,1.6, 2.2]),
    SSDSpec(1, 300, SSDBoxSizes(160,160), [1,1.5, 2])
]

priors = generate_ssd_priors3(specs, image_size)
 