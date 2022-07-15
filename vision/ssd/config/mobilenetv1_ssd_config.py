import numpy as np

from vision.utils.box_utils import SSDSpec, SSDBoxSizes, generate_ssd_priors

num_classes = 2

image_size = 300
image_mean = np.array([0, 0, 0])  # RGB layout
image_std = 255.0
iou_threshold = 0.45 #used to be 0.45,can be 0.3 or lower. . in hard nms ,if two box very close and iou >0.45, the low score one would be deleted, even if they are two boxs targeting on two diff obejects.
center_variance = 0.1
size_variance = 0.2

specs = [
    SSDSpec(19, 16, SSDBoxSizes(60, 105), [2, 3]),
    SSDSpec(10, 32, SSDBoxSizes(105, 150), [2, 3]),
    SSDSpec(5, 64, SSDBoxSizes(150, 195), [2, 3]),
    SSDSpec(3, 100, SSDBoxSizes(195, 240), [2, 3])
    ,SSDSpec(2, 150, SSDBoxSizes(240, 285), [2, 3]),
    SSDSpec(1, 300, SSDBoxSizes(285, 330), [2, 3])
    # SSDSpec(19, 16, SSDBoxSizes(40, 75), [2, 3]),
    # SSDSpec(10, 32, SSDBoxSizes(75, 110), [2, 3]),
    # SSDSpec(5, 64, SSDBoxSizes(110, 145), [2, 3]),
    # SSDSpec(3, 100, SSDBoxSizes(145, 180), [2, 3])
    # ,SSDSpec(2, 150, SSDBoxSizes(180, 215), [2, 3]),
    # SSDSpec(1, 300, SSDBoxSizes(215, 250), [2, 3])
    
    # SSDSpec(19, 16, SSDBoxSizes(40, 40), [1.6,2.2, 3]),
    # SSDSpec(10, 32, SSDBoxSizes(80, 80), [1.4 ,2,2.6]),
    # SSDSpec(5, 64, SSDBoxSizes(120, 120), [1.4, 2,2.6]),
    # SSDSpec(3, 100, SSDBoxSizes(160, 160), [1.2, 1.8, 2.4])
    # # ,SSDSpec(2, 150, SSDBoxSizes(240, 285), [2, 3]),
    # # SSDSpec(1, 300, SSDBoxSizes(285, 330), [2, 3])
]

    # SSDSpec(19, 16, SSDBoxSizes(60, 105), [2, 3]),
    # SSDSpec(10, 32, SSDBoxSizes(105, 150), [2, 3]),
    # SSDSpec(5, 64, SSDBoxSizes(150, 195), [2, 3]),
    # SSDSpec(3, 100, SSDBoxSizes(195, 240), [2, 3])
    # #,SSDSpec(2, 150, SSDBoxSizes(240, 285), [2, 3]),
    # #SSDSpec(1, 300, SSDBoxSizes(285, 330), [2, 3])


# # TODO change to  , 而且去掉正方形框换成1.4的比例,header 输出个数也要改  参考4box
# specs = [
#     SSDSpec(19, 16, SSDBoxSizes(30,30), [2,3, 4]),
#     SSDSpec(10, 32, SSDBoxSizes(74,74), [1.6,2.4, 3]),
#     SSDSpec(5, 64, SSDBoxSizes(118,118), [1.4,2.2, 3]),
#     SSDSpec(3, 100, SSDBoxSizes(162,162), [1.4,2.2, 3]),
#     SSDSpec(2, 150, SSDBoxSizes(206,206), [1.2,1.7, 2.2]),
#     SSDSpec(1, 300, SSDBoxSizes(250,250), [1,1.5, 2])
# ]

# CrowdHuman
    # SSDSpec(19, 16, SSDBoxSizes(40, 75), [2, 3]),
    # SSDSpec(10, 32, SSDBoxSizes(75, 110), [2, 3]),
    # SSDSpec(5, 64, SSDBoxSizes(110, 145), [2, 3]),
    # SSDSpec(3, 100, SSDBoxSizes(145, 180), [2, 3])
    # ,SSDSpec(2, 150, SSDBoxSizes(180, 215), [2, 3]),
    # SSDSpec(1, 300, SSDBoxSizes(215, 250), [2, 3])

priors = generate_ssd_priors(specs, image_size)
