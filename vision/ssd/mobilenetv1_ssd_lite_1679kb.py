# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 10:07:00 2022

@author: GJZ
"""



import torch
from torch.nn import Conv2d, Sequential, ModuleList, ReLU, BatchNorm2d
from ..nn.mobilenet1679 import MobileNetV1

from .ssd import SSD
from .predictor import Predictor
from .config import mobilenetv1_ssd_config as config


def SeperableConv2dExtra(in_channels, out_channels, kernel_size=1, stride=1, padding=0):
    """Replace Conv2d with a depthwise Conv2d and Pointwise Conv2d.
    """
    return Sequential(
        Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
                groups=in_channels, stride=stride, padding=padding),
        ReLU(),
        Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
        ReLU(),# newly added
    )

def SeperableConv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0):
    """Replace Conv2d with a depthwise Conv2d and Pointwise Conv2d.
    """
    return Sequential(
        Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
               groups=in_channels, stride=stride, padding=padding),
        ReLU(),
        Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
        # ReLU(),# newly added
    )

def create_mobilenetv1_ssd_lite_1679(num_classes,width_mult=1.0, is_test=False):
    base_net = MobileNetV1(num_classes=1001,width_mult=width_mult).model  # disable dropout layer

    source_layer_indexes = [
        12,
        14,
    ]
    extras = ModuleList([
        Sequential(
            # TODO in_channels=1024? channels in the back * width_mult?
            Conv2d(in_channels=222, out_channels=93, kernel_size=1),
            ReLU(),
            SeperableConv2d(in_channels=93, out_channels=111, kernel_size=3, stride=2, padding=1),
        ),
        Sequential(
            Conv2d(in_channels=111, out_channels=47, kernel_size=1),
            ReLU(),
            SeperableConv2d(in_channels=47, out_channels=56, kernel_size=3, stride=2, padding=1),
        ),
        Sequential(
            Conv2d(in_channels=56, out_channels=47, kernel_size=1),
            ReLU(),
            SeperableConv2d(in_channels=47, out_channels=56, kernel_size=3, stride=2, padding=1),
        ),
        Sequential(
            Conv2d(in_channels=56, out_channels=47, kernel_size=1),
            ReLU(),
            SeperableConv2d(in_channels=47, out_channels=154, kernel_size=3, stride=2, padding=1)
        )
    ])

    regression_headers = ModuleList([
        SeperableConv2d(in_channels=67, out_channels=6 * 4, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=222 , out_channels=6 * 4, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=111, out_channels=6 * 4, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=56, out_channels=6 * 4, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=56, out_channels=6 * 4, kernel_size=3, padding=1),
        Conv2d(in_channels=154, out_channels=6 * 4, kernel_size=1),
    ])

    classification_headers = ModuleList([
        SeperableConv2d(in_channels=67, out_channels=6 * num_classes, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=222, out_channels=6 * num_classes, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=111, out_channels=6 * num_classes, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=56, out_channels=6 * num_classes, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=56, out_channels=6 * num_classes, kernel_size=3, padding=1),
        Conv2d(in_channels=154, out_channels=6 * num_classes, kernel_size=1),
    ])
    

    return SSD(num_classes, base_net, source_layer_indexes,
               extras, classification_headers, regression_headers, is_test=is_test, config=config)


def create_mobilenetv1_ssd_lite_predictor_1679(net, candidate_size=200, nms_method=None, sigma=0.5, device=None):
    predictor = Predictor(net, config.image_size, config.image_mean,
                          config.image_std,
                          nms_method=nms_method,
                          iou_threshold=config.iou_threshold,
                          candidate_size=candidate_size,
                          sigma=sigma,
                          device=device)
    return predictor
