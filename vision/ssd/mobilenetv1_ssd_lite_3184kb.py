# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 10:07:00 2022

@author: GJZ
"""



import torch
from torch.nn import Conv2d, Sequential, ModuleList, ReLU, BatchNorm2d
from ..nn.mobilenet3184 import MobileNetV1

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

def create_mobilenetv1_ssd_lite_3184(num_classes,width_mult=1.0, is_test=False):
    base_net = MobileNetV1(num_classes=1001,width_mult=width_mult).model  # disable dropout layer

    source_layer_indexes = [
        12,
        14,
    ]
    extras = ModuleList([
        Sequential(
            # TODO in_channels=1024? channels in the back * width_mult?
            Conv2d(in_channels=352, out_channels=126, kernel_size=1),
            ReLU(),
            SeperableConv2d(in_channels=126, out_channels=177, kernel_size=3, stride=2, padding=1),
        ),
        Sequential(
            Conv2d(in_channels=177, out_channels=63, kernel_size=1),
            ReLU(),
            SeperableConv2d(in_channels=63, out_channels=89, kernel_size=3, stride=2, padding=1),
        ),
        Sequential(
            Conv2d(in_channels=89, out_channels=63, kernel_size=1),
            ReLU(),
            SeperableConv2d(in_channels=63, out_channels=89, kernel_size=3, stride=2, padding=1),
        ),
        Sequential(
            Conv2d(in_channels=89, out_channels=63, kernel_size=1),
            ReLU(),
            SeperableConv2d(in_channels=63, out_channels=180, kernel_size=3, stride=2, padding=1)
        )
    ])

    regression_headers = ModuleList([
        SeperableConv2d(in_channels=124, out_channels=6 * 4, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=352 , out_channels=6 * 4, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=177, out_channels=6 * 4, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=89, out_channels=6 * 4, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=89, out_channels=6 * 4, kernel_size=3, padding=1),
        Conv2d(in_channels=180, out_channels=6 * 4, kernel_size=1),
    ])

    classification_headers = ModuleList([
        SeperableConv2d(in_channels=124, out_channels=6 * num_classes, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=352, out_channels=6 * num_classes, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=177, out_channels=6 * num_classes, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=89, out_channels=6 * num_classes, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=89, out_channels=6 * num_classes, kernel_size=3, padding=1),
        Conv2d(in_channels=180, out_channels=6 * num_classes, kernel_size=1),
    ])
    

    return SSD(num_classes, base_net, source_layer_indexes,
               extras, classification_headers, regression_headers, is_test=is_test, config=config)


def create_mobilenetv1_ssd_lite_predictor_3184(net, candidate_size=200, nms_method=None, sigma=0.5, device=None):
    predictor = Predictor(net, config.image_size, config.image_mean,
                          config.image_std,
                          nms_method=nms_method,
                          iou_threshold=config.iou_threshold,
                          candidate_size=candidate_size,
                          sigma=sigma,
                          device=device)
    return predictor
