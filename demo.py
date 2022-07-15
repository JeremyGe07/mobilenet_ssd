# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 16:33:30 2022

@author: GJZ
"""

import torch
from torch.autograd import Variable
##单位矩阵来模拟输入
input=torch.ones(1,3,15,15)
input=Variable(input)
x=torch.nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=2, padding=1, bias=False, padding_mode='zeros')
out=x(input)
kernels=list(x.parameters())
print(kernels)
print('[8,3,3,3]')
r=kernels[0]
#如r1=[4,1,:,:]
r1=kernels[0][4][1]
print(r1.dtype)
print(r1)
r2=kernels[0][4][1][0][2]
