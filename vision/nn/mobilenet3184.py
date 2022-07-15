# borrowed from "https://github.com/marvis/pytorch-mobilenet"

import torch.nn as nn
import torch.nn.functional as F


class MobileNetV1(nn.Module):
    def __init__(self, width_mult=1.,num_classes=1024,cfg=None):
        self.width_mult=width_mult
        super(MobileNetV1, self).__init__()

        def conv_bn(inp, oup, stride):
            # oup = int(oup*self.width_mult)
            oup = round(oup*self.width_mult)
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            # inp = int(inp*self.width_mult)
            # oup = int(oup*self.width_mult)
            inp = round(inp*self.width_mult)
            oup = round(oup*self.width_mult)
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )
        def conv_pw(inp, oup):

            return nn.Sequential(
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )
        def conv_d(inp, oup, stride):

            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
            )
        if cfg is None:
            self.model = nn.Sequential(
                conv_bn(3, 17, 2),
                conv_dw(17, 32, 1),
                conv_dw(32, 63, 2),
                conv_dw(63, 63, 1),
                conv_dw(63, 126, 2),
                conv_dw(126, 126, 1),
                conv_dw(126, 252, 2),
                conv_dw(252, 252, 1),
                conv_dw(252, 252, 1),
                conv_dw(252, 252, 1),
                conv_dw(252, 252, 1),
                conv_dw(252, 124, 1),
                conv_dw(124, 502, 2),
                conv_dw(502, 352, 1),
                # conv_bn(3, 32*width_mult, 2),
                # conv_dw(32*width_mult, 64*width_mult, 1),
                # conv_dw(64*width_mult, 128*width_mult, 2),
                # conv_dw(128*width_mult, 128*width_mult, 1),
                # conv_dw(128*width_mult, 256*width_mult, 2),
                # conv_dw(256*width_mult, 256*width_mult, 1),
                # conv_dw(256*width_mult, 512*width_mult, 2),
                # conv_dw(512*width_mult, 512*width_mult, 1),
                # conv_dw(512*width_mult, 512*width_mult, 1),
                # conv_dw(512*width_mult, 512*width_mult, 1),
                # conv_dw(512*width_mult, 512*width_mult, 1),
                # conv_dw(512*width_mult, 512*width_mult, 1),
                # conv_dw(512*width_mult, 1024*width_mult, 2),
                # conv_dw(1024*width_mult, 1024*width_mult, 1),
            )
            self.fc = nn.Linear(int(80*self.width_mult), num_classes)
        else:
            layers = []
            self.in_channels = 3
            dwstride1=[1,5,9,13,15,17,19,21,25]
            dwstride2=[3,7,11,23]
            pw=[2,4,6,8,10,12,14,16,18,20,22,24,26]
            for v in range(len(cfg)):
                if cfg[v] == 0:
                    cfg[v] = 1
                if v==0:
                    layers += [conv_bn(self.in_channels,cfg[v],2)]
                elif v in dwstride1:
                    layers += [conv_d(self.in_channels,cfg[v],1)]
                elif v in dwstride2:
                    layers += [conv_d(self.in_channels,cfg[v],2)]
                elif v in pw:
                    layers += [conv_pw(self.in_channels,cfg[v])]
                self.in_channels = cfg[v]
            self.model = nn.Sequential(*layers)
            # cfg= [17, 18, 47, 55, 80, 114, 66, 103, 115, 203, 94, 149, 107, 331, 142, 264, 97, 247, 37, 289, 26, 244, 177, 31, 1, 29, 200]
            # newcfg=[18, 18, 55, 55, 114, 114, 103, 103, 203, 203, 149, 149, 331, 331, 264, 264, 247, 247, 289, 289, 244, 244, 177, 177, 29, 29, 200]
            self.fc = nn.Linear(self.in_channels, num_classes)

    def forward(self, x):
        x = self.model(x)
        x = F.avg_pool2d(x, 7)
        x = x.view(-1, int(1024*self.width_mult))
        x = self.fc(x)
        return x