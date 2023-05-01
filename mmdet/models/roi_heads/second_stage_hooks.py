# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.nn.functional as F
from wtorch.conv_module import ConvModule
from mmcv.runner import BaseModule, auto_fp16
import torch
from ..builder import NECKS


@NECKS.register_module()
class FusionFPNHook(BaseModule):
    r"""Feature Pyramid Network.

    """

    def __init__(self,
                 in_channels,
                 out_channels=None,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform'),
                return_stem=False):
        super().__init__(init_cfg)
        if out_channels is None:
            out_channels = in_channels
        self.return_stem = return_stem
        self.fusion_conv = ConvModule(in_channels*2,out_channels,3,padding=1,conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,act_cfg=act_cfg,inplace=False)

    @auto_fp16()
    def forward(self, inputs,backbone=None):
        net0 = inputs[0]
        shape = net0.shape[-2:]
        net1 = F.interpolate(inputs[1],size=shape,mode='bilinear')
        for x in inputs[2:4]:
            x = F.interpolate(x,size=shape,mode='bilinear')
            net1 = net1+x
        net = torch.cat([net0,net1],dim=1)
        net = self.fusion_conv(net)

        if self.return_stem:
            return tuple([backbone.outs['stem'],net])
        else:
            return tuple([net])