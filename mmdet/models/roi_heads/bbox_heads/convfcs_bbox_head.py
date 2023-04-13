# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from wtorch.conv_module import ConvModule
from wtorch.fc_module import FCModule
from mmdet.models.builder import HEADS
from mmdet.models.utils import build_linear_layer
from .bbox_head import BBoxHead
'''
与ConvFCBBoxHead的区别为:
1, FC的激活函数由固定的ReLU修改为与convs相同的激活函数(使用act_cfg配置)
2, 增加FC归一化函数, 与convs相同并使用norm_cfg配置
'''


@HEADS.register_module()
class WConvFCSBBoxHead(BBoxHead):
    r"""More general bbox head, with shared conv and fc layers and two optional
    separated branches.

    .. code-block:: none

            /-> cls fcs -> cls
        x 
            \-> reg convs -> reg
    """  # noqa: W605

    def __init__(self,
                 num_cls_fcs=2,
                 num_reg_convs=4,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 conv_cfg=None,
                 norm_cfg=None,
                 init_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 avg_pool_channels=1024,
                 with_avg_pool=True,
                 *args,
                 **kwargs):
        super().__init__(with_avg_pool=with_avg_pool,
            *args, init_cfg=init_cfg, **kwargs)
        self.num_cls_fcs = num_cls_fcs
        self.num_reg_convs = num_reg_convs
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.avg_pool_channels = avg_pool_channels

        self.shared_out_channels = self.in_channels

        # add cls specific branch
        self.cls_fcs, self.cls_convs,self.cls_last_dim = \
            self._add_fc_branch(
                self.num_cls_fcs, self.shared_out_channels)

        # add reg specific branch
        self.reg_convs, self.reg_last_dim = \
            self._add_conv_branch(
                self.num_reg_convs, self.shared_out_channels)

        if not self.with_avg_pool:
            self.reg_last_dim *= self.roi_feat_area

        # reconstruct fc_cls and fc_reg since input channels are changed
        if self.with_cls:
            if self.custom_cls_channels:
                cls_channels = self.loss_cls.get_cls_channels(self.num_classes)
            else:
                cls_channels = self.num_classes + 1
            self.fc_cls = build_linear_layer(
                self.cls_predictor_cfg,
                in_features=self.cls_last_dim,
                out_features=cls_channels)
        if self.with_reg:
            out_dim_reg = (4 if self.reg_class_agnostic else 4 *
                           self.num_classes)
            self.fc_reg = build_linear_layer(
                self.reg_predictor_cfg,
                in_features=self.reg_last_dim,
                out_features=out_dim_reg)

        if init_cfg is None:
            # when init_cfg is None,
            # It has been set to
            # [[dict(type='Normal', std=0.01, override=dict(name='fc_cls'))],
            #  [dict(type='Normal', std=0.001, override=dict(name='fc_reg'))]
            # after `super(WConvFCBBoxHead, self).__init__()`
            # we only need to append additional configuration
            # for `shared_fcs`, `cls_fcs` and `reg_fcs`
            self.init_cfg += [
                dict(
                    type='Xavier',
                    distribution='uniform',
                    override=[
                        dict(name='cls_fcs'),
                    ])
            ]

    def _add_conv_branch(self,
                            num_branch_convs,
                            in_channels):
        """Add shared or separable branch.

        convs -> avg pool (optional) -> fcs
        """
        last_layer_dim = in_channels
        # add branch specific conv layers
        branch_convs = []
        for i in range(num_branch_convs):
            conv_in_channels = (
                last_layer_dim if i == 0 else self.conv_out_channels)
            branch_convs.append(
                ConvModule(
                    conv_in_channels,
                    self.conv_out_channels,
                    3,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))
        last_layer_dim = self.conv_out_channels
        return nn.Sequential(*branch_convs), last_layer_dim


    def _add_fc_branch(self, num_branch_fcs,
                            in_channels,
                            ):
        """Add shared or separable branch.

        convs -> avg pool (optional) -> fcs
        """
        last_layer_dim = in_channels
        # add branch specific conv layers
        branch_fcs = []
        # for shared branch, only consider self.with_avg_pool
        # for separated branches, also consider self.num_shared_fcs
        convs = None
        if not self.with_avg_pool:
            last_layer_dim *= self.roi_feat_area
        elif self.avg_pool_channels is not None:
            convs = nn.Conv2d(last_layer_dim,self.avg_pool_channels,1,1)
            last_layer_dim = self.avg_pool_channels

        for i in range(num_branch_fcs):
            fc_in_channels = (
                last_layer_dim if i == 0 else self.fc_out_channels)
            branch_fcs.append(
                FCModule(fc_in_channels, self.fc_out_channels,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
                )
        last_layer_dim = self.fc_out_channels
        return nn.Sequential(*branch_fcs), convs,last_layer_dim

    def forward(self, x):
        # separate branches
        x_cls = x
        x_reg = x

        if self.cls_convs is not None:
            x_cls = self.cls_convs(x_cls)

        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        x_cls = self.cls_fcs(x_cls)
        x_reg = self.reg_convs(x_reg)

        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)

        cls_score = self.fc_cls(x_cls) if self.with_cls else None  #默认输出C=num_class+1(背景)
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None  #默认输出C=num_class*4
        return cls_score, bbox_pred