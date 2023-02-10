# Copyright (c) OpenMMLab. All rights reserved.
from functools import partial
from itertools import chain
from typing import Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from mmcv.cnn.bricks import DropPath
import wtorch.nn as wnn
from mmcv.runner import BaseModule
from mmcv.runner.base_module import ModuleList, Sequential
from ..builder import BACKBONES


class ConvNeXtBlock(BaseModule):
    """ConvNeXt Block.

    Args:
        in_channels (int): The number of input channels.
        norm_cfg (dict): The config dict for norm layers.
            Defaults to ``dict(type='LN2d', eps=1e-6)``.
        act_cfg (dict): The config dict for activation between pointwise
            convolution. Defaults to ``dict(type='GELU')``.
        mlp_ratio (float): The expansion ratio in both pointwise convolution.
            Defaults to 4.
        linear_pw_conv (bool): Whether to use linear layer to do pointwise
            convolution. More details can be found in the note.
            Defaults to True.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.
        layer_scale_init_value (float): Init value for Layer Scale.
            Defaults to 1e-6.

    Note:
        There are two equivalent implementations:

        1. DwConv -> LayerNorm -> 1x1 Conv -> GELU -> 1x1 Conv;
           all outputs are in (N, C, H, W).
        2. DwConv -> LayerNorm -> Permute to (N, H, W, C) -> Linear -> GELU
           -> Linear; Permute back

        As default, we use the second to align with the official repository.
        And it may be slightly faster.
    """

    def __init__(self,
                 in_channels,
                 norm_cfg='LayerNorm2d',
                 act_cfg='GELU',
                 mlp_ratio=4.,
                 linear_pw_conv=True,
                 drop_path_rate=0.,
                 layer_scale_init_value=1e-6,
                 with_cp=False):
        super().__init__()
        self.with_cp = with_cp

        self.depthwise_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=7,
            padding=3,
            groups=in_channels)

        self.linear_pw_conv = linear_pw_conv
        self.norm = wnn.get_norm(norm_cfg, in_channels)

        mid_channels = int(mlp_ratio * in_channels)
        if self.linear_pw_conv:
            # Use linear layer to do pointwise conv.
            pw_conv = nn.Linear
        else:
            pw_conv = partial(nn.Conv2d, kernel_size=1)

        self.pointwise_conv1 = pw_conv(in_channels, mid_channels)
        self.act = wnn.get_activation(act_cfg)
        self.pointwise_conv2 = pw_conv(mid_channels, in_channels)

        self.gamma = nn.Parameter(
            layer_scale_init_value * torch.ones((in_channels)),
            requires_grad=True) if layer_scale_init_value > 0 else None

        self.drop_path = DropPath(
            drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x):

        def _inner_forward(x):
            shortcut = x
            x = self.depthwise_conv(x)
            x = self.norm(x)

            if self.linear_pw_conv:
                x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)

            x = self.pointwise_conv1(x)
            x = self.act(x)
            x = self.pointwise_conv2(x)

            if self.linear_pw_conv:
                x = x.permute(0, 3, 1, 2)  # permute back

            if self.gamma is not None:
                x = x.mul(self.gamma.view(1, -1, 1, 1))

            x = shortcut + self.drop_path(x)
            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)

        return x

class MultiBranchStem24X(nn.Module):
    def __init__(self,in_channels,out_channels,activation_fn="LeakyReLU"):
        super().__init__()
        self.out_channels = out_channels
        branch_channels = out_channels//4
        self.branch0 = nn.Conv2d(in_channels,branch_channels,3,stride=2,padding=1,bias=False)
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels,4,3,3,0,bias=False),
            wnn.LayerNorm2d(num_features=4),
            wnn.get_activation(activation_fn,inplace=True),
            nn.Conv2d(4,8,3,2,1,bias=False),
            wnn.LayerNorm2d(num_features=8),
            wnn.get_activation(activation_fn,inplace=True),
            nn.Conv2d(8,branch_channels,3,2,1,bias=False))
        self.branch2 = nn.Conv2d(in_channels,branch_channels,12,12,0,bias=False)
        self.branch3 = nn.Conv2d(in_channels,branch_channels,7,stride=2,padding=3,bias=False)
        self.downsample = nn.Sequential(nn.Conv2d(out_channels,out_channels,2,2),
                            wnn.LayerNorm2d(out_channels),
                            wnn.get_activation(activation_fn,inplace=True),
                            )

    def forward(self,x):
        downsampled = torch.nn.functional.interpolate(x,(x.shape[-2]//6,x.shape[-1]//6),mode='bilinear')
        x0 = self.branch0(downsampled)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(downsampled)
        x = torch.cat([x0,x1,x2,x3],dim=1)
        x = self.downsample(x)
        return x

class MultiBranchStemS24X(nn.Module):
    def __init__(self,in_channels,out_channels,activation_fn="LeakyReLU"):
        super().__init__()
        self.out_channels = out_channels
        branch_channels = out_channels//4
        self.branch0_0 = nn.Sequential(
            nn.Conv2d(in_channels,4,3,stride=2,padding=1,bias=False),
            wnn.LayerNorm2d(num_features=4),
            wnn.get_activation(activation_fn,inplace=True),
            nn.Conv2d(4,branch_channels,3,stride=1,padding=1,bias=False))
        self.branch0_1 = nn.ModuleList([nn.MaxPool2d(3,2,1),nn.MaxPool2d(5,2,2)])
        self.branch0_2 = nn.Conv2d(branch_channels*2,branch_channels*2,3,3,padding=0,bias=False)
        self.branch1 = nn.Conv2d(in_channels,branch_channels*2,7,stride=2,padding=3,bias=False)
        self.downsample = nn.Sequential(nn.Conv2d(out_channels,out_channels,2,2),
                            wnn.LayerNorm2d(out_channels),
                            wnn.get_activation(activation_fn,inplace=True),
                            )

    def forward(self,x):
        downsampled = torch.nn.functional.interpolate(x,(x.shape[-2]//6,x.shape[-1]//6),mode='bilinear')
        x0 = self.branch0_0(x)
        x0_0 = self.branch0_1[0](x0)
        x0_1 = self.branch0_1[1](x0)
        x0 = torch.cat([x0_0,x0_1],dim=1)
        x0 = self.branch0_2(x0)
        x1 = self.branch1(downsampled)
        x = torch.cat([x0,x1],dim=1)
        x = self.downsample(x)
        return x

class MultiBranchStemSL24X(nn.Module):
    def __init__(self,in_channels,out_channels,activation_fn="LeakyReLU"):
        super().__init__()
        self.out_channels = out_channels
        branch_channels = out_channels//4
        self.branch0 = nn.Sequential(
            nn.Conv2d(in_channels,4,3,stride=3,padding=0,bias=False),
            wnn.LayerNorm2d(num_features=4),
            wnn.get_activation(activation_fn,inplace=True),
            nn.Conv2d(4,branch_channels,4,stride=4,padding=0,bias=False))
        self.branch1_0 = nn.Sequential(
            nn.Conv2d(in_channels,4,3,stride=2,padding=1,bias=False),
            wnn.LayerNorm2d(num_features=4),
            wnn.get_activation(activation_fn,inplace=True),
            nn.Conv2d(4,8,3,stride=1,padding=1,bias=False),
            wnn.LayerNorm2d(num_features=8),
            wnn.get_activation(activation_fn,inplace=True),
            )
        self.branch1_1 = nn.ModuleList([nn.MaxPool2d(3,2,1),nn.MaxPool2d(5,2,2)])
        self.branch1_2 = nn.Conv2d(branch_channels,branch_channels,3,3,padding=0,bias=False)
        self.branch2 = nn.Conv2d(in_channels,branch_channels*2,7,stride=2,padding=3,bias=False)
        self.downsample = nn.Sequential(nn.Conv2d(out_channels,out_channels,2,2),
                            wnn.LayerNorm2d(out_channels),
                            wnn.get_activation(activation_fn,inplace=True),
                            )

    def forward(self,x):
        downsampled = torch.nn.functional.interpolate(x,(x.shape[-2]//6,x.shape[-1]//6),mode='bilinear')
        x0 = self.branch0(x)
        x1 = self.branch1_0(x)
        x1_0 = self.branch1_1[0](x1)
        x1_1 = self.branch1_1[1](x1)
        x1 = torch.cat([x1_0,x1_1],dim=1)
        x1 = self.branch1_2(x1)
        x2 = self.branch2(downsampled)
        x = torch.cat([x0,x1,x2],dim=1)
        x = self.downsample(x)
        return x

@BACKBONES.register_module()
class WConvNeXt(BaseModule):
    """ConvNeXt.

    A PyTorch implementation of : `A ConvNet for the 2020s
    <https://arxiv.org/pdf/2201.03545.pdf>`_

    Modified from the `official repo
    <https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py>`_
    and `timm
    <https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/convnext.py>`_.

    Args:
        arch (str | dict): The model's architecture. If string, it should be
            one of architecture in ``ConvNeXt.arch_settings``. And if dict, it
            should include the following two keys:

            - depths (list[int]): Number of blocks at each stage.
            - channels (list[int]): The number of channels at each stage.

            Defaults to 'tiny'.
        in_channels (int): Number of input image channels. Defaults to 3.
        stem_patch_size (int): The size of one patch in the stem layer.
            Defaults to 4.
        norm_cfg (dict): The config dict for norm layers.
            Defaults to ``dict(type='LN2d', eps=1e-6)``.
        act_cfg (dict): The config dict for activation between pointwise
            convolution. Defaults to ``dict(type='GELU')``.
        linear_pw_conv (bool): Whether to use linear layer to do pointwise
            convolution. Defaults to True.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.
        layer_scale_init_value (float): Init value for Layer Scale.
            Defaults to 1e-6.
        out_indices (Sequence | int): Output from which stages.
            Defaults to -1, means the last stage.
        frozen_stages (int): Stages to be frozen (all param fixed).
            Defaults to 0, which means not freezing any parameters.
        gap_before_final_norm (bool): Whether to globally average the feature
            map before the final norm layer. In the official repo, it's only
            used in classification task. Defaults to True.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Defaults to False.
        init_cfg (dict, optional): Initialization config dict
    """  # noqa: E501
    arch_settings = {
        'tiny': {
            'depths': [3, 3, 9, 3],
            'channels': [96, 192, 384, 768]
        },
        'small': {
            'depths': [3, 3, 27, 3],
            'channels': [96, 192, 384, 768]
        },
        'base': {
            'depths': [3, 3, 27, 3],
            'channels': [128, 256, 512, 1024]
        },
        'large': {
            'depths': [3, 3, 27, 3],
            'channels': [192, 384, 768, 1536]
        },
        'xlarge': {
            'depths': [3, 3, 27, 3],
            'channels': [256, 512, 1024, 2048]
        },
    }

    def __init__(self,
                 arch='tiny',
                 in_channels=3,
                 stem_patch_size=4,
                 norm_cfg='LayerNorm2d',
                 act_cfg='GELU',
                 linear_pw_conv=True,
                 drop_path_rate=0.,
                 layer_scale_init_value=1e-6,
                 out_indices=-1,
                 frozen_stages=0,
                 gap_before_final_norm=False,
                 with_cp=False,
                 init_cfg=None,
                 deep_stem_mode="default", #default, MultiBranchStem24X, MultiBranchStemS24X, MultiBranchStemSL24X

                 ):
        super().__init__(init_cfg=init_cfg)

        if isinstance(arch, str):
            assert arch in self.arch_settings, \
                f'Unavailable arch, please choose from ' \
                f'({set(self.arch_settings)}) or pass a dict.'
            arch = self.arch_settings[arch]
        elif isinstance(arch, dict):
            assert 'depths' in arch and 'channels' in arch, \
                f'The arch dict must have "depths" and "channels", ' \
                f'but got {list(arch.keys())}.'

        self.depths = arch['depths']
        self.channels = arch['channels']
        assert (isinstance(self.depths, Sequence)
                and isinstance(self.channels, Sequence)
                and len(self.depths) == len(self.channels)), \
            f'The "depths" ({self.depths}) and "channels" ({self.channels}) ' \
            'should be both sequence with the same length.'

        self.num_stages = len(self.depths)

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must by a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = 4 + index
                assert out_indices[i] >= 0, f'Invalid out_indices {index}'
        self.out_indices = out_indices

        self.frozen_stages = frozen_stages
        self.gap_before_final_norm = gap_before_final_norm

        # stochastic depth decay rule
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(self.depths))
        ]
        block_idx = 0

        # 4 downsample layers between stages, including the stem layer.
        self.downsample_layers = ModuleList()
        self.deep_stem_mode = deep_stem_mode
        stem = self._make_deep_stem_layer(in_channels=in_channels,
                                          stem_channels=self.channels[0],
                                          stem_patch_size=stem_patch_size,
                                          norm=norm_cfg,
                                          activation_fn=act_cfg)
        self.downsample_layers.append(stem)

        # 4 feature resolution stages, each consisting of multiple residual
        # blocks
        self.stages = nn.ModuleList()

        for i in range(self.num_stages):
            depth = self.depths[i]
            channels = self.channels[i]

            if i >= 1:
                downsample_layer = nn.Sequential(
                    wnn.LayerNorm2d(self.channels[i - 1]),
                    nn.Conv2d(
                        self.channels[i - 1],
                        channels,
                        kernel_size=2,
                        stride=2),
                )
                self.downsample_layers.append(downsample_layer)

            stage = Sequential(*[
                ConvNeXtBlock(
                    in_channels=channels,
                    drop_path_rate=dpr[block_idx + j],
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    linear_pw_conv=linear_pw_conv,
                    layer_scale_init_value=layer_scale_init_value,
                    with_cp=with_cp) for j in range(depth)
            ])
            block_idx += depth

            self.stages.append(stage)

            if i in self.out_indices:
                norm_layer = wnn.get_norm(norm_cfg, channels)
                self.add_module(f'norm{i}', norm_layer)

        self._freeze_stages()

    def forward(self, x):
        outs = []
        for i, stage in enumerate(self.stages):
            x = self.downsample_layers[i](x)
            x = stage(x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                if self.gap_before_final_norm:
                    gap = x.mean([-2, -1], keepdim=True)
                    outs.append(norm_layer(gap).flatten(1))
                else:
                    # The output of LayerNorm2d may be discontiguous, which
                    # may cause some problem in the downstream tasks
                    outs.append(norm_layer(x).contiguous())

        return tuple(outs)

    def _freeze_stages(self):
        for i in range(self.frozen_stages):
            downsample_layer = self.downsample_layers[i]
            stage = self.stages[i]
            downsample_layer.eval()
            stage.eval()
            for param in chain(downsample_layer.parameters(),
                               stage.parameters()):
                param.requires_grad = False
    
    def _make_deep_stem_layer(self,in_channels,stem_channels,stem_patch_size=4,norm="LayerNorm2d",activation_fn="ReLU"):
        if self.deep_stem_mode == "default":
            stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                stem_channels,
                kernel_size=stem_patch_size,
                stride=stem_patch_size),
                wnn.get_norm(norm, stem_channels),
            )
            return stem
        elif self.deep_stem_mode == "MultiBranchStem24X":
            return MultiBranchStem24X(in_channels,stem_channels,activation_fn=activation_fn)
        elif self.deep_stem_mode == "MultiBranchStemS24X":
            return MultiBranchStemS24X(in_channels,stem_channels,activation_fn=activation_fn)
        elif self.deep_stem_mode == "MultiBranchStemSL24X":
            return MultiBranchStemSL24X(in_channels,stem_channels,activation_fn=activation_fn)
        else:
            print(f"Unknow deep stem model {self.deep_stem_mode}")

        return None

    def train(self, mode=True):
        super(WConvNeXt, self).train(mode)
        self._freeze_stages()
