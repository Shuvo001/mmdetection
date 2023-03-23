# Copyright (c) OpenMMLab. All rights reserved.
import warnings
import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import build_conv_layer, build_norm_layer, build_plugin_layer
from mmcv.runner import BaseModule
from torch.nn.modules.batchnorm import _BatchNorm
import wtorch.dist as wtd
import wtorch.nn as wnn

from ..builder import BACKBONES
from ..utils import ResLayer


class BasicBlock(BaseModule):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 dcn=None,
                 plugins=None,
                 init_cfg=None):
        super(BasicBlock, self).__init__(init_cfg)
        assert dcn is None, 'Not implemented yet.'
        assert plugins is None, 'Not implemented yet.'

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)

        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(
            conv_cfg, planes, planes, 3, padding=1, bias=False)
        self.add_module(self.norm2_name, norm2)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.with_cp = with_cp

    @property
    def norm1(self):
        """nn.Module: normalization layer after the first convolution layer"""
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: normalization layer after the second convolution layer"""
        return getattr(self, self.norm2_name)

    def forward(self, x):
        """Forward function."""

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.norm2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


class Bottleneck(BaseModule):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 dcn=None,
                 plugins=None,
                 init_cfg=None):
        """Bottleneck block for ResNet.

        If style is "pytorch", the stride-two layer is the 3x3 conv layer, if
        it is "caffe", the stride-two layer is the first 1x1 conv layer.
        """
        super(Bottleneck, self).__init__(init_cfg)
        assert style in ['pytorch', 'caffe']
        assert dcn is None or isinstance(dcn, dict)
        assert plugins is None or isinstance(plugins, list)
        if plugins is not None:
            allowed_position = ['after_conv1', 'after_conv2', 'after_conv3']
            assert all(p['position'] in allowed_position for p in plugins)

        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        self.dilation = dilation
        self.style = style
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.dcn = dcn
        self.with_dcn = dcn is not None
        self.plugins = plugins
        self.with_plugins = plugins is not None

        if self.with_plugins:
            # collect plugins for conv1/conv2/conv3
            self.after_conv1_plugins = [
                plugin['cfg'] for plugin in plugins
                if plugin['position'] == 'after_conv1'
            ]
            self.after_conv2_plugins = [
                plugin['cfg'] for plugin in plugins
                if plugin['position'] == 'after_conv2'
            ]
            self.after_conv3_plugins = [
                plugin['cfg'] for plugin in plugins
                if plugin['position'] == 'after_conv3'
            ]

        if self.style == 'pytorch':
            self.conv1_stride = 1
            self.conv2_stride = stride
        else:
            self.conv1_stride = stride
            self.conv2_stride = 1

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)
        self.norm3_name, norm3 = build_norm_layer(
            norm_cfg, planes * self.expansion, postfix=3)

        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            kernel_size=1,
            stride=self.conv1_stride,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        fallback_on_stride = False
        if self.with_dcn:
            fallback_on_stride = dcn.pop('fallback_on_stride', False)
        if not self.with_dcn or fallback_on_stride:
            self.conv2 = build_conv_layer(
                conv_cfg,
                planes,
                planes,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=dilation,
                dilation=dilation,
                bias=False)
        else:
            assert self.conv_cfg is None, 'conv_cfg must be None for DCN'
            self.conv2 = build_conv_layer(
                dcn,
                planes,
                planes,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=dilation,
                dilation=dilation,
                bias=False)

        self.add_module(self.norm2_name, norm2)
        self.conv3 = build_conv_layer(
            conv_cfg,
            planes,
            planes * self.expansion,
            kernel_size=1,
            bias=False)
        self.add_module(self.norm3_name, norm3)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

        if self.with_plugins:
            self.after_conv1_plugin_names = self.make_block_plugins(
                planes, self.after_conv1_plugins)
            self.after_conv2_plugin_names = self.make_block_plugins(
                planes, self.after_conv2_plugins)
            self.after_conv3_plugin_names = self.make_block_plugins(
                planes * self.expansion, self.after_conv3_plugins)

    def make_block_plugins(self, in_channels, plugins):
        """make plugins for block.

        Args:
            in_channels (int): Input channels of plugin.
            plugins (list[dict]): List of plugins cfg to build.

        Returns:
            list[str]: List of the names of plugin.
        """
        assert isinstance(plugins, list)
        plugin_names = []
        for plugin in plugins:
            plugin = plugin.copy()
            name, layer = build_plugin_layer(
                plugin,
                in_channels=in_channels,
                postfix=plugin.pop('postfix', ''))
            assert not hasattr(self, name), f'duplicate plugin {name}'
            self.add_module(name, layer)
            plugin_names.append(name)
        return plugin_names

    def forward_plugin(self, x, plugin_names):
        out = x
        for name in plugin_names:
            out = getattr(self, name)(out)
        return out

    @property
    def norm1(self):
        """nn.Module: normalization layer after the first convolution layer"""
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: normalization layer after the second convolution layer"""
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        """nn.Module: normalization layer after the third convolution layer"""
        return getattr(self, self.norm3_name)

    def forward(self, x):
        """Forward function."""

        def _inner_forward(x):
            identity = x
            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv1_plugin_names)

            out = self.conv2(out)
            out = self.norm2(out)
            out = self.relu(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv2_plugin_names)

            out = self.conv3(out)
            out = self.norm3(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv3_plugin_names)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out

class MultiBranchStem12X(nn.Module):
    def __init__(self,in_channels,out_channels,activation_fn="LeakyReLU"):
        super().__init__()
        self.out_channels = out_channels
        branch_channels = out_channels//4
        self.branch0 = nn.Conv2d(in_channels,branch_channels,3,stride=2,padding=1,bias=False)
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels,4,3,3,0,bias=False),
            nn.BatchNorm2d(num_features=4),
            wnn.get_activation(activation_fn,inplace=True),
            nn.Conv2d(4,8,3,2,1,bias=False),
            nn.BatchNorm2d(num_features=8),
            wnn.get_activation(activation_fn,inplace=True),
            nn.Conv2d(8,branch_channels,3,2,1,bias=False))
        self.branch2 = nn.Conv2d(in_channels,branch_channels,12,12,0,bias=False)
        self.branch3 = nn.Conv2d(in_channels,branch_channels,7,stride=2,padding=3,bias=False)
        self.norm = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            wnn.get_activation(activation_fn,inplace=True),
        )

    def forward(self,x):
        downsampled = torch.nn.functional.interpolate(x,(x.shape[-2]//6,x.shape[-1]//6),mode='bilinear')
        x0 = self.branch0(downsampled)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(downsampled)
        x = torch.cat([x0,x1,x2,x3],dim=1)
        x = self.norm(x)
        return x

class MultiBranchStemS12X(nn.Module):
    def __init__(self,in_channels,out_channels,activation_fn="LeakyReLU"):
        super().__init__()
        self.out_channels = out_channels
        branch_channels = out_channels//4
        self.branch0_0 = nn.Sequential(
            nn.Conv2d(in_channels,4,3,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(num_features=4),
            wnn.get_activation(activation_fn,inplace=True),
            nn.Conv2d(4,branch_channels,3,stride=1,padding=1,bias=False))
        self.branch0_1 = nn.ModuleList([nn.MaxPool2d(3,2,1),nn.MaxPool2d(5,2,2)])
        self.branch0_2 = nn.Conv2d(branch_channels*2,branch_channels*2,3,3,padding=0,bias=False)
        self.branch1 = nn.Conv2d(in_channels,branch_channels*2,7,stride=2,padding=3,bias=False)
        self.norm = nn.Sequential(
            nn.BatchNorm2d(out_channels),
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
        x = self.norm(x)
        return x

class MultiBranchStemS4X(nn.Module):
    def __init__(self,in_channels,out_channels,activation_fn="LeakyReLU"):
        super().__init__()
        self.out_channels = out_channels
        branch_channels = out_channels//4
        self.branch0 = nn.Sequential(
            nn.Conv2d(in_channels,4,3,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(num_features=4),
            wnn.get_activation(activation_fn,inplace=True),
            nn.Conv2d(4,branch_channels*2,3,stride=2,padding=1,bias=False))
        self.branch1 = nn.Conv2d(in_channels,branch_channels*2,7,stride=2,padding=3,bias=False)
        self.norm = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            wnn.get_activation(activation_fn,inplace=True),
        )

    def forward(self,x):
        downsampled = torch.nn.functional.interpolate(x,(x.shape[-2]//2,x.shape[-1]//2),mode='bilinear')
        x0 = self.branch0(x)
        x1 = self.branch1(downsampled)
        x = torch.cat([x0,x1],dim=1)
        x = self.norm(x)
        return x

class MultiBranchStemSL12X(nn.Module):
    def __init__(self,in_channels,out_channels,activation_fn="LeakyReLU"):
        super().__init__()
        self.out_channels = out_channels
        branch_channels = out_channels//4
        self.branch0 = nn.Sequential(
            nn.Conv2d(in_channels,4,3,stride=3,padding=0,bias=False),
            nn.BatchNorm2d(num_features=4),
            wnn.get_activation(activation_fn,inplace=True),
            nn.Conv2d(4,branch_channels,4,stride=4,padding=0,bias=False))
        self.branch1_0 = nn.Sequential(
            nn.Conv2d(in_channels,4,3,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(num_features=4),
            wnn.get_activation(activation_fn,inplace=True),
            nn.Conv2d(4,8,3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(num_features=8),
            wnn.get_activation(activation_fn,inplace=True),
            )
        self.branch1_1 = nn.ModuleList([nn.MaxPool2d(3,2,1),nn.MaxPool2d(5,2,2)])
        self.branch1_2 = nn.Conv2d(branch_channels,branch_channels,3,3,padding=0,bias=False)
        self.branch2 = nn.Conv2d(in_channels,branch_channels*2,7,stride=2,padding=3,bias=False)
        self.norm = nn.Sequential(
            nn.BatchNorm2d(out_channels),
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
        x = self.norm(x)
        return x

@BACKBONES.register_module()
class WResNet(BaseModule):
    """ResNet backbone.

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        stem_channels (int | None): Number of stem channels. If not specified,
            it will be the same as `base_channels`. Default: None.
        base_channels (int): Number of base channels of res layer. Default: 64.
        in_channels (int): Number of input image channels. Default: 3.
        num_stages (int): Resnet stages. Default: 4.
        strides (Sequence[int]): Strides of the first block of each stage.
        dilations (Sequence[int]): Dilation of each stage.
        out_indices (Sequence[int]): Output from which stages.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        deep_stem (bool): Replace 7x7 conv in input stem with 3 3x3 conv
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        norm_cfg (dict): Dictionary to construct and config norm layer.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.
        plugins (list[dict]): List of plugins for stages, each dict contains:

            - cfg (dict, required): Cfg dict to build plugin.
            - position (str, required): Position inside block to insert
              plugin, options are 'after_conv1', 'after_conv2', 'after_conv3'.
            - stages (tuple[bool], optional): Stages to apply plugin, length
              should be same as 'num_stages'.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        zero_init_residual (bool): Whether to use zero init for last norm layer
            in resblocks to let them behave as identity.
        pretrained (str, optional): model pretrained path. Default: None
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None

    Example:
        >>> from mmdet.models import ResNet
        >>> import torch
        >>> self = ResNet(depth=18)
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 32, 32)
        >>> level_outputs = self.forward(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        (1, 64, 8, 8)
        (1, 128, 4, 4)
        (1, 256, 2, 2)
        (1, 512, 1, 1)
    """

    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    '''
      kernel_size=self.first_conv_cfg['kernel_size'],
                stride=self.first_conv_cfg['stride'],
                padding=self.first_conv_cfg.get('padding',0),
    '''
    def __init__(self,
                 depth,
                 in_channels=3,
                 stem_channels=None,
                 base_channels=64,
                 num_stages=4,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=(0, 1, 2, 3),
                 style='pytorch',
                 deep_stem=False,
                 avg_down=False,
                 frozen_stages=-1,
                 conv_cfg=None,
                 first_conv_cfg={'kernel_size':7,'stride':2,'padding':3},
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=True,
                 dcn=None,
                 stage_with_dcn=(False, False, False, False),
                 plugins=None,
                 with_cp=False,
                 zero_init_residual=True,
                 pretrained=None,
                 deep_stem_mode="convs", #convs, MultiBranchStem12X, MultiBranchStemS12X, MultiBranchStemSL12X
                 stem_activation_fn="LeakyReLU", #ReLU,SiLU,LeakyReLU,SiLU
                 init_cfg=None):
        super(WResNet, self).__init__(init_cfg)
        self.zero_init_residual = zero_init_residual
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for resnet')

        block_init_cfg = None
        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be specified at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            if init_cfg is None:
                self.init_cfg = [
                    dict(type='Kaiming', layer='Conv2d'),
                    dict(
                        type='Constant',
                        val=1,
                        layer=['_BatchNorm', 'GroupNorm'])
                ]
                block = self.arch_settings[depth][0]
                if self.zero_init_residual:
                    if block is BasicBlock:
                        block_init_cfg = dict(
                            type='Constant',
                            val=0,
                            override=dict(name='norm2'))
                    elif block is Bottleneck:
                        block_init_cfg = dict(
                            type='Constant',
                            val=0,
                            override=dict(name='norm3'))
        else:
            raise TypeError('pretrained must be a str or None')

        self.deep_stem_mode = deep_stem_mode
        self.depth = depth
        if stem_channels is None:
            stem_channels = base_channels
        self.stem_channels = stem_channels
        self.base_channels = base_channels
        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 4
        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations) == num_stages
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.style = style
        self.deep_stem = deep_stem
        self.avg_down = avg_down
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.with_cp = with_cp
        self.norm_eval = norm_eval
        self.dcn = dcn
        self.stage_with_dcn = stage_with_dcn
        if dcn is not None:
            assert len(stage_with_dcn) == num_stages
        self.plugins = plugins
        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.inplanes = stem_channels
        self.first_conv_cfg = first_conv_cfg

        self._make_stem_layer(in_channels, stem_channels,activation_fn=stem_activation_fn)

        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            dcn = self.dcn if self.stage_with_dcn[i] else None
            if plugins is not None:
                stage_plugins = self.make_stage_plugins(plugins, i)
            else:
                stage_plugins = None
            planes = base_channels * 2**i
            res_layer = self.make_res_layer(
                block=self.block,
                inplanes=self.inplanes,
                planes=planes,
                num_blocks=num_blocks,
                stride=stride,
                dilation=dilation,
                style=self.style,
                avg_down=self.avg_down,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                dcn=dcn,
                plugins=stage_plugins,
                init_cfg=block_init_cfg)
            self.inplanes = planes * self.block.expansion
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self._freeze_stages()

        self.feat_dim = self.block.expansion * base_channels * 2**(
            len(self.stage_blocks) - 1)

    def make_stage_plugins(self, plugins, stage_idx):
        """Make plugins for ResNet ``stage_idx`` th stage.

        Currently we support to insert ``context_block``,
        ``empirical_attention_block``, ``nonlocal_block`` into the backbone
        like ResNet/ResNeXt. They could be inserted after conv1/conv2/conv3 of
        Bottleneck.

        An example of plugins format could be:

        Examples:
            >>> plugins=[
            ...     dict(cfg=dict(type='xxx', arg1='xxx'),
            ...          stages=(False, True, True, True),
            ...          position='after_conv2'),
            ...     dict(cfg=dict(type='yyy'),
            ...          stages=(True, True, True, True),
            ...          position='after_conv3'),
            ...     dict(cfg=dict(type='zzz', postfix='1'),
            ...          stages=(True, True, True, True),
            ...          position='after_conv3'),
            ...     dict(cfg=dict(type='zzz', postfix='2'),
            ...          stages=(True, True, True, True),
            ...          position='after_conv3')
            ... ]
            >>> self = ResNet(depth=18)
            >>> stage_plugins = self.make_stage_plugins(plugins, 0)
            >>> assert len(stage_plugins) == 3

        Suppose ``stage_idx=0``, the structure of blocks in the stage would be:

        .. code-block:: none

            conv1-> conv2->conv3->yyy->zzz1->zzz2

        Suppose 'stage_idx=1', the structure of blocks in the stage would be:

        .. code-block:: none

            conv1-> conv2->xxx->conv3->yyy->zzz1->zzz2

        If stages is missing, the plugin would be applied to all stages.

        Args:
            plugins (list[dict]): List of plugins cfg to build. The postfix is
                required if multiple same type plugins are inserted.
            stage_idx (int): Index of stage to build

        Returns:
            list[dict]: Plugins for current stage
        """
        stage_plugins = []
        for plugin in plugins:
            plugin = plugin.copy()
            stages = plugin.pop('stages', None)
            assert stages is None or len(stages) == self.num_stages
            # whether to insert plugin into current stage
            if stages is None or stages[stage_idx]:
                stage_plugins.append(plugin)

        return stage_plugins

    def make_res_layer(self, **kwargs):
        """Pack all blocks in a stage into a ``ResLayer``."""
        return ResLayer(**kwargs)

    @property
    def norm1(self):
        """nn.Module: the normalization layer named "norm1" """
        return getattr(self, self.norm1_name)

    def _make_deep_stem_layer(self,in_channels,stem_channels,activation_fn="ReLU"):
        if self.deep_stem_mode == "convs":
            moduls = []
            _in_channels = in_channels
            for cf in self.conv_cfg:
                _out_channels = cf['out_channels']
                moduls.append(nn.Conv2d(in_channels=_in_channels,
                out_channels=_out_channels,
                kernel_size=cf['kernel_size'],
                stride=cf.get('stride',1),
                padding=cf.get('padding',0),
                bias=cf.get('bias',False)
                ))
                moduls.append(wnn.get_activation(inplace=True))
                moduls.append(build_norm_layer(self.norm_cfg, _out_channels)[1])
                _in_channels = _out_channels
            return nn.Sequential(**moduls)
        elif self.deep_stem_mode == "MultiBranchStem12X":
            return MultiBranchStem12X(in_channels,stem_channels,activation_fn=activation_fn)
        elif self.deep_stem_mode == "MultiBranchStemS12X":
            return MultiBranchStemS12X(in_channels,stem_channels,activation_fn=activation_fn)
        elif self.deep_stem_mode == "MultiBranchStemS4X":
            return MultiBranchStemS4X(in_channels,stem_channels,activation_fn=activation_fn)
        elif self.deep_stem_mode == "MultiBranchStemSL12X":
            return MultiBranchStemSL12X(in_channels,stem_channels,activation_fn=activation_fn)
        else:
            print(f"Unknow deep stem model {self.deep_stem_mode}")

        return None

    def _make_stem_layer(self, in_channels, stem_channels,activation_fn="ReLU"):
        if self.deep_stem:
            self.stem = self._make_deep_stem_layer(in_channels,stem_channels,activation_fn=activation_fn)
        else:
            self.conv1 = build_conv_layer(
                self.conv_cfg,
                in_channels,
                stem_channels,
                kernel_size=self.first_conv_cfg['kernel_size'],
                stride=self.first_conv_cfg['stride'],
                padding=self.first_conv_cfg.get('padding',0),
                bias=False)
            self.norm1_name, norm1 = build_norm_layer(
                self.norm_cfg, stem_channels, postfix=1)
            self.add_module(self.norm1_name, norm1)
            self.relu = wnn.get_activation(activation_fn,inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            if self.deep_stem:
                self.stem.eval()
                for param in self.stem.parameters():
                    param.requires_grad = False
            else:
                self.norm1.eval()
                for m in [self.conv1, self.norm1]:
                    for param in m.parameters():
                        param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def forward(self, x):
        """Forward function."""
        if self.deep_stem:
            x = self.stem(x)
        else:
            try:
                x = self.conv1(x)
            except:
                print(f"ERROR:{x.shape}, {self.conv1.weight.shape}, {self.conv1.weight.device}, {x.device}, {wtd.get_rank()}")
            x = self.norm1(x)
            x = self.relu(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super(WResNet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()

