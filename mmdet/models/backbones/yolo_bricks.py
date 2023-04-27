# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from wtorch.conv_module import ConvModule
from mmcv.runner import BaseModule
from torch import Tensor
from ..builder import MODELS
from mmcv.cnn import build_norm_layer
from wtorch.nn import get_activation

@MODELS.register_module()
class EffectiveSELayer(nn.Module):
    """Effective Squeeze-Excitation.

    From `CenterMask : Real-Time Anchor-Free Instance Segmentation`
    arxiv (https://arxiv.org/abs/1911.06667)
    This code referenced to
    https://github.com/youngwanLEE/CenterMask/blob/72147e8aae673fcaf4103ee90a6a6b73863e7fa1/maskrcnn_benchmark/modeling/backbone/vovnet.py#L108-L121  # noqa

    Args:
        channels (int): The input and output channels of this Module.
        act_cfg (dict): Config dict for activation layer.
            Defaults to dict(type='HSigmoid').
    """

    def __init__(self,
                 channels: int,
                 act_cfg = dict(type='HSigmoid')):
        super().__init__()
        assert isinstance(act_cfg, dict)
        self.fc = ConvModule(channels, channels, 1, act_cfg=None)

        act_cfg_ = act_cfg.copy()  # type: ignore
        self.activate = get_activation(act_cfg_['type'],inplace=act_cfg_.get('inplace',True))

    def forward(self, x: Tensor) -> Tensor:
        """Forward process
         Args:
             x (Tensor): The input tensor.
         """
        x_se = x.mean((2, 3), keepdim=True)
        x_se = self.fc(x_se)
        return x * self.activate(x_se)

@MODELS.register_module()
class RepVGGBlock(nn.Module):
    """RepVGGBlock is a basic rep-style block, including training and deploy
    status This code is based on
    https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple): Stride of the convolution. Default: 1
        padding (int, tuple): Padding added to all four sides of
            the input. Default: 1
        dilation (int or tuple): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input
            channels to output channels. Default: 1
        padding_mode (string, optional): Default: 'zeros'
        use_se (bool): Whether to use se. Default: False
        use_alpha (bool): Whether to use `alpha` parameter at 1x1 conv.
            In PPYOLOE+ model backbone, `use_alpha` will be set to True.
            Default: False.
        use_bn_first (bool): Whether to use bn layer before conv.
            In YOLOv6 and YOLOv7, this will be set to True.
            In PPYOLOE, this will be set to False.
            Default: True.
        deploy (bool): Whether in deploy mode. Default: False
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int]] = 3,
                 stride: Union[int, Tuple[int]] = 1,
                 padding: Union[int, Tuple[int]] = 1,
                 dilation: Union[int, Tuple[int]] = 1,
                 groups: Optional[int] = 1,
                 padding_mode: Optional[str] = 'zeros',
                 norm_cfg = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg = dict(type='ReLU', inplace=True),
                 use_se: bool = False,
                 use_alpha: bool = False,
                 use_bn_first=True,
                 deploy: bool = False):
        super().__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels

        assert kernel_size == 3
        assert padding == 1

        padding_11 = padding - kernel_size // 2

        self.nonlinearity = get_activation(act_cfg['type'],inplace=act_cfg.get('inplace',True))

        if use_se:
            raise NotImplementedError('se block not supported yet')
        else:
            self.se = nn.Identity()

        if use_alpha:
            alpha = torch.ones([
                1,
            ], dtype=torch.float32, requires_grad=True)
            self.alpha = nn.Parameter(alpha, requires_grad=True)
        else:
            self.alpha = None

        if deploy:
            self.rbr_reparam = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=True,
                padding_mode=padding_mode)

        else:
            if use_bn_first and (out_channels == in_channels) and stride == 1:
                self.rbr_identity = build_norm_layer(
                    norm_cfg, num_features=in_channels)[1]
            else:
                self.rbr_identity = None

            self.rbr_dense = ConvModule(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
                norm_cfg=norm_cfg,
                act_cfg=None)
            self.rbr_1x1 = ConvModule(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                padding=padding_11,
                groups=groups,
                bias=False,
                norm_cfg=norm_cfg,
                act_cfg=None)

    def forward(self, inputs: Tensor) -> Tensor:
        """Forward process.
        Args:
            inputs (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.
        """
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)
        if self.alpha:
            return self.nonlinearity(
                self.se(
                    self.rbr_dense(inputs) +
                    self.alpha * self.rbr_1x1(inputs) + id_out))
        else:
            return self.nonlinearity(
                self.se(
                    self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))

    def get_equivalent_kernel_bias(self):
        """Derives the equivalent kernel and bias in a differentiable way.

        Returns:
            tuple: Equivalent kernel and bias
        """
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        if self.alpha:
            return kernel3x3 + self.alpha * self._pad_1x1_to_3x3_tensor(
                kernel1x1) + kernelid, bias3x3 + self.alpha * bias1x1 + biasid
        else:
            return kernel3x3 + self._pad_1x1_to_3x3_tensor(
                kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        """Pad 1x1 tensor to 3x3.
        Args:
            kernel1x1 (Tensor): The input 1x1 kernel need to be padded.

        Returns:
            Tensor: 3x3 kernel after padded.
        """
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch: nn.Module) -> Tuple[np.ndarray, Tensor]:
        """Derives the equivalent kernel and bias of a specific branch layer.

        Args:
            branch (nn.Module): The layer that needs to be equivalently
                transformed, which can be nn.Sequential or nn.Batchnorm2d

        Returns:
            tuple: Equivalent kernel and bias
        """
        if branch is None:
            return 0, 0
        if isinstance(branch, ConvModule):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, (nn.SyncBatchNorm, nn.BatchNorm2d))
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3),
                                        dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(
                    branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        """Switch to deploy mode."""
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(
            in_channels=self.rbr_dense.conv.in_channels,
            out_channels=self.rbr_dense.conv.out_channels,
            kernel_size=self.rbr_dense.conv.kernel_size,
            stride=self.rbr_dense.conv.stride,
            padding=self.rbr_dense.conv.padding,
            dilation=self.rbr_dense.conv.dilation,
            groups=self.rbr_dense.conv.groups,
            bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.deploy = True

@MODELS.register_module()
class PPYOLOEBasicBlock(nn.Module):
    """PPYOLOE Backbone BasicBlock.

    Args:
         in_channels (int): The input channels of this Module.
         out_channels (int): The output channels of this Module.
         norm_cfg (dict): Config dict for normalization layer.
             Defaults to dict(type='BN', momentum=0.1, eps=1e-5).
         act_cfg (dict): Config dict for activation layer.
             Defaults to dict(type='SiLU', inplace=True).
         shortcut (bool): Whether to add inputs and outputs together
         at the end of this layer. Defaults to True.
         use_alpha (bool): Whether to use `alpha` parameter at 1x1 conv.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 norm_cfg= dict(
                     type='BN', momentum=0.1, eps=1e-5),
                 act_cfg= dict(type='SiLU', inplace=True),
                 shortcut: bool = True,
                 use_alpha: bool = False):
        super().__init__()
        assert act_cfg is None or isinstance(act_cfg, dict)
        self.conv1 = ConvModule(
            in_channels,
            out_channels,
            3,
            stride=1,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.conv2 = RepVGGBlock(
            out_channels,
            out_channels,
            use_alpha=use_alpha,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            use_bn_first=False)
        self.shortcut = shortcut

    def forward(self, x: Tensor) -> Tensor:
        """Forward process.
        Args:
            inputs (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.
        """
        y = self.conv1(x)
        y = self.conv2(y)
        if self.shortcut:
            return x + y
        else:
            return y


class SPPFBottleneck(BaseModule):
    """Spatial pyramid pooling - Fast (SPPF) layer for
    YOLOv5, YOLOX and PPYOLOE by Glenn Jocher

    Args:
        in_channels (int): The input channels of this Module.
        out_channels (int): The output channels of this Module.
        kernel_sizes (int, tuple[int]): Sequential or number of kernel
            sizes of pooling layers. Defaults to 5.
        use_conv_first (bool): Whether to use conv before pooling layer.
            In YOLOv5 and YOLOX, the para set to True.
            In PPYOLOE, the para set to False.
            Defaults to True.
        mid_channels_scale (float): Channel multiplier, multiply in_channels
            by this amount to get mid_channels. This parameter is valid only
            when use_conv_fist=True.Defaults to 0.5.
        conv_cfg (dict): Config dict for convolution layer. Defaults to None.
            which means using conv2d. Defaults to None.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to dict(type='BN', momentum=0.03, eps=0.001).
        act_cfg (dict): Config dict for activation layer.
            Defaults to dict(type='SiLU', inplace=True).
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_sizes: Union[int, Sequence[int]] = 5,
                 use_conv_first: bool = True,
                 mid_channels_scale: float = 0.5,
                 conv_cfg = None,
                 norm_cfg = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg = dict(type='SiLU', inplace=True),
                 init_cfg = None):
        super().__init__(init_cfg)

        if use_conv_first:
            mid_channels = int(in_channels * mid_channels_scale)
            self.conv1 = ConvModule(
                in_channels,
                mid_channels,
                1,
                stride=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
        else:
            mid_channels = in_channels
            self.conv1 = None
        self.kernel_sizes = kernel_sizes
        if isinstance(kernel_sizes, int):
            self.poolings = nn.MaxPool2d(
                kernel_size=kernel_sizes, stride=1, padding=kernel_sizes // 2)
            conv2_in_channels = mid_channels * 4
        else:
            self.poolings = nn.ModuleList([
                nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)
                for ks in kernel_sizes
            ])
            conv2_in_channels = mid_channels * (len(kernel_sizes) + 1)

        self.conv2 = ConvModule(
            conv2_in_channels,
            out_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self, x: Tensor) -> Tensor:
        """Forward process
        Args:
            x (Tensor): The input tensor.
        """
        if self.conv1:
            x = self.conv1(x)
        if isinstance(self.kernel_sizes, int):
            y1 = self.poolings(x)
            y2 = self.poolings(y1)
            x = torch.cat([x, y1, y2, self.poolings(y2)], dim=1)
        else:
            x = torch.cat(
                [x] + [pooling(x) for pooling in self.poolings], dim=1)
        x = self.conv2(x)
        return x


class CSPResLayer(nn.Module):
    """PPYOLOE Backbone Stage.

    Args:
        in_channels (int): The input channels of this Module.
        out_channels (int): The output channels of this Module.
        num_block (int): Number of blocks in this stage.
        block_cfg (dict): Config dict for block. Default config is
            suitable for PPYOLOE+ backbone. And in PPYOLOE neck,
            block_cfg is set to dict(type='PPYOLOEBasicBlock',
            shortcut=False, use_alpha=False). Defaults to
            dict(type='PPYOLOEBasicBlock', shortcut=True, use_alpha=True).
        stride (int): Stride of the convolution. In backbone, the stride
            must be set to 2. In neck, the stride must be set to 1.
            Defaults to 1.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to dict(type='BN', momentum=0.1, eps=1e-5).
        act_cfg (dict): Config dict for activation layer.
            Defaults to dict(type='SiLU', inplace=True).
        attention_cfg (dict, optional): Config dict for `EffectiveSELayer`.
            Defaults to dict(type='EffectiveSELayer',
            act_cfg=dict(type='HSigmoid')).
        use_spp (bool): Whether to use `SPPFBottleneck` layer.
            Defaults to False.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 num_block: int,
                 block_cfg = dict(
                     type='PPYOLOEBasicBlock', shortcut=True, use_alpha=True),
                 stride: int = 1,
                 norm_cfg = dict(
                     type='BN', momentum=0.1, eps=1e-5),
                 act_cfg = dict(type='SiLU', inplace=True),
                 attention_cfg = dict(
                     type='EffectiveSELayer', act_cfg=dict(type='HSigmoid')),
                 use_spp: bool = False):
        super().__init__()

        self.num_block = num_block
        self.block_cfg = block_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.use_spp = use_spp
        assert attention_cfg is None or isinstance(attention_cfg, dict)

        if stride == 2:
            conv1_in_channels = conv2_in_channels = conv3_in_channels = (
                in_channels + out_channels) // 2
            blocks_channels = conv1_in_channels // 2
            self.conv_down = ConvModule(
                in_channels,
                conv1_in_channels,
                3,
                stride=2,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
        else:
            conv1_in_channels = conv2_in_channels = in_channels
            conv3_in_channels = out_channels
            blocks_channels = out_channels // 2
            self.conv_down = None

        self.conv1 = ConvModule(
            conv1_in_channels,
            blocks_channels,
            1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.conv2 = ConvModule(
            conv2_in_channels,
            blocks_channels,
            1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.blocks = self.build_blocks_layer(blocks_channels)

        self.conv3 = ConvModule(
            conv3_in_channels,
            out_channels,
            1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        if attention_cfg:
            attention_cfg = attention_cfg.copy()
            attention_cfg['channels'] = blocks_channels * 2
            self.attn = MODELS.build(attention_cfg)
        else:
            self.attn = None

    def build_blocks_layer(self, blocks_channels: int) -> nn.Module:
        """Build blocks layer.

        Args:
            blocks_channels: The channels of this Module.
        """
        blocks = nn.Sequential()
        block_cfg = self.block_cfg.copy()
        block_cfg.update(
            dict(in_channels=blocks_channels, out_channels=blocks_channels))
        block_cfg.setdefault('norm_cfg', self.norm_cfg)
        block_cfg.setdefault('act_cfg', self.act_cfg)

        for i in range(self.num_block):
            blocks.add_module(str(i), MODELS.build(block_cfg))

            if i == (self.num_block - 1) // 2 and self.use_spp:
                blocks.add_module(
                    'spp',
                    SPPFBottleneck(
                        blocks_channels,
                        blocks_channels,
                        kernel_sizes=[5, 9, 13],
                        use_conv_first=False,
                        conv_cfg=None,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg))

        return blocks

    def forward(self, x: Tensor) -> Tensor:
        """Forward process
         Args:
             x (Tensor): The input tensor.
         """
        if self.conv_down is not None:
            x = self.conv_down(x)
        y1 = self.conv1(x)
        y2 = self.blocks(self.conv2(x))
        y = torch.cat([y1, y2], axis=1)
        if self.attn is not None:
            y = self.attn(y)
        y = self.conv3(y)
        return y