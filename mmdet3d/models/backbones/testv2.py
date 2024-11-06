# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Optional, Sequence, Tuple

from mmcv.cnn import build_conv_layer, build_norm_layer
from mmengine.model import BaseModule
from torch import Tensor
from torch import nn as nn

from mmdet3d.registry import MODELS
from mmdet3d.utils import ConfigType, OptMultiConfig
from mmdet3d.models.layers import SparseBasicBlock, make_sparse_convmodule





@MODELS.register_module()
class testv2(BaseModule):
    """Backbone network for SECOND/PointPillars/PartA2/MVXNet.

    Args:
        in_channels (int): Input channels.
        out_channels (list[int]): Output channels for multi-scale feature maps.
        layer_nums (list[int]): Number of layers in each stage.
        layer_strides (list[int]): Strides of each stage.
        norm_cfg (dict): Config dict of normalization layers.
        conv_cfg (dict): Config dict of convolutional layers.
    """

    def __init__(self,
                 in_channels: int = 128,
                 out_channels: Sequence[int] = [128, 128, 256],
                 # layer_nums: Sequence[int] = [3, 5, 5],
                 layer_strides: Sequence[int] = [2, 2, 2],
                 norm_cfg: ConfigType = dict(
                     type='BN', eps=1e-3, momentum=0.01),
                 conv_cfg: ConfigType = dict(type='Conv2d', bias=False),
                 init_cfg: OptMultiConfig = None,
                 pretrained: Optional[str] = None) -> None:
        super(testv2, self).__init__(init_cfg=init_cfg)
        # print(f"Input channels: {in_channels}")# 64
        # print(f"Output channels: {out_channels[0]}")# [64, 128, 256]
        # assert len(layer_strides) == len(layer_nums)
        # assert len(out_channels) == len(layer_nums)

        in_filters = [in_channels, *out_channels[:-1]]
        print(in_filters[0])#[64, 64, 128]
        blocks = []
        # self.block1 = nn.Sequential(
        #     build_conv_layer(conv_cfg, in_channels, out_channels[0], 3, stride=layer_strides[0], padding=1),
        #     build_norm_layer(norm_cfg, out_channels[0])[1],
        #     nn.ReLU(inplace=True),
        #
        #     build_conv_layer(conv_cfg, in_channels, out_channels[0], 3, padding=1),
        #     build_norm_layer(norm_cfg, out_channels[0])[1],
        #     nn.ReLU(inplace=True),
        #
        #     build_conv_layer(conv_cfg, in_channels, out_channels[0], 3, padding=1),
        #     build_norm_layer(norm_cfg, out_channels[0])[1],
        #     nn.ReLU(inplace=True),
        #
        #     build_conv_layer(conv_cfg, in_channels, out_channels[0], 3, padding=1),
        #     build_norm_layer(norm_cfg, out_channels[0])[1],
        #     nn.ReLU(inplace=True),
        #


        self.block1 = nn.Sequential(
            SparseBasicBlock(in_channels, out_channels[0], stride=layer_strides[0], norm_cfg=norm_cfg,
                             conv_cfg='SparseConv2d' , indice_key="block1_1"),
            SparseBasicBlock(out_channels[0], out_channels[0], stride=layer_strides[0], norm_cfg=norm_cfg,
                             conv_cfg='SparseConv2d', indice_key="block1_2"),
            SparseBasicBlock(out_channels[0], out_channels[0], stride=layer_strides[0], norm_cfg=norm_cfg,
                             conv_cfg='SparseConv2d', indice_key="block1_3"),
        )


        self.block2 = nn.Sequential(


            build_conv_layer(conv_cfg, in_filters[0], out_channels[1], 3, stride=layer_strides[1], padding=1),
            build_norm_layer(norm_cfg, out_channels[1])[1],

            nn.ReLU(inplace=True),


            build_conv_layer(conv_cfg, out_channels[1], out_channels[1], 3, padding=1),
            build_norm_layer(norm_cfg, out_channels[1])[1],

            nn.ReLU(inplace=True),

            build_conv_layer(conv_cfg, out_channels[1], out_channels[1], 3, padding=1),
            build_norm_layer(norm_cfg, out_channels[1])[1],
            nn.ReLU(inplace=True),

            build_conv_layer(conv_cfg, out_channels[1], out_channels[1], 3, padding=1),
            build_norm_layer(norm_cfg, out_channels[1])[1],
            nn.ReLU(inplace=True),

            build_conv_layer(conv_cfg, out_channels[1], out_channels[1], 3, padding=1),
            build_norm_layer(norm_cfg, out_channels[1])[1],
            nn.ReLU(inplace=True),

            build_conv_layer(conv_cfg, out_channels[1], out_channels[1], 3, padding=1),
            build_norm_layer(norm_cfg, out_channels[1])[1],
            nn.ReLU(inplace=True),
        )

        self.block3 = nn.Sequential(
            build_conv_layer(conv_cfg, in_filters[2], out_channels[2], 3, stride=layer_strides[1], padding=1),
            build_norm_layer(norm_cfg, out_channels[2])[1],
            nn.ReLU(inplace=True),
            build_conv_layer(conv_cfg, out_channels[2], out_channels[2], 3, padding=1),
            build_norm_layer(norm_cfg, out_channels[2])[1],
            nn.ReLU(inplace=True),
            build_conv_layer(conv_cfg, out_channels[2], out_channels[2], 3, padding=1),
            build_norm_layer(norm_cfg, out_channels[2])[1],
            nn.ReLU(inplace=True),
            build_conv_layer(conv_cfg, out_channels[2], out_channels[2], 3, padding=1),
            build_norm_layer(norm_cfg, out_channels[2])[1],
            nn.ReLU(inplace=True),
            build_conv_layer(conv_cfg, out_channels[2], out_channels[2], 3, padding=1),
            build_norm_layer(norm_cfg, out_channels[2])[1],
            nn.ReLU(inplace=True),

            build_conv_layer(conv_cfg, out_channels[2], out_channels[2], 3, padding=1),
            build_norm_layer(norm_cfg, out_channels[2])[1],
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.ModuleList(blocks)
        # in_filters = [in_channels, *out_channels[:-1]]
        # # note that when stride > 1, conv2d with same padding isn't
        # # equal to pad-conv2d. we should use pad-conv2d.
        # blocks = []
        # for i, layer_num in enumerate(layer_nums):
        #     block = [
        #         build_conv_layer(
        #             conv_cfg,
        #             in_filters[i],
        #             out_channels[i],
        #             3,
        #             stride=layer_strides[i],
        #             padding=1),
        #         build_norm_layer(norm_cfg, out_channels[i])[1],
        #         nn.ReLU(inplace=True),
        #     ]
        #     for j in range(layer_num):
        #         block.append(
        #             build_conv_layer(
        #                 conv_cfg,
        #                 out_channels[i],
        #                 out_channels[i],
        #                 3,
        #                 padding=1))
        #         block.append(build_norm_layer(norm_cfg, out_channels[i])[1])
        #         block.append(nn.ReLU(inplace=True))
        #
        #     block = nn.Sequential(*block)
        #     blocks.append(block)
        #
        # self.blocks = nn.ModuleList(blocks)
        #
        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be setting at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is a deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        else:
            self.init_cfg = dict(type='Kaiming', layer='Conv2d')

    def forward(self, x: Tensor) -> Tuple[Tensor, ...]:
        """Forward function.

        Args:
            x (torch.Tensor): Input with shape (N, C, H, W).

        Returns:
            tuple[torch.Tensor]: Multi-scale features.
        """
        # outs = []
        # print(f"Input shape: {x.shape}")
        # for i in range(len(self.blocks)):
        #     x = self.blocks[i](x)
        #     print(f"After block {i + 1}, shape: {x.shape}")
        #     outs.append(x)
        # return tuple(outs)
        outs = []
        out1 = self.block1(x)
        # print(f"After block1: {out1.shape}")#After block1: torch.Size([12, 64, 248, 216])
        out2 = self.block2(out1)
        # print(self.block2)
        # print(f"After block2: {out2.shape}")
        out3 = self.block3(out2)
        # print(f"After block3: {out3.shape}")

        outs = [out1, out2, out3]
        return tuple(outs)