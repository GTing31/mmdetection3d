# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Optional, Sequence, Tuple

from functools import partial
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmengine.model import BaseModule
from torch import Tensor
from torch import nn as nn

from mmdet3d.registry import MODELS
from mmdet3d.utils import ConfigType, OptMultiConfig
from mmdet3d.models.layers import SparseBasicBlock, make_sparse_convmodule
from mmdet3d.models.layers.spconv import IS_SPCONV2_AVAILABLE

from .base import Sparse2DBasicBlock, Sparse2DBasicBlockV, post_act_block_dense, post_act_block

if IS_SPCONV2_AVAILABLE:
    from spconv.pytorch import SparseConvTensor, SparseModule, SparseSequential, SparseConv2d, SubMConv2d, SparseInverseConv2d, SparseReLU
else:
    from mmcv.ops import SparseConvTensor, SparseModule, SparseSequential


# def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
#                    conv_type='subm', norm_fn=None):
#
#     if conv_type == 'subm':
#         conv = SubMConv2d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
#     elif conv_type == 'spconv':
#         conv = SparseConv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
#                             bias=False, indice_key=indice_key)
#     elif conv_type == 'inverseconv':
#         conv = SparseInverseConv2d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
#     else:
#         raise NotImplementedError
#
#     SS=SparseSequential(
#         conv,
#         norm_fn(out_channels),
#         nn.ReLU(inplace=True),
#     )
#
#     return SS
#
# def post_act_block_dense(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, norm_fn=None):
#     m = nn.Sequential(
#         nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, dilation=dilation, bias=False),
#         norm_fn(out_channels),
#         nn.ReLU(inplace=True),
#     )
#
#     return m

@MODELS.register_module()
class testv1(BaseModule):
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
                 in_channels: int = 64,
                 norm_cfg: ConfigType = dict(
                     type='BN1d', eps=1e-3, momentum=0.01),
                 conv_cfg: ConfigType = dict(type='Conv2d'),
                 init_cfg: OptMultiConfig = None,
                 pretrained: Optional[str] = None) -> None:
        super(testv1, self).__init__(init_cfg=init_cfg)
        # print(f"Input channels: {in_channels}")# 64
        # print(f"Output channels: {out_channels[0]}")# [64, 128, 256]
        # assert len(layer_strides) == len(layer_nums)
        # assert len(out_channels) == len(layer_nums)


        block = post_act_block
        dense_block = post_act_block_dense

        self.conv1 = SparseSequential(
            Sparse2DBasicBlockV(in_channels, norm_cfg=norm_cfg, indice_key="res1"),
            Sparse2DBasicBlock(in_channels, norm_cfg=norm_cfg, indice_key="res1"),
        )

        self.conv2 = SparseSequential(
            SparseConv2d(in_channels, in_channels * 2, 3, 2, padding=1, bias=False),  # [752, 752] -> [376, 376]
            build_norm_layer(norm_cfg, in_channels * 2)[1],
            SparseReLU(),
            Sparse2DBasicBlock(in_channels * 2, norm_cfg=norm_cfg, indice_key="res2"),
            Sparse2DBasicBlock(in_channels * 2, norm_cfg=norm_cfg, indice_key="res2"),
        )

        self.conv3 = SparseSequential(
            SparseConv2d(in_channels * 2, in_channels * 4, 3, 2, padding=1, bias=False),  # [376, 376] -> [188, 188]
            build_norm_layer(norm_cfg, in_channels * 4)[1],
            SparseReLU(),
            Sparse2DBasicBlock(in_channels * 4, norm_cfg=norm_cfg, indice_key="res3"),
            Sparse2DBasicBlock(in_channels * 4, norm_cfg=norm_cfg, indice_key="res3"),
        )

        self.conv4 = SparseSequential(
            SparseConv2d(in_channels * 4, in_channels * 8, 3, 2, padding=1, bias=False),
            build_norm_layer(norm_cfg, in_channels * 8)[1],
            SparseReLU(),
            Sparse2DBasicBlock(in_channels * 8, norm_cfg=norm_cfg, indice_key="res4"),
            Sparse2DBasicBlock(in_channels * 8, norm_cfg=norm_cfg, indice_key="res4"),
        )

        norm_cfg = dict(type="BN", momentum=0.01, eps=1e-3)
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 2, padding=1, bias=False),
            build_norm_layer(norm_cfg, 512)[1],
            nn.ReLU(),
            dense_block(512, 3, padding=1, norm_cfg=norm_cfg),
            dense_block(512, 3, padding=1, norm_cfg=norm_cfg),
        )

        self.backbone_channels = {
            'conv1': 32,
            'conv2': 64,
            'conv3': 128,
            'conv4': 256,
        }
        self.backbone_strides = {
            'conv1': 1,
            'conv2': 2,
            'conv3': 4,
            'conv4': 8,
        }


        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be setting at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is a deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        else:
            self.init_cfg = dict(type='Kaiming', layer='Conv2d')

    def forward(self, x: SparseConvTensor) -> Tuple[SparseConvTensor, ...]:
        """Forward function.

        Args:
            x (torch.Tensor): Input with shape (N, C, H, W).

        Returns:
            tuple[torch.Tensor]: Multi-scale features.
        """


        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        x_conv4_dense = x_conv4.dense()
        x_conv5 = self.conv5(x_conv4_dense)

        backbone_features = {
            'conv1': x_conv1,
            'conv2': x_conv2,
            'conv3': x_conv3,
            'conv4': x_conv4_dense,
            'conv5': x_conv5,
        }


        # print(f"After block1: block1成功,features.shape: {x_conv1.features.shape},spatial shape: {x_conv1.spatial_shape}")#After block1: torch.Size([12, 64, 248, 216])
        #
        #
        # print(f"After block2: block2成功,features.shape: {x_conv2.features.shape},spatial shape: {x_conv2.spatial_shape}")
        #
        #
        # print(f"After block3: block3成功,features.shape: {x_conv3.features.shape},spatial shape: {x_conv3.spatial_shape}")
        #
        #
        # print(f"After block4: block4成功,features.shape: {x_conv4.features.shape},spatial shape: {x_conv4.spatial_shape}")
        #
        #
        # print(f"After block4: block4成功,features.shape: {x_conv4_dense.shape},spatial shape: {x_conv4_dense.shape},tensor type: {x_conv4_dense.dtype}")
        #
        #
        # print(f"After block5: block5成功,features.shape: {x_conv5.shape},spatial shape: {x_conv5.shape},tensor type: {x_conv5.dtype}")

        # outs = [x_conv4_dense, x_conv5]
        # return tuple(outs)

        return backbone_features