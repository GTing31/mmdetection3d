# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
from mmcv.cnn import build_conv_layer, build_norm_layer, build_upsample_layer
from mmengine.model import BaseModule
from torch import nn as nn

from mmdet3d.registry import MODELS


@MODELS.register_module()
class SECONDFPN(BaseModule):
    """FPN used in SECOND/PointPillars/PartA2/MVXNet.

    Args:
        in_channels (list[int]): Input channels of multi-scale feature maps.
        out_channels (list[int]): Output channels of feature maps.
        upsample_strides (list[int]): Strides used to upsample the
            feature maps.
        norm_cfg (dict): Config dict of normalization layers.
        upsample_cfg (dict): Config dict of upsample layers.
        conv_cfg (dict): Config dict of conv layers.
        use_conv_for_no_stride (bool): Whether to use conv when stride is 1.
        init_cfg (dict or :obj:`ConfigDict` or list[dict or :obj:`ConfigDict`],
            optional): Initialization config dict. Defaults to
            [dict(type='Kaiming', layer='ConvTranspose2d'),
             dict(type='Constant', layer='NaiveSyncBatchNorm2d', val=1.0)].
    """

    def __init__(self,
                 in_channels=[128, 128, 256],
                 out_channels=[256, 256, 256],
                 upsample_strides=[1, 2, 4],
                 norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                 upsample_cfg=dict(type='deconv', bias=False),
                 conv_cfg=dict(type='Conv2d', bias=False),
                 use_conv_for_no_stride=False,
                 init_cfg=[
                     dict(type='Kaiming', layer='ConvTranspose2d'),
                     dict(
                         type='Constant',
                         layer='NaiveSyncBatchNorm2d',
                         val=1.0)
                 ]):
        # if for GroupNorm,
        # cfg is dict(type='GN', num_groups=num_groups, eps=1e-3, affine=True)
        super(SECONDFPN, self).__init__(init_cfg=init_cfg)
        assert len(out_channels) == len(upsample_strides) == len(in_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels

        deblocks = []
        for i, out_channel in enumerate(out_channels):
            stride = upsample_strides[i]
            if stride > 1 or (stride == 1 and not use_conv_for_no_stride):
                upsample_layer = build_upsample_layer(
                    upsample_cfg,
                    in_channels=in_channels[i],
                    out_channels=out_channel,
                    kernel_size=upsample_strides[i],
                    stride=upsample_strides[i])
            else:
                stride = np.round(1 / stride).astype(np.int64)
                upsample_layer = build_conv_layer(
                    conv_cfg,
                    in_channels=in_channels[i],
                    out_channels=out_channel,
                    kernel_size=stride,
                    stride=stride)

            deblock = nn.Sequential(upsample_layer,
                                    build_norm_layer(norm_cfg, out_channel)[1],
                                    nn.ReLU(inplace=True))
            deblocks.append(deblock)
        self.deblocks = nn.ModuleList(deblocks)

    def forward(self, x):
        """Forward function.

        Args:
            x (List[torch.Tensor]): Multi-level features with 4D Tensor in
                (N, C, H, W) shape.

        Returns:
            list[torch.Tensor]: Multi-level feature maps.
        """
        assert len(x) == len(self.in_channels)
        ups = [deblock(x[i]) for i, deblock in enumerate(self.deblocks)]

        if len(ups) > 1:
            out = torch.cat(ups, dim=1)
        else:
            out = ups[0]
        return [out]

@MODELS.register_module()
class SECONDFPNv1(BaseModule):
    """FPN used in SECOND/PointPillars/PartA2/MVXNet.

    Args:
        in_channels (list[int]): Input channels of multi-scale feature maps.
        out_channels (list[int]): Output channels of feature maps.
        upsample_strides (list[int]): Strides used to upsample the
            feature maps.
        norm_cfg (dict): Config dict of normalization layers.
        upsample_cfg (dict): Config dict of upsample layers.
        conv_cfg (dict): Config dict of conv layers.
        use_conv_for_no_stride (bool): Whether to use conv when stride is 1.
        init_cfg (dict or :obj:`ConfigDict` or list[dict or :obj:`ConfigDict`],
            optional): Initialization config dict. Defaults to
            [dict(type='Kaiming', layer='ConvTranspose2d'),
             dict(type='Constant', layer='NaiveSyncBatchNorm2d', val=1.0)].
    """

    def __init__(self,
                 in_channels=[512, 512],
                 out_channels=[256, 256],
                 upsample_strides=[1, 2],
                 layer_nums=[5, 5],
                 norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                 upsample_cfg=dict(type='deconv', bias=False),
                 conv_cfg=dict(type='Conv2d', bias=False),
                 use_conv_for_no_stride=False,
                 init_cfg=[
                     dict(type='Kaiming', layer='ConvTranspose2d'),
                     dict(
                         type='Constant',
                         layer='NaiveSyncBatchNorm2d',
                         val=1.0)
                 ]):
        # if for GroupNorm,
        # cfg is dict(type='GN', num_groups=num_groups, eps=1e-3, affine=True)
        super(SECONDFPNv1, self).__init__(init_cfg=init_cfg)
        assert len(out_channels) == len(upsample_strides) == len(in_channels)# 這是一個assert語句，用來確保括號中的表達式為真，否則拋出AssertionError異常
        self.in_channels = in_channels
        self.out_channels = out_channels

        num_levels = len(layer_nums)
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    in_channels[idx], in_channels[idx], kernel_size=3,
                    stride=1, padding=0, bias=False
                ),
                nn.BatchNorm2d(in_channels[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2d(in_channels[idx], in_channels[idx], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(in_channels[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ])
            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            in_channels[idx], out_channels[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(out_channels[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            in_channels[idx], out_channels[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(in_channels[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))

        c_in = sum(out_channels)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.num_bev_features = c_in

    def forward(self, pillar_features):
        """Forward function.

        Args:
            x (List[torch.Tensor]): Multi-level features with 4D Tensor in
                (N, C, H, W) shape.

        Returns:
            list[torch.Tensor]: Multi-level feature maps.
        """
        x_conv4 = pillar_features['conv4']
        x_conv5 = pillar_features['conv5']


        ups = [self.deblocks[0](x_conv4)]

        x = self.blocks[1](x_conv5)
        ups.append(self.deblocks[1](x))

        x = torch.cat(ups, dim=1)
        x = self.blocks[0](x)

        # print(f"fpn1成功,features.shape: {x.shape},spatial shape: {x.shape},tensor type: {x.dtype}")

        return tuple([x])

        # assert len(x) == len(self.in_channels)
        # ups = [deblock(x[i]) for i, deblock in enumerate(self.deblocks)]
        #
        # out = torch.cat(ups, dim=1)
        #
        # # out = self.conv_fuse(out)
        # # print(f"fpn2成功,features.shape: {out.shape},spatial shape: {out.shape},tensor type: {out.dtype}")
        # return [out]
