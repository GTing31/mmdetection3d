# Copyright (c) OpenMMLab. All rights reserved.
from itertools import chain
from typing import Optional, Sequence, Tuple
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmengine.model import BaseModule

from mmdet3d.registry import MODELS
from mmdet3d.utils import ConfigType, OptMultiConfig
from timm.models.layers import trunc_normal_, DropPath
from mmdet3d.registry import MODELS


class LayerNorm2d(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise ValueError("data_format must be 'channels_last' or 'channels_first'")
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, kernel_size=7, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)
        self.norm = LayerNorm2d(dim, eps=1e-6, data_format="channels_last")
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = shortcut + self.drop_path(x)
        return x

@MODELS.register_module()
class ConvNeXtPC(BaseModule):
    """适用于点云的 ConvNeXt 主干网络。"""

    arch_settings = {
        'lite': {
            'depths': [2, 2, 1, 1],
            'channels': [48, 96, 192, 192],
            'out_indices': [3]

        },

        'tiny': {
            'depths': [2, 2, 1, 1, 1],
            'channels': [48, 96, 96, 96, 96],
            'out_indices': [2, 3, 4]
        },
        'small': {
            'depths': [3, 3, 2, 1, 1],
            'channels': [48, 96, 192, 192, 192],
            'out_indices': [2, 3, 4]
        },
        'base': {
            'depths': [4, 4, 2, 2, 1],
            'channels': [64, 192, 384, 384, 384],
            'out_indices': [2, 3, 4]
        },
        'large': {
            'depths': [6, 6, 4, 2, 1],
            'channels': [96, 192, 384, 384, 384],
            'out_indices': [2, 3, 4]
        },
    }

    def __init__(self,
                 arch='tiny',
                 in_channels=3,
                 stem_patch_size=4,
                 norm_cfg=dict(type='LN', eps=1e-6),
                 act_cfg=dict(type='GELU'),
                 linear_pw_conv=True,
                 drop_path_rate=0.,
                 layer_scale_init_value=1e-6,
                 out_indices=-1,
                 frozen_stages=0,
                 gap_before_final_norm=True,
                 first_downsample=1,
                 large_arch=None,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        self.first_downsample = first_downsample


        # 处理架构设置
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
        out_indices = arch['out_indices']

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
        if self.first_downsample == 0:
            self.downsample_layers = nn.ModuleList()
        else:
            self.downsample_layers = nn.ModuleList([None])

        self.bias = torch.nn.Parameter(torch.randn(3))

        # 4 feature resolution stages, each consisting of multiple residual
        # blocks
        self.stages = nn.ModuleList()

        for i in range(self.num_stages):
            depth = self.depths[i]
            if i == 0:
                self.channels[i] = in_channels  # NOTE
            channels = self.channels[i]

            if i >= self.first_downsample:
                if self.first_downsample == 0 and i == 0:
                    downsample_layer = nn.Sequential(
                        LayerNorm2d(in_channels, data_format="channels_first"),
                        nn.Conv2d(
                            in_channels,
                            channels,
                            kernel_size=2,
                            stride=2),
                    )
                    self.downsample_layers.append(downsample_layer)
                else:
                    downsample_layer = nn.Sequential(
                        LayerNorm2d(self.channels[i - 1], data_format="channels_first"),
                        nn.Conv2d(
                            self.channels[i - 1],
                            channels,
                            kernel_size=2,
                            stride=2),
                    )
                    self.downsample_layers.append(downsample_layer)

            stage = nn.Sequential(*[
                ConvNeXtBlock(
                    dim=channels,
                    drop_path=dpr[block_idx + j],
                    layer_scale_init_value=layer_scale_init_value)
                for j in range(depth)
            ])

            block_idx += depth

            self.stages.append(stage)

            if i in self.out_indices:
                norm_layer = LayerNorm2d(self.channels[i], eps=1e-6, data_format="channels_first")
                self.add_module(f'norm{i}', norm_layer)



    def forward(self, x):

        start_time = time.time()
        # x.shape [4, 96, 1024, 1024]
        # print(f"Input shape: {x.shape}")
        outs = []
        print("out_indices", self.out_indices)
        for i, stage in enumerate(self.stages):
            if i >= self.first_downsample:
                x = self.downsample_layers[i](x)  # NOTE, pretrain_weight
                # print(f"After downsampling layer {i}, shape: {x.shape}")
            # print(stage)
            x = stage(x)


            # print(f"After stage {i}, shape: {x.shape}")
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                if self.gap_before_final_norm:
                    gap = x.mean([-2, -1], keepdim=True)
                    outs.append(norm_layer(gap).flatten(1))
                else:
                    # The output of LayerNorm2d may be discontiguous, which
                    # may cause some problem in the downstream tasks
                    outs.append(norm_layer(x).contiguous())
                print(f"Output at layer {i} shape: {outs[-1].shape}")

        end_time = time.time()
        elapsed_time = end_time - start_time
        # print(f"Backbone 花費：{elapsed_time:.6f} 秒")
        # print(outs[0].shape)
        return tuple(outs)

