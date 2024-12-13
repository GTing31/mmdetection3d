
import torch
import torch.nn as nn

from mmcv.runner import BaseModule
from mmdet.models import BACKBONES
from timm.models.layers.helpers import to_2tuple
from timm.models.layers import trunc_normal_, DropPath


class InceptionDWConv2d(nn.Module):
    """ Inception depthweise convolution
    """

    def __init__(self, in_channels, square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125):
        super().__init__()

        gc = int(in_channels * branch_ratio)  # channel numbers of a convolution branch
        self.dwconv_hw = nn.Conv2d(gc, gc, square_kernel_size, padding=square_kernel_size // 2, groups=gc)
        self.dwconv_w = nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2),
                                  groups=gc)
        self.dwconv_h = nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0),
                                  groups=gc)
        self.split_indexes = (in_channels - 3 * gc, gc, gc, gc)

    def forward(self, x):
        x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
        return torch.cat((x_id, self.dwconv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h)), dim=1, )


class ConvMlp(nn.Module):
    """ MLP using 1x1 convs that keeps spatial dims
    copied from timm: https://github.com/huggingface/pytorch-image-models/blob/v0.6.11/timm/models/layers/mlp.py
    """

    def __init__(
            self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU,
            norm_layer=None, bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)

        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1, bias=bias[0])
        self.norm = norm_layer(hidden_features) if norm_layer else nn.Identity()
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1, bias=bias[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


class MetaNeXtBlock(nn.Module):
    """ MetaNeXtBlock Block
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        ls_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(
            self,
            dim,
            token_mixer=nn.Identity,
            norm_layer=nn.BatchNorm2d,
            mlp_layer=ConvMlp,
            mlp_ratio=4,
            act_layer=nn.ReLU,
            ls_init_value=1e-6,
            drop_path=0.,

    ):
        super().__init__()
        self.token_mixer = token_mixer(dim)
        self.norm = norm_layer(dim)
        self.mlp = mlp_layer(dim, int(mlp_ratio * dim), act_layer=act_layer)
        self.gamma = nn.Parameter(ls_init_value * torch.ones(dim)) if ls_init_value else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.token_mixer(x)
        x = self.norm(x)
        x = self.mlp(x)
        if self.gamma is not None:
            x = x.mul(self.gamma.reshape(1, -1, 1, 1))
        x = self.drop_path(x) + shortcut
        return x


class MetaNeXtStage(nn.Module):
    def __init__(
            self,
            in_chs,
            out_chs,
            ds_stride=2,
            depth=2,
            drop_path_rates=None,
            ls_init_value=1.0,
            token_mixer=nn.Identity,
            act_layer=nn.GELU,
            norm_layer=None,
            mlp_ratio=4,
    ):
        super().__init__()
        self.grad_checkpointing = False
        if ds_stride > 1:
            self.downsample = nn.Sequential(
                norm_layer(in_chs),
                nn.Conv2d(in_chs, out_chs, kernel_size=ds_stride, stride=ds_stride),
            )
        else:
            self.downsample = nn.Identity()

        drop_path_rates = drop_path_rates or [0.] * depth
        stage_blocks = []
        for i in range(depth):
            stage_blocks.append(MetaNeXtBlock(
                dim=out_chs,
                drop_path=drop_path_rates[i],
                ls_init_value=ls_init_value,
                token_mixer=token_mixer,
                act_layer=act_layer,
                norm_layer=norm_layer,
                mlp_ratio=mlp_ratio,
            ))
            in_chs = out_chs
        self.blocks = nn.Sequential(*stage_blocks)

    def forward(self, x):
        x = self.downsample(x)

        x = self.blocks(x)
        return x


@BACKBONES.register_module()
class InceptionNext(BaseModule):

    arch_settings = {
        'lite': {
            'depths': [3, 5, 5],
            'channels': [64, 128, 256],
            'out_indices': [0, 1, 2]
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
    }
    def __init__(self,
                 arch='tiny',
                 in_channels=3,
                 first_downsample=True,
                 token_mixers=InceptionDWConv2d,
                 norm_layer=nn.BatchNorm2d,
                 act_layer=nn.GELU,
                 mlp_ratios=4,
                 drop_rate=0.,
                 drop_path_rate=(0., 0., 0., 0., 0.),
                 init_values=1e-6,
                 **kwargs,
                 ):
        super().__init__()

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
        self.out_indices = arch['out_indices']
        self.first_downsample = first_downsample

        num_stage = len(self.depths)
        if not isinstance(token_mixers, (list, tuple)):
            token_mixers = [token_mixers] * num_stage
        if not isinstance(mlp_ratios, (list, tuple)):
            mlp_ratios = [mlp_ratios] * num_stage

        self.drop_rate = drop_rate


        self.stages = nn.Sequential()

        dp_rates = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(self.depths)).split(self.depths)]
        # print(f"dp_rates: {dp_rates}")
        stage = []
        prev_chs = in_channels
        for i in range(num_stage):
            if self.first_downsample:
                ds_stride = 2
            else:
                ds_stride = 2 if i > 0 else 1
            out_chs = self.channels[i]
            stage.append(
                MetaNeXtStage(
                    in_chs=prev_chs,
                    out_chs=out_chs,
                    ds_stride=ds_stride,
                    depth=self.depths[i],
                    drop_path_rates=dp_rates[i],
                    ls_init_value=init_values,
                    token_mixer=token_mixers[i],
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    mlp_ratio=mlp_ratios[i],
                )
            )
            prev_chs = out_chs
        self.stages = nn.Sequential(*stage)
        self.num_features = prev_chs

    def forward(self, x):
        outs = []
        # print(f"out_indices: {self.out_indices}")
        # print(f"input shape: {x.shape}")
        # print(self.stages)

        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i in self.out_indices:
                outs.append(x)
                print(f"stage {i} shape: {x.shape}")

        return tuple(outs)
