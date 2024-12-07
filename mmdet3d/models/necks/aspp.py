import torch
import torch.nn as nn
from mmdet3d.utils import ConvBlock, BasicBlock
import torch.utils.checkpoint as cp
import torch.nn.functional as F

from mmdet3d.registry import MODELS

@MODELS.register_module()
class ASPPNeck(nn.Module):
    def __init__(self, in_channels):

        super(ASPPNeck, self).__init__()

        self.pre_conv = BasicBlock(in_channels)
        self.conv1x1 = nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, bias=False, padding=0)
        self.weight = nn.Parameter(torch.randn(in_channels, in_channels, 3, 3))
        self.post_conv = ConvBlock(in_channels * 6, in_channels, kernel_size=1, stride=1)

    def _forward(self, x):

        # if isinstance(x, tuple):
        #     x = x[0]  # 提取 torch.Tensor
        #
        # # 確保 x 是 torch.Tensor
        # assert isinstance(x, torch.Tensor), "輸入 x 必須是 torch.Tensor！"

        x = self.pre_conv(x)
        # print("pre_conv", x.shape)
        branch1x1 = self.conv1x1(x)
        branch1 = F.conv2d(x, self.weight, stride=1,
                           bias=None, padding=1, dilation=1)
        branch6 = F.conv2d(x, self.weight, stride=1,
                           bias=None, padding=6, dilation=6)
        branch12 = F.conv2d(x, self.weight, stride=1,
                            bias=None, padding=12, dilation=12)
        branch18 = F.conv2d(x, self.weight, stride=1,
                            bias=None, padding=18, dilation=18)
        x = self.post_conv(
            torch.cat((x, branch1x1, branch1, branch6, branch12, branch18), dim=1))
        # print("after cat", x.shape)
        return x

    def forward(self, x):
        if isinstance(x, tuple):
            x = x[0]  # 提取 torch.Tensor
            print("x", x.shape)
        if x.requires_grad:
            out = cp.checkpoint(self._forward, x)
        else:
            out = self._forward(x)

        # print(out.shape)

        return [out]