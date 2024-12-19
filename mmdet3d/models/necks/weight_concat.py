import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleFusionWithChannels(nn.Module):
    def __init__(self, in_channels):
        super(MultiScaleFusionWithChannels, self).__init__()
        # 假設最終想要回到in_channels大小的特徵維度
        # 並假設有3個尺度特徵 concat 後為 3*in_channels
        self.fusion_conv = nn.Conv2d(in_channels * 3, in_channels, kernel_size=1, bias=False)

        # 加一個注意力 (SE Block)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # f1: (N, C, H/4, W/4)
        # f2: (N, C, H/8, W/8)
        # f3: (N, C, H/16, W/16)

        f1, f2, f3 = x[0], x[1], x[2]
        _, C, H, W = f1.shape

        # 上採樣f2, f3到f1大小
        f2_up = F.interpolate(f2, size=(H, W), mode='bilinear', align_corners=False)
        f3_up = F.interpolate(f3, size=(H, W), mode='bilinear', align_corners=False)

        # 沿channel方向concat -> (N, 3C, H, W)
        fused = torch.cat([f1, f2_up, f3_up], dim=1)

        # 1x1 conv 將 3C 壓回 C，並透過參數學習相當於對原本的特徵進行加權組合
        fused = self.fusion_conv(fused)

        # 加入SE注意力，動態調整各通道的權重
        att = self.se(fused)
        fused = fused * att

        return fused


# Test

