import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import build_norm_layer, build_conv_layer
from mmcv.runner import BaseModule, auto_fp16

from mmdet3d.models.builder import NECKS


class DepthwiseConvBlock(nn.Module):
    """Depthwise Separable Convolution Block: Depthwise + Pointwise + BN + ReLU."""

    def __init__(self,
                 in_channels,
                 out_channels,
                 norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01)):
        super(DepthwiseConvBlock, self).__init__()
        # depthwise
        self.depthwise = build_conv_layer(
            dict(type='Conv2d', bias=False),
            in_channels,
            in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=in_channels
        )
        # pointwise
        self.pointwise = build_conv_layer(
            dict(type='Conv2d', bias=False),
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.bn = build_norm_layer(norm_cfg, out_channels)[1]
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)


class ConvBlock(nn.Module):
    """Simple Conv + BN + ReLU block."""

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01)):
        super(ConvBlock, self).__init__()
        self.conv = build_conv_layer(
            dict(type='Conv2d', bias=False),
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        self.bn = build_norm_layer(norm_cfg, out_channels)[1]
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.act(x)


class BiFPNBlock(nn.Module):
    """
    實作一層BiFPN融合:
    這裡假設輸入為五個層級特徵: P3, P4, P5, P6, P7
    （若你只有三層C3, C4, C5，則需要自行增減P6,P7的生成）
    """

    def __init__(self, feature_size=64, epsilon=0.0001, norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01)):
        super(BiFPNBlock, self).__init__()
        self.epsilon = epsilon

        # Top-down depthwise conv
        self.p3_td = DepthwiseConvBlock(feature_size, feature_size, norm_cfg)
        self.p4_td = DepthwiseConvBlock(feature_size, feature_size, norm_cfg)


        # Bottom-up depthwise conv
        self.p4_out = DepthwiseConvBlock(feature_size, feature_size, norm_cfg)
        self.p5_out = DepthwiseConvBlock(feature_size, feature_size, norm_cfg)


        # 可學習權重
        # w1用來做top-down fusion, 假設有4個fusion點，各有2個輸入權重
        self.w1 = nn.Parameter(torch.ones(2, 4))
        # w2用來做bottom-up fusion, 假設有4個fusion點，各有3個輸入權重
        self.w2 = nn.Parameter(torch.ones(3, 4))

        self.w1_relu = nn.ReLU()
        self.w2_relu = nn.ReLU()

    def forward(self, inputs):
        # inputs假設為 [P3, P4, P5, P6, P7]
        p3_x, p4_x, p5_x = inputs

        # Normalize weights for top-down
        w1 = self.w1_relu(self.w1)
        w1 = w1 / (torch.sum(w1, dim=0, keepdim=True) + self.epsilon)

        # Top-down pathway
        # p7_td = p7_x (最高層不需要融合)
        p5_td = p5_x
        p4_td = self.p4_td(w1[0, 2] * p4_x + w1[1, 2] * F.interpolate(p5_td, scale_factor=2))
        p3_td = self.p3_td(w1[0, 3] * p3_x + w1[1, 3] * F.interpolate(p4_td, scale_factor=2))

        # Normalize weights for bottom-up
        w2 = self.w2_relu(self.w2)
        w2 = w2 / (torch.sum(w2, dim=0, keepdim=True) + self.epsilon)

        # Bottom-up pathway
        # p3_out = p3_td (最低層不需要融合)
        p3_out = p3_td
        p4_out = self.p4_out(w2[0, 0] * p4_x + w2[1, 0] * p4_td + w2[2, 0] * F.interpolate(p3_out, scale_factor=0.5))
        p5_out = self.p5_out(w2[0, 1] * p5_x + w2[1, 1] * p5_td + w2[2, 1] * F.interpolate(p4_out, scale_factor=0.5))


        return [p3_out, p4_out, p5_out]
class AutoAdjustBiFPNOutput(nn.Module):
    def __init__(self, align_layer=1):  # align_layer 定義基準層索引
        super(AutoAdjustBiFPNOutput, self).__init__()
        self.align_layer = align_layer  # 默認以 P4 為基準層（索引為 1）

    def forward(self, features):
        # 自動選擇基準層大小
        target_size = features[self.align_layer].shape[2:]  # 獲取對齊層的高和寬

        # 將其他層對齊到 target_size
        adjusted_features = [
            F.interpolate(f, size=target_size, mode='bilinear', align_corners=False) for f in features
        ]

        # 拼接特徵
        fused_features = torch.cat(adjusted_features, dim=1)
        return fused_features

@NECKS.register_module()
class BiFPN(BaseModule):
    """
    整合BiFPN作為mmdet3d的neck。
    假設輸入為[C3, C4, C5]三層backbone特徵，
    並透過下採樣產生P6, P7。
    最終輸出為[P3_out, P4_out, P5_out, P6_out, P7_out]五層特徵。
    你可以依需求調整輸入輸出層數。
    """

    def __init__(self,
                 in_channels=[64, 128, 256],  # 根據你的Backbone輸出調整
                 out_channels=128,
                 num_outs=5,
                 num_layers=2,
                 norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                 init_cfg=None):
        super(BiFPN, self).__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_outs = num_outs
        self.num_layers = num_layers
        assert len(in_channels) == 3, "此示例假設只有C3, C4, C5三層輸入"

        # 將C3, C4, C5透過1x1 conv調整channel
        self.p3_conv = ConvBlock(in_channels[0], out_channels, kernel_size=1, stride=1, padding=0, norm_cfg=norm_cfg)
        self.p4_conv = ConvBlock(in_channels[1], out_channels, kernel_size=1, stride=1, padding=0, norm_cfg=norm_cfg)
        self.p5_conv = ConvBlock(in_channels[2], out_channels, kernel_size=1, stride=1, padding=0, norm_cfg=norm_cfg)



        # 建立多層BiFPN Block
        self.bifpn_blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.bifpn_blocks.append(BiFPNBlock(feature_size=out_channels, epsilon=1e-4, norm_cfg=norm_cfg))

    @auto_fp16()
    def forward(self, inputs):
        """
        inputs: [C3, C4, C5]
        """
        C3, C4, C5 = inputs

        P3 = self.p3_conv(C3)
        P4 = self.p4_conv(C4)
        P5 = self.p5_conv(C5)
        # print(f"P3:{P3.shape}")
        # print(f"P4:{P4.shape}")
        # print(f"P5:{P5.shape}")

        features = [P3, P4, P5]

        # 通過多層的BiFPN
        for bifpn in self.bifpn_blocks:
            features = bifpn(features)

        auto_adjust = AutoAdjustBiFPNOutput(align_layer=1)  # 使用 P4 的分辨率為基準
        fused_features = auto_adjust(features)


        # features: [P3_out, P4_out, P5_out, P6_out, P7_out]
        # print(f"BiFPN output shapes after adjustment: {fused_features.shape}")
        # out = features[1]
        # print(f"BiFPN output shapes after adjustment: {out.shape}")
        return [fused_features]
