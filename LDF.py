import torch
import torch.nn as nn
import torch.nn.functional as F


class LightweightDepthEstimation(nn.Module):
    """
    轻量级深度评估模块
    利用RGB特征生成伪深度特征，使用多尺度感受野和注意力机制
    """

    def __init__(self, in_channels, out_channels=None, reduction=4):
        super(LightweightDepthEstimation, self).__init__()

        if out_channels is None:
            out_channels = in_channels

        mid_channels = max(in_channels // reduction, 16)

        # 多尺度深度卷积分支 - 捕获不同尺度的深度线索
        self.depth_conv_1x1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )

        self.depth_conv_3x3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, mid_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )

        self.depth_conv_5x5 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 5, 1, 2, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, mid_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )

        # 空洞卷积分支 - 捕获全局上下文
        self.depth_conv_dilated = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 2, dilation=2, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, mid_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )

        # 通道注意力模块 - 自适应调整不同分支的权重
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(mid_channels * 4, mid_channels * 4 // 4, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels * 4 // 4, mid_channels * 4, 1, 1, 0),
            nn.Sigmoid()
        )

        # 融合层
        self.fusion = nn.Sequential(
            nn.Conv2d(mid_channels * 4, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # 深度精炼模块 - 使用分组卷积
        refine_groups = min(out_channels, 32)  # 限制分组数，避免通道数太小
        while out_channels % refine_groups != 0:
            refine_groups -= 1

        self.refine = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, groups=refine_groups, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        # 残差连接的投影层
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels)
        ) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        """
        Args:
            x: 输入特征 [B, C, H, W]
        Returns:
            depth_feature: 伪深度特征 [B, C_out, H, W]
        """
        # 多尺度特征提取
        feat_1x1 = self.depth_conv_1x1(x)
        feat_3x3 = self.depth_conv_3x3(x)
        feat_5x5 = self.depth_conv_5x5(x)
        feat_dilated = self.depth_conv_dilated(x)

        # 拼接所有分支
        multi_scale_feat = torch.cat([feat_1x1, feat_3x3, feat_5x5, feat_dilated], dim=1)

        # 通道注意力
        attention = self.channel_attention(multi_scale_feat)
        multi_scale_feat = multi_scale_feat * attention

        # 融合
        depth_feature = self.fusion(multi_scale_feat)

        # 精炼
        depth_feature = self.refine(depth_feature)

        # 残差连接
        identity = self.shortcut(x)
        depth_feature = depth_feature + identity

        return depth_feature


class LDF(LightweightDepthEstimation):
    """LDF是LightweightDepthEstimation的简短别名"""
    pass


# 测试代码
if __name__ == '__main__':
    # 测试不同尺度的特征
    model = LightweightDepthEstimation(in_channels=64, out_channels=64).cuda()

    # 模拟PVT-v2-b0的4个stage输出
    test_cases = [
        (1, 32, 88, 88),  # Stage 1
        (1, 64, 44, 44),  # Stage 2
        (1, 160, 22, 22),  # Stage 3
        (1, 256, 11, 11),  # Stage 4
    ]

    for i, (b, c, h, w) in enumerate(test_cases):
        lde = LightweightDepthEstimation(in_channels=c, out_channels=c).cuda()
        x = torch.randn(b, c, h, w).cuda()
        out = lde(x)
        print(f"Stage {i + 1}: Input {x.shape} -> Output {out.shape}")

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,} ({total_params / 1e6:.2f}M)")