import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


class FreqProcessor(nn.Module):
    """频域处理器 - 针对水下图像特点优化"""

    def __init__(self, channels=256):
        super().__init__()
        self.channels = channels

        # 多尺度频域处理
        self.low_freq_weight = Parameter(torch.randn(channels, channels) * 0.01)
        self.high_freq_weight = Parameter(torch.randn(channels, channels) * 0.01)
        self.freq_bias = Parameter(torch.zeros(channels))

        # 频域注意力机制
        self.freq_attention = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, H, W = x.shape

        # FFT变换
        x_freq = torch.fft.rfft2(x, norm='ortho')
        real, imag = x_freq.real, x_freq.imag

        # 分离低频和高频成分
        h_half, w_half = real.shape[2] // 2, real.shape[3] // 2

        # 低频处理（中心区域）
        low_real = torch.einsum('bchw,cd->bdhw', real[:, :, :h_half, :w_half], self.low_freq_weight)
        low_imag = torch.einsum('bchw,cd->bdhw', imag[:, :, :h_half, :w_half], self.low_freq_weight)

        # 高频处理（边缘区域）
        high_real = torch.einsum('bchw,cd->bdhw', real, self.high_freq_weight) + self.freq_bias.view(1, -1, 1, 1)
        high_imag = torch.einsum('bchw,cd->bdhw', imag, self.high_freq_weight) + self.freq_bias.view(1, -1, 1, 1)

        # 重构
        x_freq_low = torch.complex(low_real, low_imag)
        x_freq_high = torch.complex(high_real, high_imag)

        x_low = torch.fft.irfft2(x_freq_low, s=(h_half, w_half), norm='ortho')
        x_high = torch.fft.irfft2(x_freq_high, s=(H, W), norm='ortho')

        # 上采样低频到原始尺寸
        x_low = F.interpolate(x_low, size=(H, W), mode='bilinear', align_corners=True)

        # 自适应融合
        freq_concat = torch.cat([x_low, x_high], dim=1)
        attention_weight = self.freq_attention(freq_concat)

        output = attention_weight * x_high + (1 - attention_weight) * x_low

        return output


class DeformConv(nn.Module):
    """可变形卷积 - 更稳定和高效"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # 偏移预测网络（减少参数量）
        self.offset_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 3, stride, padding),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, 2 * kernel_size * kernel_size, 3, stride, padding)
        )

        # 主卷积
        self.regular_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

        # 初始化
        nn.init.constant_(self.offset_conv[-1].weight, 0)
        nn.init.constant_(self.offset_conv[-1].bias, 0)

    def forward(self, x):
        # 预测偏移
        offset = self.offset_conv(x)
        offset = torch.tanh(offset) * (self.kernel_size // 2)  # 限制偏移范围

        # 应用偏移采样
        B, C, H, W = x.shape
        offset = offset.view(B, 2, -1, H, W)
        offset_mean = offset.mean(dim=2)  # [B, 2, H, W]

        # 创建采样网格
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=x.device),
            torch.linspace(-1, 1, W, device=x.device),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).repeat(B, 1, 1, 1)

        # 添加偏移
        offset_norm = offset_mean.permute(0, 2, 3, 1) * 0.1
        grid = grid + offset_norm

        # 应用变形采样
        x_offset = F.grid_sample(x, grid, mode='bilinear', padding_mode='border', align_corners=True)

        # 标准卷积
        output = self.regular_conv(x_offset)

        return output


class ColorCorrection(nn.Module):
    """水下颜色校正模块"""

    def __init__(self, channels):
        super().__init__()
        self.channels = channels

        # 通道注意力用于颜色校正
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid()
        )

        # 颜色增强
        self.color_enhance = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 通道注意力
        ca = self.channel_attention(x)
        x_weighted = x * ca

        # 颜色增强
        x_enhanced = self.color_enhance(x_weighted)

        # 残差连接
        return x + x_enhanced


class SpatialAttention(nn.Module):
    """空间注意力模块"""

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 通道维度的最大值和平均值
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        # 拼接并生成空间注意力图
        concat = torch.cat([avg_out, max_out], dim=1)
        attention = self.conv(concat)

        return x * attention


class DynamicFusion(nn.Module):
    """动态融合模块 - 单张量版本"""

    def __init__(self, rgb_channels, depth_channels, out_channels=None):
        super().__init__()

        if out_channels is None:
            out_channels = rgb_channels

        self.rgb_channels = rgb_channels
        self.depth_channels = depth_channels
        self.out_channels = out_channels

        # 水下颜色校正模块
        self.color_correction = ColorCorrection(out_channels)

        # RGB和Depth投影
        self.rgb_projection = nn.Sequential(
            nn.Conv2d(rgb_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.depth_projection = nn.Sequential(
            nn.Conv2d(depth_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # 改进的权重生成
        self.weight_gen = nn.Sequential(
            nn.Linear(out_channels * 2, out_channels // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(out_channels // 2, 2),
            nn.Softmax(dim=-1)
        )

        # 特征融合
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels)
        )

        # 空间注意力
        self.spatial_attention = SpatialAttention(out_channels)

    def forward(self, rgb_feat, depth_feat):
        B, _, H, W = rgb_feat.shape

        # 投影到相同维度
        rgb_proj = self.rgb_projection(rgb_feat)
        depth_proj = self.depth_projection(depth_feat)

        # 应用水下颜色校正
        rgb_proj = self.color_correction(rgb_proj)

        # 确保尺寸一致
        if depth_proj.shape[2:] != rgb_proj.shape[2:]:
            depth_proj = F.interpolate(depth_proj, size=rgb_proj.shape[2:],
                                       mode='bilinear', align_corners=True)

        # 全局上下文
        rgb_global = F.adaptive_avg_pool2d(rgb_proj, 1).squeeze(-1).squeeze(-1)
        depth_global = F.adaptive_avg_pool2d(depth_proj, 1).squeeze(-1).squeeze(-1)
        global_context = torch.cat([rgb_global, depth_global], dim=1)

        # 动态权重
        weights = self.weight_gen(global_context)
        rgb_weight = weights[:, 0].view(B, 1, 1, 1)
        depth_weight = weights[:, 1].view(B, 1, 1, 1)

        # 加权融合
        weighted_fusion = rgb_weight * rgb_proj + depth_weight * depth_proj

        # 深度融合
        concat_fused = torch.cat([rgb_proj, depth_proj], dim=1)
        deep_fusion = self.fusion_conv(concat_fused)

        # 空间注意力
        deep_fusion = self.spatial_attention(deep_fusion)

        # 残差连接
        output = weighted_fusion + deep_fusion + rgb_proj

        return output


class FrequencyGuidedFusion(nn.Module):
    """频域引导融合 - 单张量版本"""

    def __init__(self, rgb_channels, depth_channels, out_channels=None):
        super().__init__()

        if out_channels is None:
            out_channels = rgb_channels

        self.rgb_channels = rgb_channels
        self.depth_channels = depth_channels
        self.out_channels = out_channels

        # 频域处理分支
        self.freq_processor = FreqProcessor(rgb_channels)

        # 可变形卷积分支
        self.deform_conv = DeformConv(depth_channels, depth_channels)

        # 动态融合
        self.dynamic_fusion = DynamicFusion(
            rgb_channels=rgb_channels,
            depth_channels=depth_channels,
            out_channels=out_channels
        )

    def forward(self, rgb_feat, depth_feat):
        """
        Args:
            rgb_feat: RGB特征 [B, C_rgb, H, W]
            depth_feat: 深度特征 [B, C_depth, H, W]
        Returns:
            fused_feat: 融合后的特征 [B, C_out, H, W]
        """
        # RGB频域处理
        rgb_freq = self.freq_processor(rgb_feat)

        # Depth可变形卷积处理
        depth_deform = self.deform_conv(depth_feat)

        # 动态融合
        output = self.dynamic_fusion(rgb_freq, depth_deform)

        return output, torch.sigmoid(rgb_freq) * rgb_feat + rgb_freq


# 测试代码
if __name__ == '__main__':
    # 测试不同通道配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 测试case 1: 相同通道数
    model1 = FrequencyGuidedFusion(
        rgb_channels=64,
        depth_channels=64,
        out_channels=64
    ).to(device)

    rgb_feat1 = torch.randn(2, 64, 32, 32).to(device)
    depth_feat1 = torch.randn(2, 64, 32, 32).to(device)

    output1 = model1(rgb_feat1, depth_feat1)
    print(f"Test 1 - Same channels:")
    print(f"  RGB: {rgb_feat1.shape} + Depth: {depth_feat1.shape} -> Output: {output1.shape}")

    # 测试case 2: 不同通道数
    model2 = FrequencyGuidedFusion(
        rgb_channels=160,
        depth_channels=160,
        out_channels=128
    ).to(device)

    rgb_feat2 = torch.randn(2, 160, 16, 16).to(device)
    depth_feat2 = torch.randn(2, 160, 16, 16).to(device)

    output2 = model2(rgb_feat2, depth_feat2)
    print(f"\nTest 2 - Different output channels:")
    print(f"  RGB: {rgb_feat2.shape} + Depth: {depth_feat2.shape} -> Output: {output2.shape}")

    # 测试case 3: 不同尺寸（会自动调整）
    rgb_feat3 = torch.randn(2, 64, 32, 32).to(device)
    depth_feat3 = torch.randn(2, 64, 16, 16).to(device)

    output3 = model1(rgb_feat3, depth_feat3)
    print(f"\nTest 3 - Different spatial sizes:")
    print(f"  RGB: {rgb_feat3.shape} + Depth: {depth_feat3.shape} -> Output: {output3.shape}")

    # 计算参数量
    total_params = sum(p.numel() for p in model1.parameters())
    print(f"\nModel parameters: {total_params:,} ({total_params / 1e6:.2f}M)")