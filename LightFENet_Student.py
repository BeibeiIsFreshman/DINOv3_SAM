import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from KDNet.PVTv2 import pvt_v2_b0
from torch.nn.functional import interpolate


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_planes, out_planes,
                      kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=False))

    def forward(self, x):
        return self.conv(x)


class Up(nn.Module):
    def __init__(self, scale_factor, mode, align_corners=True):
        super(Up, self).__init__()
        self.interp = interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        return self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)


class Decoder(nn.Module):
    """解码器 - 添加边缘增强"""
    def __init__(self, in1, in2, in3, in4):
        super(Decoder, self).__init__()
        self.bcon4 = nn.Sequential(
            Up(scale_factor=2, mode='bilinear', align_corners=True),
            BasicConv2d(in4, in4, kernel_size=3, stride=1, padding=1)
        )
        self.bcon3 = BasicConv2d(in3, in4, kernel_size=3, stride=1, padding=1)
        self.bcon2 = BasicConv2d(in2, in3, kernel_size=3, stride=1, padding=1)
        self.bcon1 = BasicConv2d(in_planes=in1, out_planes=in2, kernel_size=1, stride=1, padding=0)

        self.bcon4_3 = nn.Sequential(
            Up(scale_factor=2, mode='bilinear', align_corners=True),
            BasicConv2d(in4 * 2, in3, kernel_size=3, stride=1, padding=1)
        )
        self.bcon3_2 = nn.Sequential(
            Up(scale_factor=2, mode='bilinear', align_corners=True),
            BasicConv2d(in3 * 2, in2, kernel_size=3, stride=1, padding=1)
        )
        self.bcon2_1 = nn.Sequential(
            Up(scale_factor=2, mode='bilinear', align_corners=True),
            BasicConv2d(in2 * 2, in1, kernel_size=3, stride=1, padding=1)
        )

        self.conv_d1 = nn.Sequential(
            Up(scale_factor=2, mode='bilinear', align_corners=True),
            BasicConv2d(in3, in2, kernel_size=3, stride=1, padding=1)
        )
        self.conv_d2 = nn.Sequential(
            Up(scale_factor=2, mode='bilinear', align_corners=True),
            BasicConv2d(in2 * 2, in1, kernel_size=3, stride=1, padding=1)
        )
        self.conv_d3 = BasicConv2d(in2, in1, kernel_size=3, stride=1, padding=1)
        
        # 边缘增强模块
        self.edge_enhance = nn.Sequential(
            nn.Conv2d(in1, in1, 3, 1, 1),
            nn.BatchNorm2d(in1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in1, in1, 3, 1, 1),
            nn.BatchNorm2d(in1)
        )

    def forward(self, f):
        f[3] = self.bcon4(f[3])
        f[2] = self.bcon3(f[2])
        f[1] = self.bcon2(f[1])
        f[0] = self.bcon1(f[0])

        d43 = self.bcon4_3(torch.cat((f[3], f[2]), 1))
        d32 = self.bcon3_2(torch.cat((d43, f[1]), 1))
        d21 = self.bcon2_1(torch.cat((d32, f[0]), 1))
        out = d21

        d43 = self.conv_d1(d43)
        d32 = torch.cat((d43, d32), dim=1)
        d32 = self.conv_d2(d32)
        d21 = torch.cat((d32, d21), dim=1)
        d21 = self.conv_d3(d21)
        
        # 边缘增强
        d21_enhanced = self.edge_enhance(d21)
        d21 = d21 + d21_enhanced

        return d21, out, d32, d43


class LightFENet(nn.Module):
    """轻量级频域增强网络 - Lightweight Frequency-Enhanced Network"""
    def __init__(self):
        super().__init__()
        self.decoder = Decoder(32, 64, 160, 256)
        self.backbone = pvt_v2_b0(True)

        self.out1_best = nn.Sequential(
            Up(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(32, 1, 3, 1, 1)
        )
        self.out2_best = nn.Sequential(
            Up(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(32, 1, 3, 1, 1)
        )
        self.out3_best = nn.Sequential(
            Up(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(32, 1, 3, 1, 1)
        )
        self.out4_best = nn.Sequential(
            Up(scale_factor=4, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 1, 3, 1, 1)
        )

    def forward(self, rgb):
        # 通过骨干网络提取特征
        rgb_list, dep_list = self.backbone(rgb)
        # 通过解码器生成最终输出
        output = self.decoder(rgb_list)

        return (F.sigmoid(self.out1_best(output[0])), 
                F.sigmoid(self.out2_best(output[1])), 
                F.sigmoid(self.out3_best(output[2])), 
                F.sigmoid(self.out4_best(output[3])),
                dep_list
                )


def test_model():
    """测试模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = LightFENet().to(device)
    
    test_rgb = torch.randn(1, 3, 256, 256).to(device)
    
    with torch.no_grad():
        outputs = model(test_rgb)
    
    print("输出特征形状:")
    for i, out in enumerate(outputs):
        print(f"  Output {i+1}: {out.shape}")
    
    try:
        from thop import profile
        flops, params = profile(model, (test_rgb,))
        print(f"\nFLOPs: {flops / 1e9:.2f} G")
        print(f"Params: {params / 1e6:.2f} M")
    except:
        print("\n提示: 安装 thop 库可查看模型复杂度")
    
    return model, outputs


if __name__ == "__main__":
    model, outputs = test_model()
    print("\n✅ LightFENet模型测试成功!")