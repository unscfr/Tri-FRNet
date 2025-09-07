# 在c3k=True时，使用Bottleneck_WT特征融合，为false的时候我们使用普通的Bottleneck提取特征
import math

from torch import nn
from torch.nn.init import trunc_normal_

from ultralytics.nn.modules import C2f, Bottleneck, Conv, C3, ChannelAttention, DWConv
from ultralytics.nn.modules.Myconv import WTConv2d
from ultralytics.nn.modules.new.DyConv import DYConv2d
from ultralytics.nn.modules.new.MobileNetv4 import DepthwiseSeparableConv


class QFU(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.pw_linear = nn.Sequential(
            nn.Conv2d(in_dim, out_dim * 4, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_dim * 4),
            nn.SiLU()
        )
        self.upsample = nn.PixelShuffle(upscale_factor=2)

    def forward(self, x):
        x = self.pw_linear(x)            # [B, out_dim*4, H, W]
        x = self.upsample(x)             # [B, out_dim, H*2, W*2]
        return x

class oldQFU(nn.Module):
    def __init__(self, dim, out_dim):
        super().__init__()
        self.hidden = dim // 4  # 假设是2倍上采样

        self.pw_linear = nn.Sequential(
            nn.Conv2d(dim, out_dim * 4, kernel_size=1, stride=1, padding=0),  # 输出通道扩大4倍
            nn.BatchNorm2d(out_dim * 4),
            nn.SiLU()
        )

    def forward(self, x):
        x = self.pw_linear(x)
        b, c, h, w = x.shape

        # 将通道维度重排为空间维度实现上采样
        x = x.view(b, c // 4, 2, 2, h, w)  # 将通道分成4组
        x = x.permute(0, 1, 4, 2, 5, 3).contiguous()  # 重排维度
        return x.reshape(b, c // 4, h * 2, w * 2)  # 合并得到上采样结果

class DUpsampling(nn.Module):
    def __init__(self, inplanes, scale, num_class=4, pad=0):
        super(DUpsampling, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, num_class * scale * scale, kernel_size=1, padding=pad, bias=False)
        self.scale = scale

    def forward(self, x):
        x = self.conv1(x)
        N, C, H, W = x.size()

        # N, H, W, C
        x_permuted = x.permute(0, 2, 3, 1)

        # N, H, W*scale, C/scale
        x_permuted = x_permuted.contiguous().view((N, H, W * self.scale, int(C / (self.scale))))

        # N, W*scale,H, C/scale
        x_permuted = x_permuted.permute(0, 2, 1, 3)
        # N, W*scale,H*scale, C/(scale**2)
        x_permuted = x_permuted.contiguous().view(
            (N, W * self.scale, H * self.scale, int(C / (self.scale * self.scale))))

        # N,C/(scale**2),W*scale,H*scale
        x = x_permuted.permute(0, 3, 2, 1)

        return x


class MDC(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MDC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 两遍卷积
        self.cv1 = Conv(in_channels, in_channels//2, 1, 1)
        self.cv2 = Conv(in_channels//2, in_channels//2, 3, 2, 1)
        self.cv3 = Conv(in_channels, out_channels, 1, 1)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.dwcv1 = DepthwiseSeparableConv(in_channels//4, in_channels//4, 3, 2)

        self.bn1 = nn.BatchNorm2d(in_channels//2)
        self.bn2 = nn.BatchNorm2d(in_channels//2)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.silu(self.bn1(self.cv1(x)))
        x2 = x
        channels = x.size(1)
        x1, x3 = torch.split(x, [channels // 2, channels // 2], dim=1)
        x1 = self.maxpool(x1)
        x2 = self.silu(self.bn2(self.cv2(x2)))
        x3 = self.dwcv1(x3)
      #  print(x1.size(), x2.size(), x3.size())
        x = torch.cat((x1, x2, x3), dim=1)
        x = self.silu(self.bn3(self.cv3(x)))

        return x



class C3k_WT(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck_WT(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))

class C3k2_WT(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k_WT(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )

class Bottleneck_WT(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = WTConv2d(c_, c2)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


import torch
import torch.nn as nn


class EUCB(nn.Module):
    def __init__(self, in_channels, out_channels, upsampling_mode='nearest'):
        """
        Args:
            in_channels (int): 输入特征图的通道数
            out_channels (int): 输出特征图的通道数（匹配下一阶段）
            upsampling_mode (str): 上采样模式，支持 'nearest' 或 'bilinear'
        """
        super().__init__()
        # 上采样层（放大分辨率）
        self.up = nn.Upsample(
            scale_factor=2,
            mode=upsampling_mode,
            align_corners=False if upsampling_mode == 'bilinear' else None
        )

        # 3x3深度卷积 + BN + ReLU
        self.dw_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=in_channels,  # 深度卷积
            bias=False
        )
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)

        # 1x1卷积（调整通道数）
        self.conv1x1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )

    def forward(self, x):
        # 上采样
        x = self.up(x)
        # 深度卷积 + BN + 激活
        x = self.dw_conv(x)
        x = self.bn(x)
        x = self.relu(x)
        # 通道调整
        x = self.conv1x1(x)
        return x


class SCCF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):
        """
        Initializes the SPPF layer with given input/output channels and kernel size.

        This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = Conv(c_, c_, 3, 1)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        x = self.cv2(torch.cat(y, 1))
        #print(x.shape)
        return x

class C2f_Dy(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=2, shortcut=False, g=1, e=0.5):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(DyBottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(1))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))



class DyBottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = DYConv2d(c1, c_, 3, 0.25, 1, 1)
        self.cv2 = DYConv2d(c_, c2, 3, 0.25, 1, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))



class DWMultiScaleAttentionBS(nn.Module):
    def __init__(self, c1, c2):
        super(DWMultiScaleAttentionBS, self).__init__()
        # 使用深度可分离卷积
        self.conv1 = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=1, groups=c1),
            nn.Conv2d(c1, c2, kernel_size=1),
            nn.BatchNorm2d(c2),
            nn.SiLU()  # 或 nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, padding=1, groups=c1),
            nn.Conv2d(c1, c2, kernel_size=1),
            nn.BatchNorm2d(c2),
            nn.SiLU()  # 或 nn.ReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=5, padding=2, groups=c1),
            nn.Conv2d(c1, c2, kernel_size=1),
            nn.BatchNorm2d(c2),
            nn.SiLU()  # 或 nn.ReLU()
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        scale1 = self.conv1(x)
        scale3 = self.conv3(x)
        scale5 = self.conv5(x)
        attention_map = self.sigmoid(scale1 + scale3 + scale5)
        return x * attention_map


class C4f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.25):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, self.c, 1, 1)
        #self.cv2 = Conv((2 + n) * self.c, self.c, 1)  # optional act=FReLU(c2)
        self.cv3 = Conv(self.c * 3, c2, 1, 1)
        self.m = Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0)
        self.m1 = Bottleneck(self.c * 3, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0)
        self.ca = ChannelAttention(self.c * 2)

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.m(x1)
        x2 = torch.cat((x1, x2), 1)
        x4 = self.ca(x2)
        x5 = torch.cat((x1, x4), 1)
        x6 = self.m1(x5)
        x6 = torch.cat((x4, x6), 1)
        x7 = self.cv3(x6)
        return x7


class BasicRFB_a(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, scale=0.1):
        super(BasicRFB_a, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // 4

        self.branch0 = nn.Sequential(
            Conv(in_planes, inter_planes, 1, 1),
            Conv(inter_planes, inter_planes, 3, 1, 1)
        )
        self.branch1 = nn.Sequential(
            Conv(in_planes, inter_planes, 1, 1),
            Conv(inter_planes, inter_planes, (3, 1), 1, (1, 0)),
            Conv(inter_planes, inter_planes, 3, 1, 3, d=3)
        )
        self.branch2 = nn.Sequential(
            Conv(in_planes, inter_planes, 1, 1),
            Conv(inter_planes, inter_planes, (1, 3), stride, (0, 1)),
            Conv(inter_planes, inter_planes, 3, 1, 3, d=3)
        )
        self.branch3 = nn.Sequential(
            Conv(in_planes, inter_planes // 2, 1, 1),
            Conv(inter_planes // 2, (inter_planes // 4) * 3, (1, 3), 1, (0, 1)),
            Conv((inter_planes // 4) * 3, inter_planes, (3, 1), stride, (1, 0)),
            Conv(inter_planes, inter_planes, 3, 1, 5, d=5)
        )

        self.ConvLinear = Conv(4 * inter_planes, out_planes, 1, 1)
        self.shortcut = Conv(in_planes, out_planes, 1, stride)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        out = torch.cat((x0, x1, x2, x3), 1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out * self.scale + short
        out = self.relu(out)

        return out

class C2f_RFB(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(BasicRFB_a(self.c, self.c) for _ in range(n))


    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        #print(y[0].shape, y[1].shape)
        return self.cv2(torch.cat(y, 1))


import torch
import torch.nn as nn
import torch.nn.functional as F


class h_swish(nn.Module):
    def __init__(self, inplace=False):
        super(h_swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, groups=32,
                 activation='h_swish',
                 pool_type='avg',
                 residual=True,
                 combine='multiply'):
        super(CoordAtt, self).__init__()

        # Pooling layers
        if pool_type == 'avg':
            self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
            self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        elif pool_type == 'max':
            self.pool_h = nn.AdaptiveMaxPool2d((None, 1))
            self.pool_w = nn.AdaptiveMaxPool2d((1, None))
        else:
            raise ValueError("pool_type must be either 'avg' or 'max'")

        # Calculate mip with group alignment
        mip = max(8, inp // groups)
        mip = (mip // groups) * groups
        if mip < groups:
            mip = groups

        # Shared layers
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)

        # Activation function
        if activation == 'h_swish':
            self.relu = h_swish()
        elif activation == 'relu':
            self.relu = nn.ReLU()
        elif activation == 'leaky_relu':
            self.relu = nn.LeakyReLU(0.1)
        else:
            raise ValueError("activation must be 'h_swish', 'relu' or 'leaky_relu'")

        # Attention branches
        self.conv2 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

        # Configuration
        self.residual = residual
        self.combine = combine

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()

        # Pooling
        x_h = self.pool_h(x)  # (n, c, h, 1)
        x_w = self.pool_w(x).transpose(2, 3)  # (n, c, 1, w) -> (n, c, w, 1)

        # Concatenate and process
        y = torch.cat([x_h, x_w], dim=2)  # (n, c, h+w, 1)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.relu(y)

        # Split
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.transpose(2, 3)  # (n, c, w, 1) -> (n, c, 1, w)

        # Attention weights
        x_h = self.conv2(x_h).sigmoid()  # (n, oup, h, 1)
        x_w = self.conv3(x_w).sigmoid()  # (n, oup, 1, w)

        # Combine attention
        if self.combine == 'multiply':
            att = x_w * x_h  # (n, oup, h, w) - broadcasting
        elif self.combine == 'add':
            att = (x_w + x_h) / 2.0
        elif self.combine == 'concat':
            att = torch.cat([x_w.expand(-1, -1, h, -1),
                             x_h.expand(-1, -1, -1, w)], dim=1)
        else:
            raise ValueError("combine must be 'multiply', 'add' or 'concat'")

        # Apply attention
        if self.residual:
            y = identity * att + identity
        else:
            y = identity * att

        return y

class C2f_CA(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(CoordAtt(self.c, self.c) for _ in range(n))


    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        #print(y[0].shape, y[1].shape)
        return self.cv2(torch.cat(y, 1))


import torch  # 导入 PyTorch 库
from torch import nn  # 从 PyTorch 中导入神经网络模块


class EMA(nn.Module):  # 定义一个继承自 nn.Module 的 EMA 类
    def __init__(self, channels, c2=None, factor=32):  # 构造函数，初始化对象
        super(EMA, self).__init__()  # 调用父类的构造函数
        self.groups = factor  # 定义组的数量为 factor，默认值为 32
        assert channels // self.groups > 0  # 确保通道数可以被组数整除
        self.softmax = nn.Softmax(-1)  # 定义 softmax 层，用于最后一个维度
        self.agp = nn.AdaptiveAvgPool2d((1, 1))  # 定义自适应平均池化层，输出大小为 1x1
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # 定义自适应平均池化层，只在宽度上池化
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # 定义自适应平均池化层，只在高度上池化
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)  # 定义组归一化层
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1,
                                 padding=0)  # 定义 1x1 卷积层
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1,
                                 padding=1)  # 定义 3x3 卷积层

    def forward(self, x):  # 定义前向传播函数
        b, c, h, w = x.size()  # 获取输入张量的大小：批次、通道、高度和宽度
        group_x = x.reshape(b * self.groups, -1, h, w)  # 将输入张量重新形状为 (b * 组数, c // 组数, 高度, 宽度)
        x_h = self.pool_h(group_x)  # 在高度上进行池化
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)  # 在宽度上进行池化并交换维度
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))  # 将池化结果拼接并通过 1x1 卷积层
        x_h, x_w = torch.split(hw, [h, w], dim=2)  # 将卷积结果按高度和宽度分割
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())  # 进行组归一化，并结合高度和宽度的激活结果
        x2 = self.conv3x3(group_x)  # 通过 3x3 卷积层
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))  # 对 x1 进行池化、形状变换、并应用 softmax
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # 将 x2 重新形状为 (b * 组数, c // 组数, 高度 * 宽度)
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))  # 对 x2 进行池化、形状变换、并应用 softmax
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # 将 x1 重新形状为 (b * 组数, c // 组数, 高度 * 宽度)
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)  # 计算权重
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)  # 应用权重并将形状恢复为原始大小