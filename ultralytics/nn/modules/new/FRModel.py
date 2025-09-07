from typing import Any

import torch
import torch.nn as nn

from ultralytics.nn.modules import Conv
from ultralytics.nn.modules.mamba_yolo import SS2D


class BottleneckSS2D(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5,
                 ssm_d_state: int = 16,
                 ssm_ratio=2.0,
                 ssm_rank_ratio=2.0,
                 ssm_dt_rank: Any = "auto",
                 ssm_act_layer=nn.SiLU,
                 ssm_conv: int = 3,
                 ssm_conv_bias=True,
                 ssm_drop_rate: float = 0,
                 hidden_dim: int = 8,
                 n: int = 1,
                 mlp_ratio=4.0,
                 drop_path: float = 0,

                 ):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        self.hidden_dim = hidden_dim
        c_ = 8 #int(c2 * e)  # hidden channels
        #print(c_)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.ss2d = nn.Sequential(*(SS2D(d_model=self.hidden_dim,
                                         d_state=ssm_d_state,
                                         ssm_ratio=ssm_ratio,
                                         ssm_rank_ratio=ssm_rank_ratio,
                                         dt_rank=ssm_dt_rank,
                                         act_layer=ssm_act_layer,
                                         d_conv=ssm_conv,
                                         conv_bias=ssm_conv_bias,
                                         dropout=ssm_drop_rate, ) for _ in range(n)))
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.ss2d(self.cv1(x)) if self.add else self.ss2d(self.cv1(x))


class FRModel(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FRModel, self).__init__()
        self.CBS1 = nn.Sequential(
            nn.Conv2d(in_channels // 2, in_channels // 2, 5, 1, 2),
            #nn.BatchNorm2d(in_channels // 2),
            #nn.SiLU(inplace=True)
        )
        self.CBS2 = nn.Sequential(
            nn.Conv2d(in_channels // 2, in_channels // 2, 3, 1, 1),
            #nn.BatchNorm2d(in_channels // 2),
            #nn.SiLU(inplace=True)
        )
        self.CBS3 = nn.Sequential(
            nn.Conv2d(in_channels // 2, in_channels // 2, 1, 1, 0),
            #nn.BatchNorm2d(in_channels // 2),
            #nn.SiLU(inplace=True)
        )
        self.cv1 = nn.Sequential(
            nn.Conv2d(in_channels // 2, in_channels // 4, 1, 1, 0),
            nn.BatchNorm2d(in_channels // 4),
            nn.SiLU(inplace=True)
        )
        self.cv2 = nn.Sequential(
            nn.Conv2d(392, 256, 1, 1, 0),
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True)
        )
        #self.spatial = SpatialAttention()
        #self.channel = ChannelAttention(in_channels // 4)
        self.bottleneckss2d = BottleneckSS2D(in_channels // 4, in_channels // 2)


    def forward(self, x):
        x1, x2 = torch.chunk(x, 2, 1)
        y1 = self.CBS1(x1)
        y2 = self.CBS2(x1)
        y3 = self.CBS3(x1)
        x2 = self.cv1(x2)
        #print(x2.shape)
        y4 = self.bottleneckss2d(x2)
        #print(y1.shape, y2.shape, y3.shape, y4.shape)
        x1 = torch.cat((y1, y2, y3, y4), 1)
        x1 = self.cv2(x1)
        #print(x1.shape)
        return x1

if __name__ == "__main__":
    # 输入大小: [batch_size, channels, height, width]
    x = torch.randn(16, 32, 8, 8)  # 64通道，32x32尺寸的输入

    model = FRModel(in_channels=32, out_channels=64)
    output = model(x)

    print(f"Output shape: {output.shape}")  # 输出形状
