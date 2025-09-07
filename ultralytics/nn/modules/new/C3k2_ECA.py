import torch
# 在c3k=True时，使用Bottleneck_WT特征融合，为false的时候我们使用普通的Bottleneck提取特征
from torch import nn

from ultralytics.nn.modules import C2f, Bottleneck, Conv, C3
from ultralytics.nn.modules.Myconv import WTConv2d

class C3k_ECA(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck_ECA(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))

class C3k2_ECA(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k_ECA(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck_ECA(self.c, self.c, shortcut, g) for _ in range(n)
        )

class Bottleneck_ECA(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2)
        self.eca = ECA(c_)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.eca(self.cv1(x))) if self.add else self.cv2(self.cv1(x))

class ECA(nn.Module):
    def __init__(self, channels, k_size=3):
        super(ECA, self).__init__()
        #print(k_size)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = torch.mean(x, dim=(2, 3), keepdim=True)  # 全局平均池化
        y = y.squeeze(-1).transpose(-1, -2)  # 转换为 (B, C, 1)
        y = self.sigmoid(self.conv(y)).transpose(-1, -2).unsqueeze(-1)
        return x * y.expand_as(x)