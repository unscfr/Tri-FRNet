import torch
from torch import nn

from ultralytics.nn.modules import C2f, Bottleneck, Conv
from ultralytics.nn.modules.block import C3k
from ultralytics.nn.modules.new.cga import SpatialAttention, ChannelAttention, PixelAttention


class C2f1(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(128, 256, 1, 1)
        self.cv2 = Conv(256, 128, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        x = self.cv2(torch.cat(y, 1))
        #print('c2f1',x.shape)
        return x


class C2f2(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(192, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C2f3(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(128, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class CGAFusion1(nn.Module):
    def __init__(self, dim, reduction=8):
        super(CGAFusion1, self).__init__()
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(dim, reduction)
        self.pa = PixelAttention(dim)
        self.conv = nn.Conv2d(dim, dim, 1, bias=True)
        self.sigmoid = nn.Sigmoid()
        #self.dim = dim

    def forward(self, input):
        #print(self.dim)
        #print('input1',input[0].shape, input[1].shape)
        x, y = input[0], input[1]
        initial = x + y

        cattn = self.ca(initial)
        sattn = self.sa(initial)
        pattn1 = sattn + cattn
        pattn2 = self.sigmoid(self.pa(initial, pattn1))
        result = initial + pattn2 * x + (1 - pattn2) * y
        result = self.conv(result)
        #print('cga1',result.shape)
        return result

class CGAFusion2(nn.Module):
    def __init__(self, dim, reduction=8):
        super(CGAFusion2, self).__init__()
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(dim, reduction)
        self.pa = PixelAttention(dim)
        self.conv = nn.Conv2d(dim, dim, 1, bias=True)
        self.sigmoid = nn.Sigmoid()
        #self.dim = dim

    def forward(self, input):
        #print(self.dim)
        #print('input2',input[0].shape, input[1].shape)
        x, y = input[0], input[1]
        initial = x + y

        cattn = self.ca(initial)
        sattn = self.sa(initial)
        pattn1 = sattn + cattn
        pattn2 = self.sigmoid(self.pa(initial, pattn1))
        result = initial + pattn2 * x + (1 - pattn2) * y
        result = self.conv(result)
        #print('cga2',result.shape)
        return result

class CGAFusion3(nn.Module):
    def __init__(self, dim, reduction=8):
        super(CGAFusion3, self).__init__()
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(dim, reduction)
        self.pa = PixelAttention(dim)
        self.conv = nn.Conv2d(dim, dim, 1, bias=True)
        self.sigmoid = nn.Sigmoid()
        #self.dim = dim

    def forward(self, input):
        #print(self.dim)
        #print('input3',input[0].shape, input[1].shape)
        x, y = input[0], input[1]
        initial = x + y

        cattn = self.ca(initial)
        sattn = self.sa(initial)
        pattn1 = sattn + cattn
        pattn2 = self.sigmoid(self.pa(initial, pattn1))
        result = initial + pattn2 * x + (1 - pattn2) * y
        result = self.conv(result)
        #print('cga3',result.shape)
        return result
# 测试模块
# if __name__ == "__main__":
#     # 假设输入为(batch_size, in_channels, height, width)
#     input_tensor1 = torch.randn(3, 64, 64, 64)  # Batch size 8, 输入通道64，32x32的特征图
#     input_tensor2 = torch.randn(3, 64, 64, 64)  # Batch size 8, 输入通道64，32x32的特征图
#     model = CGAFusion(64, 8)
#     output_tensor = model(input_tensor1, input_tensor2)
#     print(f"Output shape: {output_tensor.shape}")  # 输出形状