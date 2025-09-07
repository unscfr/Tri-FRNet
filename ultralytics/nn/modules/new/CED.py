import torch
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks import DropPath
from torch import nn

from ultralytics.nn.modules import Conv

BaseBlock = None
FocalBlock = None
act_layer = nn.GELU
ls_init_value = 1e-6



class DBottleneck(nn.Module):

    def __init__(self, dim, drop_path=0., dilation=3, norm_cfg=dict(type='BN'), **kwargs):
        super(DBottleneck, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False),
            build_norm_layer(norm_cfg, dim)[1],
            act_layer(),
        )

        self.dwconv1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=7, padding=3 * dilation, dilation=dilation, groups=dim),
            build_norm_layer(norm_cfg, dim)[1],
            act_layer())

        self.conv2 = nn.Sequential(
            nn.Conv2d(dim, dim * 4, kernel_size=1, stride=1, bias=False),
            build_norm_layer(norm_cfg, dim * 4)[1],
            act_layer(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(dim * 4, dim, kernel_size=1, bias=False),
            build_norm_layer(norm_cfg, dim)[1],
        )

        self.act = act_layer()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.conv1(x) + x
        x = self.dwconv1(x) + x
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.act(self.drop_path(x) + input)
        return x


class DNeXtBlock(nn.Module):
    def __init__(self, c1, c2, drop_path=0., dilation=3, norm_cfg=dict(type='BN'),  ** kwargs):
        super().__init__()

        # 确保dilation是正整数
        assert isinstance(dilation, int) and dilation >= 1, "dilation must be ≥ 1"
        #print('dim',c1,'out',c2, 'drop_path',drop_path,'dilation',dilation)


        # dwconv1: groups必须等于dim
        self.dwconv1 = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=7, padding=3,
                      dilation=1, groups=c1),  # 关键修改：groups=dim
            build_norm_layer(norm_cfg, c1)[1],
            act_layer())

        # dwconv2: groups必须等于dim
        self.dwconv2 = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=7,
                      padding=dilation*3,  # padding=dilation
                      dilation=dilation,
                      groups=c1),  # 关键修改：groups=dim
            build_norm_layer(norm_cfg, c1)[1],
            act_layer())

        # 其余部分保持不变
        self.pwconv1 = nn.Linear(c1, 4 * c1)
        self.act = act_layer()
        self.pwconv2 = nn.Linear(4 * c1, c2)
        self.gamma = nn.Parameter(
            ls_init_value * torch.ones((c2)), requires_grad=True) if ls_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.c1 = c1
        self.cv1 = Conv(c1, c2, 1, 1)

    def forward(self, x):

        input = x
        #print(x.shape)
        #print('c1',self.c1)
        x = self.dwconv1(x) + x

        x1 = self.dwconv2(x)
        #print(x.shape), print(x1.shape)
        x = x1 + x

        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        input = self.cv1(input)
        x = input + self.drop_path(x)
        return x

class DNeXtBlockDW(nn.Module):
    def __init__(self, c1, c2, drop_path=0., dilation=3, norm_cfg=dict(type='BN'),  ** kwargs):
        super().__init__()

        # 确保dilation是正整数
        assert isinstance(dilation, int) and dilation >= 1, "dilation must be ≥ 1"
        #print('dim',c1,'out',c2, 'drop_path',drop_path,'dilation',dilation)


        # dwconv1: groups必须等于dim
        self.dwconv1 = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=7, padding=3,
                      dilation=1, groups=c1),  # 关键修改：groups=dim
            build_norm_layer(norm_cfg, c1)[1],
            act_layer())

        # dwconv2: groups必须等于dim
        self.dwconv2 = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=7,
                      padding=dilation*3,  # padding=dilation
                      dilation=dilation,
                      groups=c1),  # 关键修改：groups=dim
            build_norm_layer(norm_cfg, c1)[1],
            act_layer())

        # 其余部分保持不变
        self.pwconv1 = nn.Conv2d(c1, 4 * c1, 1)
        self.act = act_layer()
        self.pwconv2 = nn.Conv2d(4 * c1, c2, 1)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.c1 = c1
        self.cv1 = Conv(c1, c2, 1, 1)

    def forward(self, x):

        input = x
        #print(x.shape)
        #print('c1',self.c1)
        x = self.dwconv1(x) + x

        x1 = self.dwconv2(x)
        #print(x.shape), print(x1.shape)
        x = x1 + x

        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        input = self.cv1(input)
        x = input + self.drop_path(x)
        return x
# # 测试模块
# if __name__ == "__main__":
#     # 输入大小: [batch_size, channels, height, width]
#     x = torch.randn(1, 64, 16, 16)  # 64通道，32x32尺寸的输入
#     #print(f"Input shape: {x.shape}")  # 输出形状
#     model1 = DNeXtBlock(64)  # 输入64通道，输出128通道
#     model2 = DBottleneck(64)
#     output1 = model1(x)
#     output2 = model2(x)
#
#     print(f"Output1 shape: {output1.shape}")  # 输出形状
#     print(f"Output2 shape: {output2.shape}")  # 输出形状
