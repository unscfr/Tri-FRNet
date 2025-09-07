from torch import nn


class ConvNextBlock(nn.Module):
    """ ConvNeXtV2 Block.

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """

    def __init__(self, dim1, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim1, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        from torch.nn import LayerNorm
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        from ultralytics.nn.modules.new.ConvNext.uilts import GRN
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        from ultralytics.models.sam.modules.blocks import DropPath
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        #print(dim1,dim)

    def forward(self, x):


        #print(x.shape)
        x = self.dwconv(x)
        input = x
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x