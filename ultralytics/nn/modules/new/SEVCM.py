import torch
import torch.nn as nn
from torch.functional import F



class SEVCM(nn.Module):
    def __init__(self, dim, out_dim):
        super().__init__()
        self.hidden = int(dim * 4)

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.hidden, self.hidden//16, 1),
            nn.SiLU(),
            nn.Conv2d(self.hidden//16, self.hidden, 1),
            nn.Sigmoid()
        )

        self.pw_linear = nn.Sequential(
            nn.Conv2d(self.hidden, out_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_dim),
            nn.SiLU()
        )

    def forward(self, x):
        x = torch.cat([
            x[..., ::2, ::2],
            x[..., 1::2, ::2],
            x[..., ::2, 1::2],
            x[..., 1::2, 1::2]
        ], dim=1)
        atten = self.se(x)
        x = x * atten
        return self.pw_linear(x)