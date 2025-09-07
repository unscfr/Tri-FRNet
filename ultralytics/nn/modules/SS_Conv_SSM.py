from functools import partial
from typing import Callable

import torch
from mmcv.cnn.bricks import DropPath
from torch import nn, channel_shuffle

from ultralytics.nn.modules import Conv
from ultralytics.nn.modules.mamba_yolo import SS2D


class SS_Conv_SSM(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        **kwargs,
    ):
        super().__init__()
        #self.ln_1 = norm_layer(hidden_dim)
        #print(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim//2, dropout=attn_drop_rate, d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path)

        self.conv33conv33conv11 = nn.Sequential(
            nn.BatchNorm2d(hidden_dim // 2),
            nn.Conv2d(in_channels=hidden_dim//2,out_channels=hidden_dim//2,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(hidden_dim//2),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_dim // 2, out_channels=hidden_dim // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_dim // 2, out_channels=hidden_dim // 2, kernel_size=1, stride=1),
            nn.ReLU()
        )
        # self.finalconv11 = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=1, stride=1)
        self.cv1 = Conv(128, 64, 1)
        self.cv2 = Conv(hidden_dim, 64, 1)
        self.cv = Conv(32, 128, 1)
        self.cv3 = Conv(128, 64,1)
    def forward(self, input: torch.Tensor):
        input_left, input_right = input.chunk(2,dim=-1)
        input_right = self.cv1(input_right)
        x = self.drop_path(self.self_attention(input_right))
        input_left = input_left.permute(0,3,1,2).contiguous()
        input_left = self.cv2(input_left)
        #print(input_left.shape)
        input_left = self.conv33conv33conv11(input_left)
        input_left = input_left.permute(0,2,3,1).contiguous()
        x = x.permute(0,3,1,2).contiguous()
        #print(input_left.shape, x.shape)
        x = self.cv(x)
        output = torch.cat((input_left,x),dim=-1)
        output = self.cv3(output)
        output = output.permute(0,3,2,1).contiguous()
        #print(output.shape)
        output = channel_shuffle(output,groups=2)
        #print(output.shape, input.shape)
        return output+input