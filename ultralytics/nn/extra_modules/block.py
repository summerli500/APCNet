"""C3k2_RFCBAMConv + CGAFusion."""
import torch
import torch.nn as nn
from einops import rearrange

from ..modules.conv import Conv
from ..modules.block import Bottleneck, C3k, C3k2
from .RFAconv import RFCBAMConv

__all__ = [
    "Bottleneck_RFCBAMConv",
    "C3k_RFCBAMConv",
    "C3k2_RFCBAMConv",
    "SpatialAttention_CGA",
    "ChannelAttention_CGA",
    "PixelAttention_CGA",
    "CGAFusion",
]


class Bottleneck_RFCBAMConv(Bottleneck):
    """Standard bottleneck with RFCBAMConv."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = RFCBAMConv(c_, c2, k[1])


class C3k_RFCBAMConv(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)
        self.m = nn.Sequential(
            *(Bottleneck_RFCBAMConv(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n))
        )


class C3k2_RFCBAMConv(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(
            C3k_RFCBAMConv(self.c, self.c, 2, shortcut, g)
            if c3k
            else Bottleneck_RFCBAMConv(self.c, self.c, shortcut, g)
            for _ in range(n)
        )


class SpatialAttention_CGA(nn.Module):
    def __init__(self):
        super().__init__()
        self.sa = nn.Conv2d(2, 1, 7, padding=3, padding_mode="reflect", bias=True)

    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        x2 = torch.concat([x_avg, x_max], dim=1)
        return self.sa(x2)


class ChannelAttention_CGA(nn.Module):
    def __init__(self, dim, reduction=8):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1, padding=0, bias=True),
        )

    def forward(self, x):
        return self.ca(self.gap(x))


class PixelAttention_CGA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pa2 = nn.Conv2d(2 * dim, dim, 7, padding=3, padding_mode="reflect", groups=dim, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, pattn1):
        x = x.unsqueeze(dim=2)
        pattn1 = pattn1.unsqueeze(dim=2)
        x2 = torch.cat([x, pattn1], dim=2)
        x2 = rearrange(x2, "b c t h w -> b (c t) h w")
        return self.sigmoid(self.pa2(x2))


class CGAFusion(nn.Module):
    def __init__(self, dim, reduction=8):
        super().__init__()
        self.sa = SpatialAttention_CGA()
        self.ca = ChannelAttention_CGA(dim, reduction)
        self.pa = PixelAttention_CGA(dim)
        self.conv = nn.Conv2d(dim, dim, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, data):
        x, y = data
        initial = x + y
        cattn = self.ca(initial)
        sattn = self.sa(initial)
        pattn1 = sattn + cattn
        pattn2 = self.sigmoid(self.pa(initial, pattn1))
        result = initial + pattn2 * x + (1 - pattn2) * y
        return self.conv(result)
