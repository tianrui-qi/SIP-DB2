import torch
import torch.nn as nn
from torch import Tensor

from typing import List

__all__ = ["FCN"]


class FCN(nn.Module):
    def __init__(self, feats: List[int]) -> None:
        super(FCN, self).__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(feats[i], feats[i+1]) for i in range(len(feats) - 1)]
        )
        # output
        self.fc = nn.Linear(self.feats[-1], 1)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers: x = layer(x)
        x = self.fc(x)
        return x


class _ChannelAttentionModule(nn.Module):
    def __init__(self, dim: int, in_c: int, ratio: int = 16) -> None:
        super(_ChannelAttentionModule, self).__init__()
        if   dim == 2: 
            Conv = nn.Conv2d
            AdaptiveAvgPool = nn.AdaptiveAvgPool2d
            AdaptiveMaxPool = nn.AdaptiveMaxPool2d
        elif dim == 3:
            Conv = nn.Conv3d
            AdaptiveAvgPool = nn.AdaptiveAvgPool3d
            AdaptiveMaxPool = nn.AdaptiveMaxPool3d
        else: raise ValueError("dim must be 2 or 3")

        self.avg_pool = AdaptiveAvgPool(1)
        self.max_pool = AdaptiveMaxPool(1)
        self.MLP = nn.Sequential(
            Conv(in_c, in_c // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            Conv(in_c // ratio, in_c, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        return self.sigmoid(
            self.MLP(self.avg_pool(x)) + self.MLP(self.max_pool(x))
        )


class _SpatialAttentionModule(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 7) -> None:
        super(_SpatialAttentionModule, self).__init__()
        if   dim == 2: Conv = nn.Conv2d
        elif dim == 3: Conv = nn.Conv3d
        else: raise ValueError("dim must be 2 or 3")

        self.conv = Conv(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class _DualConv(nn.Module):
    def __init__(
        self, dim: int, in_c: int, out_c: int, 
        use_cbam: bool = False, use_res: bool = False
    ) -> None:
        super(_DualConv, self).__init__()
        self.use_cbam = use_cbam
        self.use_res  = use_res

        if   dim == 2: Conv, BatchNorm = nn.Conv2d, nn.BatchNorm2d
        elif dim == 3: Conv, BatchNorm = nn.Conv3d, nn.BatchNorm3d
        else: raise ValueError("dim must be 2 or 3")

        # dual convolution
        self.conv1 = nn.Sequential(
            Conv(in_c, out_c, 3, padding=1, bias=False),
            BatchNorm(out_c)
        )
        self.conv2 = nn.Sequential(
            Conv(out_c, out_c, 3, padding=1, bias=False),
            BatchNorm(out_c)
        )
        self.relu = nn.ReLU(inplace=True)

        # cbam
        if self.use_cbam:
            self.channel_attention = _ChannelAttentionModule(dim, out_c)
            self.spatial_attention = _SpatialAttentionModule(dim)

        # residual
        if self.use_res:
            if in_c == out_c:
                self.skip = nn.Identity()
            else:
                self.skip = nn.Sequential(
                    Conv(in_c, out_c, 1, bias=False),
                    BatchNorm(out_c),
                )

    def forward(self, x: Tensor) -> Tensor:
        if self.use_res: 
            res = x
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        if self.use_cbam: 
            x = self.channel_attention(x) * x
            x = self.spatial_attention(x) * x
        if self.use_res: 
            x += self.skip(res)  # type: ignore
        return self.relu(x)


class ResAttNet(nn.Module):
    def __init__(self, feats: List[int], use_cbam: bool, use_res: bool) -> None:
        super(ResAttNet, self).__init__()
        self.feats = feats
        self.use_cbam = use_cbam
        self.use_res  = use_res

        # encoder
        self.encoder = nn.ModuleList([
            nn.Sequential(
                _DualConv(
                    3, self.feats[i] , self.feats[i+1], 
                    self.use_cbam, self.use_res
                ), 
                _DualConv(
                    3, self.feats[i+1], self.feats[i+1], 
                    self.use_cbam, self.use_res
                ), 
                _DualConv(
                    3, self.feats[i+1], self.feats[i+1], 
                    self.use_cbam, self.use_res
                )
            ) for i in range(len(self.feats)-1)
        ])
        self.maxpool = nn.ModuleList([
            nn.MaxPool3d(2) for _ in range(len(self.feats)-2)
        ])
        # output
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(self.feats[-1], 1)

    def forward(self, x: Tensor) -> Tensor:
        x = x.unsqueeze(1)
        # encoder
        x = self.encoder[0](x)
        for i in range(0, len(self.feats)-2):
            x = self.maxpool[ i ](x)
            x = self.encoder[i+1](x)
        # output
        x = self.avgpool(x)  # (B, C, 1, 1, 1)
        x = x.view(x.size(0), -1)  # (B, C)
        x = self.fc(x)
        return x.squeeze()
