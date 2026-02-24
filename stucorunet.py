"""
stucorunet.py

StuCorrUNet:
- Siamese "correlation" encoder for I0 and I1 (plus their gradients) -> local cost volume at 1/8 scale.
- Context encoder-decoder (ResUNet) operating on X=[I0,I1,I1-I0,|∇I0|,|∇I1|] (5ch).

Outputs:
    F01_pred: [B,2,H,W]  (t0->t1, defined on I0 grid)
    F10_pred: [B,2,H,W]  (t1->t0, defined on I1 grid)
"""

from __future__ import annotations

from typing import Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _choose_gn_groups(c: int) -> int:
    """
    Heuristic:
    - prefer 16 groups if possible, else 8, else 4, else 1
    """
    for g in (16, 8, 4, 2, 1):
        if c % g == 0:
            return g
    return 1


class ConvGNAct(nn.Module):
    def __init__(self, cin: int, cout: int, k: int = 3, s: int = 1, p: Optional[int] = None):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(cin, cout, k, stride=s, padding=p, bias=False)
        self.gn = nn.GroupNorm(_choose_gn_groups(cout), cout, eps=1e-5, affine=True)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.gn(self.conv(x)))


class ResBlockGN(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(_choose_gn_groups(c), c, eps=1e-5, affine=True)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(_choose_gn_groups(c), c, eps=1e-5, affine=True)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = x
        x = self.act(self.gn1(self.conv1(x)))
        x = self.gn2(self.conv2(x))
        return self.act(x + r)


class DownGN(nn.Module):
    def __init__(self, cin: int, cout: int, num_res: int = 2):
        super().__init__()
        self.down = ConvGNAct(cin, cout, k=3, s=2, p=1)
        self.res = nn.Sequential(*[ResBlockGN(cout) for _ in range(int(num_res))])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down(x)
        return self.res(x)


class UpGN(nn.Module):
    def __init__(self, cin: int, cout: int, num_res: int = 2):
        super().__init__()
        self.up = nn.ConvTranspose2d(cin, cout, kernel_size=2, stride=2, bias=False)
        self.gn = nn.GroupNorm(_choose_gn_groups(cout), cout, eps=1e-5, affine=True)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.fuse = nn.Conv2d(cout * 2, cout, 3, padding=1, bias=False)
        self.fuse_gn = nn.GroupNorm(_choose_gn_groups(cout), cout, eps=1e-5, affine=True)
        self.res = nn.Sequential(*[ResBlockGN(cout) for _ in range(int(num_res))])

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.act(self.gn(self.up(x)))
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.act(self.fuse_gn(self.fuse(x)))
        return self.res(x)


def local_correlation(f0: torch.Tensor, f1: torch.Tensor, radius: int = 4) -> torch.Tensor:
    """
    Local cost volume correlation between f0 and f1 at the SAME resolution.
    f0,f1: [B,C,H,W]
    return: [B,(2r+1)^2,H,W], normalized by sqrt(C).
    """
    assert f0.ndim == 4 and f1.ndim == 4
    B, C, H, W = f0.shape
    r = int(radius)
    k = 2 * r + 1
    f1_unf = F.unfold(f1, kernel_size=k, padding=r)
    f1_unf = f1_unf.view(B, C, k * k, H, W)
    corr = (f0.unsqueeze(2) * f1_unf).sum(dim=1)  # [B,k*k,H,W]
    corr = corr / math.sqrt(float(C) + 1e-6)
    return corr


class CorrEncoder(nn.Module):
    """
    Lightweight shared-weight encoder to produce correlation features at 1/8 scale.
    Input: 2ch (I, grad) in [0,1] range.
    """
    def __init__(self, in_ch: int = 2, base_ch: int = 24, num_res: int = 1):
        super().__init__()
        c1 = base_ch
        c2 = base_ch * 2
        c3 = base_ch * 4

        self.stem = nn.Sequential(
            ConvGNAct(in_ch, c1, 3, 1, 1),
            ResBlockGN(c1),
        )
        self.down1 = DownGN(c1, c2, num_res=num_res)  # 1/2
        self.down2 = DownGN(c2, c3, num_res=num_res)  # 1/4
        self.down3 = DownGN(c3, c3, num_res=num_res)  # 1/8

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        return x


class StudentCorrUNetGN(nn.Module):
    def __init__(
            self,
            in_ch: int = 5,
            base_ch: int = 32,
            num_res: int = 2,
            max_disp: float = 60.0,
            out_act: str = "atan",
            corr_radius: int = 4,
            corr_base_ch: int = 24,
    ):
        super().__init__()
        self.max_disp = float(max_disp)
        self.out_act = str(out_act).lower().strip()
        self.corr_radius = int(corr_radius)

        c1 = base_ch
        c2 = base_ch * 2
        c3 = base_ch * 4
        c4 = base_ch * 8
        c5 = base_ch * 16

        self.stem = nn.Sequential(
            ConvGNAct(in_ch, c1, 3, 1, 1),
            ResBlockGN(c1),
        )
        self.down1 = DownGN(c1, c2, num_res=num_res)
        self.down2 = DownGN(c2, c3, num_res=num_res)
        self.down3 = DownGN(c3, c4, num_res=num_res)  # -> 1/8
        self.down4 = DownGN(c4, c5, num_res=num_res)  # -> 1/16

        self.bottleneck = nn.Sequential(
            ResBlockGN(c5),
            ResBlockGN(c5),
        )

        self.up4 = UpGN(c5, c4, num_res=num_res)
        self.up3 = UpGN(c4, c3, num_res=num_res)
        self.up2 = UpGN(c3, c2, num_res=num_res)
        self.up1 = UpGN(c2, c1, num_res=num_res)

        self.corr_enc = CorrEncoder(in_ch=2, base_ch=corr_base_ch, num_res=1)

        cv_ch = (2 * self.corr_radius + 1) ** 2
        self.fuse_corr = nn.Sequential(
            nn.Conv2d(c4 + 2 * cv_ch, c4, 1, bias=False),
            nn.GroupNorm(_choose_gn_groups(c4), c4, eps=1e-5, affine=True),
            nn.LeakyReLU(0.1, inplace=True),
            ResBlockGN(c4),
            ResBlockGN(c4),
        )

        self.head_f01 = nn.Conv2d(c1, 2, 3, padding=1)
        self.head_f10 = nn.Conv2d(c1, 2, 3, padding=1)

        for m in [self.head_f01, self.head_f10]:
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def _flow_act(self, raw: torch.Tensor) -> torch.Tensor:
        if self.out_act == "tanh":
            return torch.tanh(raw) * self.max_disp
        if self.out_act == "softsign":
            return (raw / (1.0 + raw.abs())) * self.max_disp
        return (2.0 / math.pi) * torch.atan(raw) * self.max_disp

    def forward(self, x: torch.Tensor, dcond: torch.Tensor | None = None):
        s1 = self.stem(x)
        s2 = self.down1(s1)
        s3 = self.down2(s2)
        s4 = self.down3(s3)

        I0g = torch.cat([x[:, 0:1, ...], x[:, 3:4, ...]], dim=1)
        I1g = torch.cat([x[:, 1:2, ...], x[:, 4:5, ...]], dim=1)
        f0 = self.corr_enc(I0g)
        f1 = self.corr_enc(I1g)
        cv01 = local_correlation(f0, f1, radius=self.corr_radius)
        cv10 = local_correlation(f1, f0, radius=self.corr_radius)

        s4f = self.fuse_corr(torch.cat([s4, cv01, cv10], dim=1))

        s5 = self.down4(s4f)
        b = self.bottleneck(s5)

        d4 = self.up4(b, s4f)
        d3 = self.up3(d4, s3)
        d2 = self.up2(d3, s2)
        d1 = self.up1(d2, s1)

        f01 = self._flow_act(self.head_f01(d1))
        f10 = self._flow_act(self.head_f10(d1))
        return f01, f10
