#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
phi2flux_deep.py
----------------
Deeper residual architecture for Phi→Flux:
- Spatial3DEncoderDeep: residual 3D conv stacks over (r, theta, toroidal), per time step
- TemporalDilatedTCN: dilated 1D residual TCN over time
- Phi2FluxDeep: end-to-end wrapper producing multi-horizon 3-channel outputs (Qi, Qe, Gamma)

Expected input x: [B, T, C=2, R, TH, TOR]
Output y:         [B, H, 3]
"""

from typing import Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class ResBlock3d(nn.Module):
    def __init__(self, ch: int, norm='bn', dropout: float = 0.0):
        super().__init__()
        if norm == 'bn':
            Norm = nn.BatchNorm3d
        elif norm == 'gn':
            g = max(1, min(8, ch))
            Norm = lambda c: nn.GroupNorm(g, c)
        else:
            raise ValueError("norm must be 'bn' or 'gn'")
        self.conv1 = nn.Conv3d(ch, ch, kernel_size=3, padding=1)
        self.norm1 = Norm(ch)
        self.conv2 = nn.Conv3d(ch, ch, kernel_size=3, padding=1)
        self.norm2 = Norm(ch)
        self.dropout = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.relu(h, inplace=True)
        h = self.dropout(h)
        h = self.conv2(h)
        h = self.norm2(h)
        return F.relu(h + x, inplace=True)

class Spatial3DEncoderDeep(nn.Module):
    def __init__(self, in_channels=2, base_channels=32, depth=8, norm='bn', dropout=0.0, proj_dim=256):
        super().__init__()
        assert depth >= 2
        if norm == 'bn':
            StemNorm = nn.BatchNorm3d
        else:
            StemNorm = lambda c: nn.GroupNorm(max(1, min(8, c)), c)
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, kernel_size=3, padding=1),
            StemNorm(base_channels),
            nn.ReLU(inplace=True),
        )
        blocks = []
        ch = base_channels
        for i in range(depth):
            if i > 0 and i % 3 == 0:
                new_ch = ch * 2
                if norm == 'bn':
                    blocks += [nn.Conv3d(ch, new_ch, 1), nn.BatchNorm3d(new_ch), nn.ReLU(inplace=True)]
                else:
                    blocks += [nn.Conv3d(ch, new_ch, 1), nn.GroupNorm(max(1, min(8, new_ch)), new_ch), nn.ReLU(inplace=True)]
                ch = new_ch
            blocks.append(ResBlock3d(ch, norm=norm, dropout=dropout))
        self.body = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(ch, proj_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout if dropout > 0 else 0.0),
        )
        self.out_dim = proj_dim

    def forward(self, x_vol):
        h = self.stem(x_vol)
        h = self.body(h)
        return self.head(h)

class TCNBlock(nn.Module):
    def __init__(self, ch: int, dilation: int, dropout: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv1d(ch, ch, kernel_size=3, dilation=dilation, padding=dilation)
        self.norm1 = nn.BatchNorm1d(ch)
        self.conv2 = nn.Conv1d(ch, ch, kernel_size=3, dilation=dilation, padding=dilation)
        self.norm2 = nn.BatchNorm1d(ch)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.relu(h, inplace=True)
        h = self.dropout(h)
        h = self.conv2(h)
        h = self.norm2(h)
        return F.relu(h + x, inplace=True)

class TemporalDilatedTCN(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, blocks: int = 3, dropout: float = 0.0):
        super().__init__()
        self.in_proj = nn.Conv1d(in_dim, hidden, kernel_size=1)
        layers = []
        for b in range(blocks):
            layers.append(TCNBlock(hidden, dilation=2**b, dropout=dropout))
        self.tcn = nn.Sequential(*layers)
        self.out_dim = hidden

    def forward(self, x_seq):
        x = x_seq.transpose(1, 2)   # [B, F, T]
        h = self.in_proj(x)
        h = self.tcn(h)
        return h.transpose(1, 2)    # [B, T, F]

class Phi2FluxDeep(nn.Module):
    def __init__(self, Tc: int, horizons: Sequence[int],
                 base_channels: int = 32, depth: int = 8,
                 tcn_channels: int = 128, tcn_blocks: int = 3,
                 dropout: float = 0.0, norm: str = 'bn'):
        super().__init__()
        self.Tc = Tc
        self.horizons = list(horizons)
        self.encoder = Spatial3DEncoderDeep(in_channels=2, base_channels=base_channels,
                                            depth=depth, norm=norm, dropout=dropout,
                                            proj_dim=tcn_channels)
        self.temporal = TemporalDilatedTCN(in_dim=tcn_channels, hidden=tcn_channels,
                                           blocks=tcn_blocks, dropout=dropout)
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(tcn_channels, tcn_channels//2),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout if dropout > 0 else 0.0),
                nn.Linear(tcn_channels//2, 3)
            ) for _ in self.horizons
        ])

    def forward(self, x):
        # x: [B, T, C=2, R, TH, TOR]
        B, T, C, R, TH, TOR = x.shape
        assert T == self.Tc, f"Expected T={self.Tc}, got {T}"
        x_flat = x.reshape(B*T, C, R, TH, TOR)
        feats = self.encoder(x_flat)     # [B*T, F]
        enc=feats
        feats = feats.view(B, T, -1)     # [B, T, F]
        seq = self.temporal(feats)       # [B, T, F]
        last = seq[:, -1, :]             # [B, F]
        outs = [head(last) for head in self.heads]
      
        if not hasattr(self, "_printed"):   # print only once per rank
           print(f"[{os.getpid()}] Φ2FluxDeep forward: input {tuple(x.shape)}", flush=True)
           self._printed = True

        # 3D encoder stage
        #if self.training and torch.rand(1).item() < 0.01:  # occasionally
        #print(f"[{os.getpid()}] Encoder output {tuple(enc.shape)}", flush=True)

        # temporal TCN stage
        #out = seq
        #if self.training and torch.rand(1).item() < 0.01:
        #print(f"[{os.getpid()}] TCN output {tuple(out.shape)}", flush=True)

        return torch.stack(outs, dim=1)  # [B, H, 3]

if __name__ == "__main__":
    B, T, C, R, TH, TOR = 2, 64, 2, 64, 1, 16
    x = torch.randn(B, T, C, R, TH, TOR)
    model = Phi2FluxDeep(Tc=T, horizons=[1,2,5], base_channels=32, depth=8, tcn_channels=128, tcn_blocks=3, dropout=0.2)
    y = model(x)
    print("Output shape:", y.shape)
