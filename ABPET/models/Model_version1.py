"""
Model architecture for Amyloid PET Centiloid Prediction.

Baseline 3D CNN with tracer conditioning.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Sequence, Tuple, Type, Union

class ConvBlock(nn.Module):
    """Conv3d -> BatchNorm -> ReLU -> MaxPool."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
        )

    def forward(self, x):
        return self.block(x)

class BasicBlock3D(nn.Module):
    """Two 3×3×3 convs  –  expansion = 1  (ResNet-10/18/34)."""
    expansion: int = 1

    def __init__(self, in_ch: int, out_ch: int,
                 stride: int = 1, downsample: Optional[nn.Module] = None) -> None:
        super().__init__()
        self.conv1      = nn.Conv3d(in_ch,  out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1        = nn.BatchNorm3d(out_ch)
        self.conv2      = nn.Conv3d(out_ch, out_ch, 3, stride=1,      padding=1, bias=False)
        self.bn2        = nn.BatchNorm3d(out_ch)
        self.relu       = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)

class Bottleneck3D(nn.Module):
    """1×1 → 3×3×3 → 1×1  bottleneck  –  expansion = 4  (ResNet-50/101)."""
    expansion: int = 4

    def __init__(self, in_ch: int, out_ch: int,
                 stride: int = 1, downsample: Optional[nn.Module] = None) -> None:
        super().__init__()
        self.conv1      = nn.Conv3d(in_ch,                    out_ch, 1, bias=False)
        self.bn1        = nn.BatchNorm3d(out_ch)
        self.conv2      = nn.Conv3d(out_ch,                   out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn2        = nn.BatchNorm3d(out_ch)
        self.conv3      = nn.Conv3d(out_ch, out_ch * self.expansion, 1, bias=False)
        self.bn3        = nn.BatchNorm3d(out_ch * self.expansion)
        self.relu       = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)

class ResNet3D(nn.Module):
    """
    3-D ResNet with tracer-conditioned regression output.

                         ┌─────────────────────────────────┐
     volume (1,128³) ──► │  ResNet backbone  (stem+4 stages)│ ──► global avg pool ──► feat (512,)
                         └─────────────────────────────────┘              │
     tracer_idx ──────► Embedding(4, embed_dim) ──► tracer_feat ──────── cat
                                                                           │
                                                              ┌────────────┘
                                                              ▼
                                                   Regression MLP ──► scalar float

    Parameters
    ----------
    block        BasicBlock3D | Bottleneck3D
    layers       [n1, n2, n3, n4] blocks per stage
    in_channels  input image channels (1 for mono 3-D volume)
    num_tracers  number of tracer classes (4)
    embed_dim    tracer embedding dimension (64)
    dropout      dropout in regression head (0.5)
    """

    def __init__(
        self,
        block:       Type[Union[BasicBlock3D, Bottleneck3D]],
        layers:      List[int],
        in_channels: int   = 1,
        num_tracers: int   = 4,
        embed_dim:   int   = 64,
        dropout:     float = 0.5,
        mean_centiloid: float = 0.0
    ) -> None:
        super().__init__()
        self._in_ch = 64

        # ── Stem  128³ ──stride-2──► 64³ ──maxpool──► 32³ ────────────────────
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
        )

        # ── Residual stages ───────────────────────────────────────────────────
        #   32³ →  32³   (stride 1)
        #   32³ →  16³   (stride 2)
        #   16³ →   8³   (stride 2)
        #    8³ →   4³   (stride 2)
        self.layer1 = self._make_layer(block,  64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool  = nn.AdaptiveAvgPool3d(1)
        self.feat_dim = 512 * block.expansion   # 512 or 2048

        # ── Tracer embedding  (index → dense vector) ──────────────────────────
        self.tracer_embed = nn.Embedding(num_tracers, embed_dim)

        # ── Regression head  (feat + embed → scalar) ─────────────────────────
        fused_dim = self.feat_dim + embed_dim
        self.reg_head = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),    # raw float — no activation
        )

        self._init_weights()

    # ── layer builder ─────────────────────────────────────────────────────────

    def _make_layer(
        self,
        block:      Type[Union[BasicBlock3D, Bottleneck3D]],
        out_ch:     int,
        num_blocks: int,
        stride:     int = 1,
    ) -> nn.Sequential:
        planes     = out_ch * block.expansion
        downsample = None
        if stride != 1 or self._in_ch != planes:
            downsample = nn.Sequential(
                nn.Conv3d(self._in_ch, planes, 1, stride=stride, bias=False),
                nn.BatchNorm3d(planes),
            )
        layers      = [block(self._in_ch, out_ch, stride=stride, downsample=downsample)]
        self._in_ch = planes
        layers     += [block(self._in_ch, out_ch) for _ in range(1, num_blocks)]
        return nn.Sequential(*layers)

    # ── weight init ───────────────────────────────────────────────────────────

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm3d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        volume:     torch.Tensor,   # (B, 1, 128, 128, 128)
        tracer_idx: torch.Tensor,   # (B,)  long  ∈ {0,1,2,3}
    ) -> torch.Tensor:              # (B,)  float
        """
        Parameters
        ----------
        volume     : FloatTensor  (B, 1, 128, 128, 128)
        tracer_idx : LongTensor   (B,)

        Returns
        -------
        value : FloatTensor (B,)  –  raw regression output (z-normalised space)
                Denormalise with:  value * VALUE_STD + VALUE_MEAN
        """
        # CNN backbone
        x = self.stem(volume)       # (B,  64, 32, 32, 32)
        x = self.layer1(x)          # (B,  64, 32, 32, 32)
        x = self.layer2(x)          # (B, 128, 16, 16, 16)
        x = self.layer3(x)          # (B, 256,  8,  8,  8)
        x = self.layer4(x)          # (B, 512,  4,  4,  4)
        x = self.avgpool(x)         # (B, 512,  1,  1,  1)
        x = x.flatten(1)            # (B, 512)

        # Tracer embedding
        t = self.tracer_embed(tracer_idx)   # (B, embed_dim)

        # Fuse and regress
        fused = torch.cat([x, t], dim=1)    # (B, 512 + embed_dim)
        out   = self.reg_head(fused)        # (B, 1)
        return out.squeeze(1)               # (B,)

class BaselineCNN(nn.Module):
    """
    Simple 3D CNN for centiloid regression.

    Architecture:
        4 conv blocks:  1 -> 32 -> 64 -> 128 -> 256  (each halves spatial dims)
        Global average pool -> 256-dim feature vector
        Concatenate with tracer embedding (8-dim)
        MLP head -> scalar prediction

    Input:  (B, 1, 128, 128, 128)
    Output: (B,)  predicted centiloid scores
    """

    def __init__(self, num_tracers: int, emb_dim: int = 8, mean_centiloid: float = 0.0):
        super().__init__()

        self.encoder = nn.Sequential(
            ConvBlock(1, 32),    # -> (B, 32, 64, 64, 64)
            ConvBlock(32, 64),   # -> (B, 64, 32, 32, 32)
            ConvBlock(64, 128),  # -> (B, 128, 16, 16, 16)
            ConvBlock(128, 256), # -> (B, 256, 8, 8, 8)
        )

        self.gap = nn.AdaptiveAvgPool3d(1)  # -> (B, 256, 1, 1, 1)

        self.tracer_emb = nn.Embedding(num_tracers, emb_dim)

        self.head = nn.Sequential(
            nn.Linear(256 + emb_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

        # Initialize final bias to dataset mean for faster convergence
        nn.init.constant_(self.head[-1].bias, mean_centiloid)

    def forward(self, x, tracer_idx):
        features = self.encoder(x)
        features = self.gap(features).flatten(1)             # (B, 256)
        tracer_features = self.tracer_emb(tracer_idx)        # (B, emb_dim)
        combined = torch.cat([features, tracer_features], 1) # (B, 264)
        return self.head(combined).squeeze(1)                # (B, )
