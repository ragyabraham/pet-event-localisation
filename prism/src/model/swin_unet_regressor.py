from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


# -----------------------------------------------------------------------
# 1. Helper Blocks (Convolution, Upsampling, GRU, Heads)
# -----------------------------------------------------------------------

class ConvBNAct(nn.Module):
    def __init__(
            self,
            in_ch: int,
            out_ch: int,
            k: int = 3,
            s: int = 1,
            p: int | None = None,
            circular_padding: bool = False,
    ):
        super().__init__()
        if p is None:
            p = (k - 1) // 2
        self.circular_padding = circular_padding
        self.pad_h = p
        self.pad_w = p
        # Manual padding for circular support
        conv_padding = 0 if circular_padding else p
        self.conv = nn.Conv2d(
            in_ch, out_ch, kernel_size=k, stride=s, padding=conv_padding, bias=False
        )
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.circular_padding and self.pad_w > 0:
            # Circular padding on Width (circumference), Constant on Height (axial)
            x = F.pad(x, (self.pad_w, self.pad_w, 0, 0), mode='circular')
            x = F.pad(x, (0, 0, self.pad_h, self.pad_h), mode='constant', value=0)
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, circular_padding: bool = False):
        super().__init__()
        self.upsample = nn.Upsample(
            scale_factor=2.0, mode="bilinear", align_corners=False
        )
        self.conv1 = ConvBNAct(in_ch + skip_ch, out_ch, k=3, circular_padding=circular_padding)
        self.conv2 = ConvBNAct(out_ch, out_ch, k=3, circular_padding=circular_padding)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        # Center crop skip if dimensions mismatch slightly
        if x.shape[-2] != skip.shape[-2] or x.shape[-1] != skip.shape[-1]:
            dh = skip.shape[-2] - x.shape[-2]
            dw = skip.shape[-1] - x.shape[-1]
            skip = skip[
                :,
                :,
                (dh // 2):(skip.shape[-2] - (dh - dh // 2)),
                (dw // 2):(skip.shape[-1] - (dw - dw // 2)),
            ]
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class TemporalGRU(nn.Module):
    """
    Aggregates the temporal burst dimension (T) into a single feature vector.
    Input: (B, T, F) -> Output: (B, F)
    """

    def __init__(
            self,
            feat_dim: int,
            hidden: int = 384,  # Adjusted default
            num_layers: int = 1,
            bidirectional: bool = False,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=feat_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.out_dim = hidden * (2 if bidirectional else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        out, h = self.gru(x)
        h_last = h[-1]
        return h_last


class MLPHead3D(nn.Module):
    """
    Regresses 3 Cartesian coordinates (x, y, z) from the feature vector.
    """

    def __init__(
            self,
            in_dim: int,
            hidden: int = 256,
            dropout_p: float = 0.3,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden, 3),  # Output: x, y, z
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # (B, 3)


# -----------------------------------------------------------------------
# 2. Frame Encoder
# -----------------------------------------------------------------------

class SwinUNetV2EncoderDecoder1C(nn.Module):
    """
    Encodes a single frame (1 channel) into a feature vector.
    Uses SwinV2 backbone with U-Net decoder to preserve spatial details.
    """

    def __init__(
            self,
            backbone_name: str = "swinv2_tiny_window16_256",
            pretrained: bool = True,
            decoder_out_channels: int = 256,
            embed_dim: int = 384,  # <--- CHANGED from 512 to 384
            circular_padding: bool = True,
    ):
        super().__init__()
        self.circular_padding = circular_padding

        # Adapt 1-channel input (PET intensity) to 3-channel (ImageNet expectation)
        self.input_adapter = nn.Conv2d(1, 3, kernel_size=1, stride=1, padding=0, bias=True)
        with torch.no_grad():
            self.input_adapter.weight.fill_(0.0)
            # Initialise as averaging or repeating
            self.input_adapter.weight[:, 0].fill_(1.0)
            if self.input_adapter.bias is not None:
                self.input_adapter.bias.zero_()

        # Encoder: Swin Transformer V2
        self.encoder = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),
        )
        chs = self.encoder.feature_info.channels()

        # Input resolution handling
        self._expected_hw = (256, 256)

        # Neck
        self.neck = ConvBNAct(chs[-1], decoder_out_channels, k=1, s=1, p=0)

        # Decoder (U-Net style)
        self.up3 = UpBlock(decoder_out_channels, chs[-2], decoder_out_channels // 2, circular_padding)
        self.up2 = UpBlock(decoder_out_channels // 2, chs[-3], decoder_out_channels // 4, circular_padding)
        self.up1 = UpBlock(decoder_out_channels // 4, chs[-4], decoder_out_channels // 8, circular_padding)

        self.out_conv = ConvBNAct(
            decoder_out_channels // 8, decoder_out_channels // 8, k=3, circular_padding=circular_padding
        )

        # Projection to embedding vector
        self.project = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(decoder_out_channels // 8, embed_dim),
        )
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, 1, H, W)
        x3 = self.input_adapter(x)

        # Interpolate to backbone size if needed
        th, tw = self._expected_hw
        if (x3.shape[-2], x3.shape[-1]) != (th, tw):
            x3 = F.interpolate(x3, size=(th, tw), mode="bilinear", align_corners=False)

        # Extract multi-scale features
        feats = self.encoder(x3)
        f0, f1, f2, f3 = feats

        # Ensure NCHW format
        def to_nchw(t, c):
            if t.ndim == 4 and t.shape[-1] == c:
                return t.permute(0, 3, 1, 2).contiguous()
            return t

        chs = self.encoder.feature_info.channels()
        f0, f1, f2, f3 = [to_nchw(f, c) for f, c in zip(feats, chs)]

        # Decode
        x = self.neck(f3)
        x = self.up3(x, f2)
        x = self.up2(x, f1)
        x = self.up1(x, f0)
        x = self.out_conv(x)

        # Pool to vector
        emb = self.project(x)
        return emb


# -----------------------------------------------------------------------
# 3. Main Regressor Model (With Adjusted Defaults)
# -----------------------------------------------------------------------

class SwinUNetV2Regressor(nn.Module):
    """
    Twin-stream Swin-UNet V2 for PET event localisation in Cartesian coordinates.

    Architecture:
      1. Twin Streams: Processes 'Inner' and 'Outer' detector images separately.
      2. Frame Encoder: Shared SwinV2+UNet extracts features per frame.
      3. Temporal Aggregation: GRU collapses the time dimension (burst).
      4. Heads:
         - Inner Head -> (x1, y1, z1)
         - Outer Head -> (x2, y2, z2)
         - Event Head -> Logit (True vs Random)

    Input: (B, T, 2, H, W)
    Output: (B, 7) -> [x1, y1, z1, x2, y2, z2, event_logit]
    """

    def __init__(
            self,
            embed_dim: int = 384,  # <--- CHANGED from 512 to 384
            temporal_hidden: int = 384,  # <--- CHANGED from 512 to 384
            backbone_name: str = "swinv2_tiny_window16_256",
            pretrained_backbone: bool = True,
            head_hidden: int = 256,
            feature_dropout_p: float = 0.2,
            head_dropout_p: float = 0.3,
            circular_padding: bool = True,
    ):
        super().__init__()

        # Shared Encoder
        self.frame_encoder = SwinUNetV2EncoderDecoder1C(
            backbone_name=backbone_name,
            pretrained=pretrained_backbone,
            decoder_out_channels=256,
            embed_dim=embed_dim,
            circular_padding=circular_padding,
        )

        # Temporal Aggregator
        self.temporal = TemporalGRU(
            feat_dim=self.frame_encoder.embed_dim,
            hidden=temporal_hidden,
            num_layers=1,
            bidirectional=False,
        )
        self.feature_dropout = nn.Dropout(p=feature_dropout_p)

        # 3D Cartesian Heads
        self.head_inner = MLPHead3D(
            in_dim=self.temporal.out_dim,
            hidden=head_hidden,
            dropout_p=head_dropout_p,
        )
        self.head_outer = MLPHead3D(
            in_dim=self.temporal.out_dim,
            hidden=head_hidden,
            dropout_p=head_dropout_p,
        )

        # Event Classification Head
        self.event_type_head = nn.Sequential(
            nn.Linear(self.temporal.out_dim * 2, head_hidden),
            nn.GELU(),
            nn.Dropout(head_dropout_p),
            nn.Linear(head_hidden, 1),
        )

    def _encode_stream(self, x_stream: torch.Tensor) -> torch.Tensor:
        """
        Encodes a single surface stream (e.g., inner or outer).
        Input: (B, T, 1, H, W)
        Output: (B, Hidden_Dim)
        """
        b, t, _, h, w = x_stream.shape

        # Chunk frames to save memory
        chunk_size = 4
        frame_features = []

        for i in range(0, b * t, chunk_size):
            end_idx = min(i + chunk_size, b * t)
            x_chunk = x_stream.reshape(b * t, 1, h, w)[i:end_idx]

            # Use checkpointing during training for memory efficiency
            if self.training:
                f_chunk = torch.utils.checkpoint.checkpoint(
                    self.frame_encoder, x_chunk, use_reentrant=False
                )
            else:
                f_chunk = self.frame_encoder(x_chunk)

            frame_features.append(f_chunk)

        f_flat = torch.cat(frame_features, dim=0)  # (B*T, D)
        f = f_flat.view(b, t, -1)  # (B, T, D)

        f_agg = self.temporal(f)
        f_agg = self.feature_dropout(f_agg)
        return f_agg

    def forward(self, burst: torch.Tensor) -> torch.Tensor:

        inner = burst[:, :, 0:1]
        outer = burst[:, :, 1:2]

        f_in = self._encode_stream(inner)
        f_out = self._encode_stream(outer)

        xyz1 = self.head_inner(f_in)
        xyz2 = self.head_outer(f_out)

        f_combined = torch.cat([f_in, f_out], dim=-1)
        event_type = self.event_type_head(f_combined)

        return torch.cat([xyz1, xyz2, event_type], dim=-1)