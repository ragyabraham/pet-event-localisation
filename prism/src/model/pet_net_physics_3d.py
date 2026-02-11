import torch
import torch.nn as nn
from typing import Literal
from src.model.utils import ResidualBlock3D, make_norm, circular_pad_w, CircularSpatialSoftArgmax3d, CBAM3D


# ==========================================
# 2. UPDATED MODEL
# ==========================================
class PetNetPhysics3D(nn.Module):
    """
    Physics-Informed 3D PetNet with CBAM and Dual-Head Architecture.
    """

    def __init__(
            self,
            in_channels: int = 3,
            base_channels: int = 16,
            norm: Literal["group", "batch", "instance"] = "batch",
            groups_for_gn: int = 8,
            dropout3d: float = 0.10,
            fc_dropout: float = 0.40,
    ):
        super().__init__()

        # --- Stem ---
        extended_in_channels = in_channels + 3
        self.conv_in = nn.Conv3d(
            extended_in_channels,
            base_channels,
            kernel_size=(1, 3, 3),  # 3x3 Kernel
            stride=1,
            padding=(0, 1, 0),  # We handle padding manually
            bias=False,
        )
        self.n_in = make_norm(norm, base_channels, groups_for_gn)
        self.act = nn.ReLU(inplace=True)
        self.drop3d = nn.Dropout3d(dropout3d) if dropout3d > 0 else nn.Identity()

        # --- Backbone ---
        self.layer1 = ResidualBlock3D(base_channels, base_channels * 2, stride=(1, 2, 2), norm=norm)
        self.layer2 = ResidualBlock3D(base_channels * 2, base_channels * 4, stride=(1, 2, 2), norm=norm)

        # Layer 3 (Semantic Sweet Spot)
        self.layer3 = ResidualBlock3D(base_channels * 4, base_channels * 8, stride=(1, 2, 2), norm=norm)

        # ATTENTION: Replaces the need for massive padding
        self.cbam = CBAM3D(channels=base_channels * 8)

        self.layer4 = ResidualBlock3D(base_channels * 8, base_channels * 16, stride=(1, 2, 2), norm=norm)

        # --- Temporal Attention ---
        self.temporal_gate = nn.Sequential(
            nn.Conv3d(base_channels * 16, base_channels * 16, kernel_size=1),
            nn.Sigmoid()
        )

        # --- HEAD (Dual Heatmaps) ---
        # 2 Channels: One for P1, One for P2
        self.final_conv = nn.Conv3d(base_channels * 16, 2, kernel_size=1)

        # Returns 8 coords total (2 points * 4 vals)
        self.soft_argmax = CircularSpatialSoftArgmax3d()

        # Maps 8 inputs to 7 outputs
        self.head = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 7)  # Output: x1,y1,z1, x2,y2,z2, class
        )

        self._initialize_weights()

    @staticmethod
    def _generate_positional_encodings(x: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = x.shape
        device = x.device
        theta = torch.linspace(0, 2 * torch.pi, W, device=device)
        sin_t = torch.sin(theta).view(1, 1, 1, 1, W).expand(B, 1, T, H, W)
        cos_t = torch.cos(theta).view(1, 1, 1, 1, W).expand(B, 1, T, H, W)
        z = torch.linspace(-1, 1, H, device=device)
        z_map = z.view(1, 1, 1, H, 1).expand(B, 1, T, H, W)
        return torch.cat([sin_t, cos_t, z_map], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        pos_enc = self._generate_positional_encodings(x)
        x = torch.cat([x, pos_enc], dim=1)

        # CORRECT: Pad 1 for 3x3 kernel. CBAM handles the global context.
        x = circular_pad_w(x, pad=1)
        x = self.conv_in(x)
        x = self.n_in(x)
        x = self.act(x)
        x = self.drop3d(x)

        x = self.layer1(x)
        x = self.layer2(x)

        # Apply Layer 3 -> Attention -> Layer 4
        x = self.layer3(x)

        # Apply Attention
        x = self.cbam(x)

        x = self.layer4(x)
        weights = self.temporal_gate(x)
        x = x * weights

        # Dual Head Prediction
        heatmap = self.final_conv(x)

        # Output: (B, 8) -> [t1, h1, sin1, cos1, t2, h2, sin2, cos2]
        coords = self.soft_argmax(heatmap)

        return self.head(coords)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)