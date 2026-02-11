import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal
from src.model.utils import ResidualBlock3D, make_norm, circular_pad_w, CircularSpatialSoftArgmax3d, CBAM3D
from src.utils.config import Geometry  # Ensure Geometry is imported


class PetNetPhysics3DPolar(nn.Module):
    """
    Physics-Informed 3D PetNet with Hard Geometric Constraints.
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

        # --- Geometry Constraints ---
        PADDING = 5.0
        self.r_min = Geometry.R_INNER - PADDING
        self.r_max = Geometry.R_OUTER + PADDING
        self.z_max = Geometry.Z_HALF + PADDING  # Pad Z too just in case

        self.r_diff = self.r_max - self.r_min

        # --- Stem ---
        extended_in_channels = in_channels + 3
        self.conv_in = nn.Conv3d(
            extended_in_channels,
            base_channels,
            kernel_size=(1, 3, 3),
            stride=1,
            padding=(0, 1, 0),
            bias=False,
        )
        self.n_in = make_norm(norm, base_channels, groups_for_gn)
        self.act = nn.ReLU(inplace=True)
        self.drop3d = nn.Dropout3d(dropout3d) if dropout3d > 0 else nn.Identity()

        # --- Backbone ---
        self.layer1 = ResidualBlock3D(base_channels, base_channels * 2, stride=(1, 2, 2), norm=norm)
        self.layer2 = ResidualBlock3D(base_channels * 2, base_channels * 4, stride=(1, 2, 2), norm=norm)
        self.layer3 = ResidualBlock3D(base_channels * 4, base_channels * 8, stride=(1, 2, 2), norm=norm)

        self.cbam = CBAM3D(channels=base_channels * 8)
        self.layer4 = ResidualBlock3D(base_channels * 8, base_channels * 16, stride=(1, 2, 2), norm=norm)

        self.temporal_gate = nn.Sequential(
            nn.Conv3d(base_channels * 16, base_channels * 16, kernel_size=1),
            nn.Sigmoid()
        )

        self.final_conv = nn.Conv3d(base_channels * 16, 2, kernel_size=1)
        self.soft_argmax = CircularSpatialSoftArgmax3d()

        # --- UPDATED HEAD ---
        # We need 9 outputs now instead of 7:
        # P1: (r_logit, u, v, z_logit) -> 4 values
        # P2: (r_logit, u, v, z_logit) -> 4 values
        # Class: 1 value
        self.head = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 9)
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

    def _apply_hard_constraints(self, raw_p: torch.Tensor) -> torch.Tensor:
        """
        Converts raw logits into strictly bounded Cartesian coordinates.
        Input: (B, 4) -> [r_logit, u, v, z_logit]
        Output: (B, 3) -> [x, y, z]
        """
        # 1. Radius: Sigmoid bounds output to [0, 1], then scale to [R_in, R_out]
        r_norm = torch.sigmoid(raw_p[:, 0])
        r = r_norm * self.r_diff + self.r_min

        # 2. Angle: Normalise (u, v) to get unit vector direction
        # We add epsilon to prevent division by zero
        uv = raw_p[:, 1:3]
        uv_norm = F.normalize(uv, p=2, dim=1, eps=1e-8)

        # 3. Z-Axis: Tanh bounds output to [-1, 1], then scale to [-Z_max, +Z_max]
        z_norm = torch.tanh(raw_p[:, 3])
        z = z_norm * self.z_max

        # 4. Convert Polar to Cartesian
        x = r * uv_norm[:, 0]
        y = r * uv_norm[:, 1]

        return torch.stack([x, y, z], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        pos_enc = self._generate_positional_encodings(x)
        x = torch.cat([x, pos_enc], dim=1)

        x = circular_pad_w(x, pad=1)
        x = self.conv_in(x)
        x = self.n_in(x)
        x = self.act(x)
        x = self.drop3d(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.cbam(x)
        x = self.layer4(x)
        weights = self.temporal_gate(x)
        x = x * weights

        heatmap = self.final_conv(x)
        coords = self.soft_argmax(heatmap)

        # --- Get Raw Output ---
        raw_out = self.head(coords)  # Shape (B, 9)

        # Split into components
        p1_raw = raw_out[:, 0:4]  # [r, u, v, z] for P1
        p2_raw = raw_out[:, 4:8]  # [r, u, v, z] for P2
        cls_logits = raw_out[:, 8:]  # Class

        # --- Apply Hard Constraints ---
        p1_xyz = self._apply_hard_constraints(p1_raw)
        p2_xyz = self._apply_hard_constraints(p2_raw)

        # Recombine: x1,y1,z1, x2,y2,z2, class
        return torch.cat([p1_xyz, p2_xyz, cls_logits], dim=1)

    def _initialize_weights(self):
        # 1. Standard Kaiming Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)