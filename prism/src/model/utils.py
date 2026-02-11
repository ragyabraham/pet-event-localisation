from typing import Literal, Tuple
from src.utils.config import Geometry
from torch import nn as nn
from torch.nn import functional as F
import torch.distributed as dist
import torch


class CircularSpatialSoftArgmax3d(nn.Module):
    """
    Calculates the expected coordinate (Center of Mass) of the feature map.
    Handles the Circular Dimension (W) by calculating Sin/Cos expectations.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        # x: (B, C, T, H, W)
        B, C, T, H, W = x.shape

        # Softmax to turn features into a probability distribution
        flat_x = x.view(B, C, -1)
        softmax_attention = F.softmax(flat_x, dim=2)  # (B, C, N)

        device = x.device

        # 1. Linear Coordinates for T (Depth) and H (Height/Z) -> Range [-1, 1]
        rng_t = torch.linspace(-1, 1, T, device=device)
        rng_h = torch.linspace(-1, 1, H, device=device)

        # 2. Circular Coordinates for W (Arc/Theta) -> Range [0, 2pi]
        rng_w = torch.linspace(0, 2 * torch.pi, W, device=device)

        # Generate Grids
        # grid_t, grid_h are simple linear gradients repeated
        # grid_theta is the angle map
        grid_t, grid_h, grid_theta = torch.meshgrid(rng_t, rng_h, rng_w, indexing='ij')

        # 3. Convert Theta to vector components (Sin/Cos) to preserve circularity
        grid_sin = torch.sin(grid_theta)
        grid_cos = torch.cos(grid_theta)

        # Stack 4 coordinate maps: (Depth, Height, Sin_Arc, Cos_Arc)
        # Shape: (4, T*H*W)
        coords = torch.stack([grid_t, grid_h, grid_sin, grid_cos], dim=0).view(4, -1)

        # Calculate Expectation: Sum(Prob * Coord)
        # (B, C, N) x (4, N)^T -> (B, C, 4)
        expected_coords = torch.bmm(
            softmax_attention,
            coords.transpose(0, 1).unsqueeze(0).expand(B, -1, -1)
        )

        # Output: (B, C*4) -> [t, h, sin_theta, cos_theta] for each channel
        return expected_coords.view(B, -1)

def freeze_backbone(model, freeze=True):
    """
    Freezes CNN weights but handles Batch Norm carefully.
    """

    status = "Freezing" if freeze else "Unfreezing"
    if dist.is_initialized() and dist.get_rank() == 0:
        print(f"❄️ {status} Backbone...")
    elif not dist.is_initialized():
        print(f"❄️ {status} Backbone...")

    for name, module in model.named_modules():
        # If it's a Batch Norm layer, we generally want to keep it trainable
        # or at least running (track_running_stats=True) to adapt to the new domain.
        if isinstance(module, (nn.BatchNorm3d, nn.GroupNorm, nn.InstanceNorm3d)):
            # OPTION A (Recommended): Keep BN fully active
            module.train()
            for param in module.parameters():
                param.requires_grad = True  # Allow affine parameters (scale/shift) to adapt
            continue

    for name, param in model.named_parameters():
        is_head = any(x in name for x in ["head", "fc_", "linear"])
        is_bn = "norm" in name or "bn" in name or "n_in" in name  # Check param names too

        if is_head or is_bn:
            param.requires_grad = True  # Keep Heads AND Norms trainable
        else:
            param.requires_grad = not freeze  # Freeze Conv weights only


def _cart_to_polar_pixels(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor):
    theta = torch.atan2(y, x)
    two_pi = torch.as_tensor(2.0 * torch.pi, device=theta.device, dtype=theta.dtype)
    theta_wrapped = torch.where(theta < 0, theta + two_pi, theta)
    theta_pix = (theta_wrapped / two_pi) * Geometry.CIRCUMFERENTIAL

    # FIXED Z AXIS MAPPING
    z_half = torch.as_tensor(Geometry.Z_HALF, device=z.device, dtype=z.dtype)
    z_pix = ((z + z_half) / (2.0 * z_half)) * Geometry.AXIAL

    return theta_pix, z_pix


def convert_targets_to_polar(target: torch.Tensor) -> torch.Tensor:
    """
    target: (..., 6) with [x1, y1, z1, x2, y2, z2]
    returns: (..., 4) with [theta1_pix, z1_pix, theta2_pix, z2_pix]

    For single interaction events (where x1,y1,z1 == x2,y2,z2), the second point
    (theta2_pix, z2_pix) is set to impossible values (-1, -1) to indicate this
    to the model.
    """
    if target.size(-1) != 6:
        raise ValueError(f"Expected target last dim = 6, got {target.size(-1)}")

    # Split
    x1, y1, z1, x2, y2, z2 = torch.unbind(target, dim=-1)

    # Calculate Radius, Theta, Z
    r1 = torch.sqrt(x1 ** 2 + y1 ** 2)
    theta1, z1p = _cart_to_polar_pixels(x1, y1, z1)

    r2 = torch.sqrt(x2 ** 2 + y2 ** 2)
    theta2, z2p = _cart_to_polar_pixels(x2, y2, z2)

    # Stack 6 coordinates: [r1, theta1, z1, r2, theta2, z2]
    out = torch.stack([r1, theta1, z1p, r2, theta2, z2p], dim=-1)
    return out


def _polar_to_cart_pixels(theta: torch.Tensor, r: torch.Tensor):
    x = r * torch.cos(theta)
    y = r * torch.sin(theta)

    return x, y


def convert_target_to_cartesian(target: torch.Tensor):
    if target.size(-1) != 6:
        raise ValueError(f"Expected target last dim = 6, got {target.size(-1)}")

    r1, theta1, z1, r2, theta2, z2 = torch.unbind(target, dim=-1)

    x1, y1 = _polar_to_cart_pixels(theta1, r1)
    x2, y2 = _polar_to_cart_pixels(theta2, r2)

    out = torch.stack([x1, y1, z1, x2, y2, z2], dim=-1)

    return out


# utility function to count the number of parameters in a model
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"Total parameters": total, "Trainable parameters": trainable}


# Utility function to add circular padding along the azimuthal angle which corresponds to the circumference of the cylinder
def circular_pad_w(x: torch.Tensor, pad: int = 1) -> torch.Tensor:
    """
    Apply circular padding only along the last dimension (W/azimuth)
    of a 5D tensor (N, C, T, H, W).
    """
    # F.pad pads in reverse order:
    # (W_left, W_right, H_left, H_right, D_left, D_right)
    return F.pad(x, (pad, pad, 0, 0, 0, 0), mode="circular")


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        # shape: (B, 1, 1, 1, 1) to broadcast across C, D, H, W
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(
            shape, dtype=x.dtype, device=x.device
        )
        random_tensor.floor_()  # binarize
        return x / keep_prob * random_tensor


class ResidualBlock3D(nn.Module):
    """
    A 3D residual block with configurable normalisation and stochastic depth.
    Uses (3,3,3) kernels and an optional downsampling via stride=(1,2,2)
    by default.
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: Tuple[int, int, int] = (1, 2, 2),
            norm: Literal["group", "batch", "instance"] = "batch",
            drop_path: float = 0.0,
            groups_for_gn: int = 8,
            circular_pad_w_enable: bool = True,
    ):
        super().__init__()

        self.conv1 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=(1, 1, 0),
            bias=False,
        )
        self.n1 = make_norm(norm, out_channels, groups_for_gn)
        self.act = nn.ReLU()

        self.conv2 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=(1, 1, 0),
            bias=False,
        )
        self.n2 = make_norm(norm, out_channels, groups_for_gn)

        self.shortcut = None
        if stride != (1, 1, 1) or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                make_norm(norm, out_channels, groups_for_gn),
            )

        self.drop_path = (
            DropPath(drop_prob=drop_path) if drop_path > 0.0 else nn.Identity()
        )
        self.use_circular_pad_w = circular_pad_w_enable

    def forward(self, x):
        identity = x

        out = self.conv1(circular_pad_w(x) if self.use_circular_pad_w else x)
        out = self.n1(out)
        out = self.act(out)

        out = self.conv2(
            circular_pad_w(out) if self.use_circular_pad_w else out
        )
        out = self.n2(out)

        if self.shortcut is not None:
            identity = self.shortcut(identity)

        out = identity + self.drop_path(out)
        out = self.act(out)
        return out


def make_norm(
        norm: Literal["group", "batch", "instance"],
        num_channels: int,
        groups: int = 8,
):
    if norm == "group":
        # clamp groups so they divide num_channels
        g = max(1, min(groups, num_channels))
        while num_channels % g != 0 and g > 1:
            g -= 1
        return nn.GroupNorm(g, num_channels)
    elif norm == "batch":
        return nn.BatchNorm3d(num_channels)
    else:
        return nn.InstanceNorm3d(num_channels, affine=True)


# ==========================================
# 1. CBAM MODULES (Add these classes)
# ==========================================
class ChannelAttention3D(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        # Shared MLP
        hidden_planes = max(in_planes // ratio, 1)
        self.mlp = nn.Sequential(
            nn.Conv3d(in_planes, hidden_planes, 1, bias=False),
            nn.ReLU(),
            nn.Conv3d(hidden_planes, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention3D(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        # Standard kernel size for spatial attention is usually 7
        padding = 3 if kernel_size == 7 else 1

        # Compresses channels to 2 (Max + Avg) -> Convolves to 1
        self.conv1 = nn.Conv3d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Channel-wise pooling
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)

        return self.sigmoid(self.conv1(x_cat))


class CBAM3D(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_gate = ChannelAttention3D(channels, reduction)
        self.spatial_gate = SpatialAttention3D(kernel_size)

    def forward(self, x):
        # 1. Channel Attention: "What" to look for (Scattered vs True features)
        x_out = self.channel_gate(x) * x
        # 2. Spatial Attention: "Where" to look (Suppress scattered tails)
        x_out = self.spatial_gate(x_out) * x_out
        return x_out


if __name__ == "__main__":
    from src.model.pet_net_improved_3d import PetNetImproved3D

    petnet_parameters = count_parameters(PetNetImproved3D())
    print(f"PetNetImproved3D: {petnet_parameters}")
