import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================
# 1. HELPER FUNCTIONS & BLOCKS
# (Included here so the code is self-contained)
# ==========================================

def circular_pad_w(x, pad=1):
    """
    Circular padding for the Width dimension (Theta in cylindrical coords).
    x: (B, C, T, H, W)
    Pads W dimension: (left, right, top, bottom, front, back)
    """
    return F.pad(x, (pad, pad, 0, 0, 0, 0), mode='circular')


def make_norm(norm_type, channels):
    if norm_type == "batch":
        return nn.BatchNorm3d(channels)
    elif norm_type == "instance":
        return nn.InstanceNorm3d(channels)
    else:
        return nn.Identity()


class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, norm="batch"):
        super().__init__()
        self.conv1 = nn.Conv3d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.n1 = make_norm(norm, out_channels)
        self.act = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.n2 = make_norm(norm, out_channels)

        # Shortcut connection to handle channel/stride changes
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                make_norm(norm, out_channels)
            )

    def forward(self, x):
        # Note: We assume circular padding is handled externally or via specific layers
        # for standard ResBlocks, usually we rely on standard zero-padding in the conv
        # unless explicitly wrapped. Here we use standard 3x3 padding=1.

        residual = self.shortcut(x)

        out = self.conv1(x)
        out = self.n1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.n2(out)

        out += residual
        out = self.act(out)
        return out


# ==========================================
# 2. SOFT ARGMAX
# ==========================================

class SpatialSoftArgmax3d(nn.Module):
    """
    Calculates the expected coordinate (Center of Mass) of the feature map.
    Preserves spatial precision better than Average Pooling.
    """

    def __init__(self, normalized_coordinates=True):
        super().__init__()
        self.normalized_coordinates = normalized_coordinates

    def forward(self, x):
        # x: (B, C, T, H, W)
        B, C, T, H, W = x.shape

        # Softmax to turn features into a probability distribution (heat map)
        # Flatten spatial dims: (B, C, T*H*W)
        flat_x = x.view(B, C, -1)
        softmax_attention = F.softmax(flat_x, dim=2)

        # Generate coordinate grids
        # We want to find the "center" in (t, y, x) space
        device = x.device

        # Create coordinates normalised to [-1, 1]
        rng_t = torch.linspace(-1, 1, T, device=device)
        rng_h = torch.linspace(-1, 1, H, device=device)
        rng_w = torch.linspace(-1, 1, W, device=device)

        grid_t, grid_h, grid_w = torch.meshgrid(rng_t, rng_h, rng_w, indexing='ij')

        # Stack and flatten: (3, T*H*W)
        coords = torch.stack([grid_t, grid_h, grid_w], dim=0).view(3, -1)

        # Calculate Expectation: Sum(Prob * Coord)
        # (B, C, N) x (3, N)^T -> (B, C, 3)
        expected_coords = torch.bmm(
            softmax_attention,
            coords.transpose(0, 1).unsqueeze(0).expand(B, -1, -1)
        )

        # Output: (B, C*3) - The (t,y,x) center of mass for each channel
        return expected_coords.view(B, -1)


# ==========================================
# 3. MAIN NETWORK
# ==========================================

class PetNetImproved3D(nn.Module):
    def __init__(self, in_channels=2, base_channels=16, norm="batch", **kwargs):
        super().__init__()

        # 1. Positional Encodings add 3 channels (sin, cos, z)
        # Input channels become in_channels + 3
        self.conv_in = nn.Conv3d(
            in_channels + 3,
            base_channels,
            kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 0), bias=False
        )
        self.n_in = make_norm(norm, base_channels)
        self.act = nn.ReLU(inplace=True)

        # 2. ResNet Backbone (Layers 1-7)
        # We progressively double channels: 16 -> 32 -> 64 -> 128 -> 256 -> 512 -> 1024 -> 2048
        # We add striding to reduce memory usage and increase receptive field

        # Layer 1: 16 -> 32 channels (Downsample)
        self.layer1 = ResidualBlock3D(base_channels, base_channels * 2, stride=2, norm=norm)

        # Layer 2: 32 -> 64 channels (Downsample)
        self.layer2 = ResidualBlock3D(base_channels * 2, base_channels * 4, stride=2, norm=norm)

        # Layer 3: 64 -> 128 channels (Downsample)
        self.layer3 = ResidualBlock3D(base_channels * 4, base_channels * 8, stride=2, norm=norm)

        # Layer 4: 128 -> 256 channels (Keep spatial)
        self.layer4 = ResidualBlock3D(base_channels * 8, base_channels * 16, stride=1, norm=norm)

        # Layer 5: 256 -> 512 channels (Keep spatial)
        self.layer5 = ResidualBlock3D(base_channels * 16, base_channels * 32, stride=1, norm=norm)

        # Layer 6: 512 -> 1024 channels (Keep spatial)
        self.layer6 = ResidualBlock3D(base_channels * 32, base_channels * 64, stride=1, norm=norm)

        # Layer 7: 1024 -> 2048 channels (Keep spatial)
        # Note: 128 * base_channels (16) = 2048 channels
        self.layer7 = ResidualBlock3D(base_channels * 64, base_channels * 128, stride=1, norm=norm)

        # 3. New Readout Head (Soft Argmax)
        # We project the deep features (2048 channels) down to 1 channel (a single heat map)
        self.final_conv = nn.Conv3d(base_channels * 128, 1, kernel_size=1)
        self.soft_argmax = SpatialSoftArgmax3d()

        # 4. Final Correction MLP
        # Takes the expected coordinate (3 values: t, y, x) and maps to (x1, y1, z1...)
        self.head = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 7)  # x1, y1, z1, x2, y2, z2, class
        )

    def _generate_positional_encodings(self, x):
        B, C, T, H, W = x.shape
        device = x.device

        # Theta (W)
        theta = torch.linspace(0, 2 * torch.pi, W, device=device)
        sin_t = torch.sin(theta).view(1, 1, 1, 1, W).expand(B, 1, T, H, W)
        cos_t = torch.cos(theta).view(1, 1, 1, 1, W).expand(B, 1, T, H, W)

        # Z (H)
        z = torch.linspace(-1, 1, H, device=device)
        z_map = z.view(1, 1, 1, H, 1).expand(B, 1, T, H, W)

        return torch.cat([sin_t, cos_t, z_map], dim=1)

    def forward(self, x):
        # x: (B, T, 2, H, W) based on your previous descriptions
        # Permute to (B, C, T, H, W) for 3D Conv
        x = x[:, :, :2, :, :]
        if x.shape[1] != 2 and x.shape[2] == 2:
            x = x.permute(0, 2, 1, 3, 4).contiguous()

            # Inject GPS (Positional Encodings)
        pos = self._generate_positional_encodings(x)
        x = torch.cat([x, pos], dim=1)  # (B, 5, T, H, W)

        # Stem
        # Apply circular padding to W dimension before convolution
        x = circular_pad_w(x, pad=1)
        x = self.conv_in(x)
        x = self.n_in(x)
        x = self.act(x)

        # Body (ResBlocks)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)

        # Head
        # Create heatmap (B, 1, T', H', W')
        heatmap = self.final_conv(x)

        # Soft Argmax -> (B, 3) -> [t_center, h_center, w_center]
        coords = self.soft_argmax(heatmap)

        # MLP Correction to final targets
        out = self.head(coords)

        return out


# ==========================================
# 4. TEST BLOCK
# ==========================================
if __name__ == "__main__":
    # Test with random data to ensure shapes align
    # Batch=2, Time=10, Channels=2, Height=32, Width=32
    dummy_input = torch.randn(2, 10, 2, 32, 32)

    model = PetNetImproved3D(in_channels=2, base_channels=16)

    # Move to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
        dummy_input = dummy_input.cuda()

    output = model(dummy_input)
    print("Output shape:", output.shape)  # Should be (2, 7)
    print("Success! Model is fully working.")