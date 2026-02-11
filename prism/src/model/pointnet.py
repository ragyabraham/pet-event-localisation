import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.config import Geometry


class PointNetPET(nn.Module):
    """
    PointNet-based regressor for PET Event Localisation.

    Why this works:
    - Instead of processing empty voxels, we convert the burst to a Point Cloud.
    - Each 'Point' is a fired SiPM pixel with features: [x, y, z, time, intensity].
    - We map Inner/Outer sensors to their physical 3D locations (R_in, R_out).
    - PointNet is naturally permutation invariant and handles sparsity perfectly.
    """

    def __init__(self, num_points=512, in_channels=5):
        super().__init__()
        self.num_points = num_points
        self.debug = False

        # --- GEOMETRY CONSTANTS (Cached) ---
        self.R_INNER = Geometry.R_INNER  # e.g. 235.422
        self.R_OUTER = Geometry.R_OUTER  # e.g. 278.296
        self.Z_HALF = Geometry.Z_HALF  # e.g. 148.0
        self.CIRCUMFERENCE = Geometry.CIRCUMFERENTIAL

        # --- POINTNET ENCODER ---
        # Input: [Batch, 5, Num_Points] -> Features: [x, y, z, t, amp]
        self.conv1 = nn.Conv1d(in_channels, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        # --- REGRESSION HEAD ---
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)

        # Output: 7 values [x1, y1, z1, x2, y2, z2, class_logit]
        self.fc3 = nn.Linear(256, 7)

        self.bn_fc1 = nn.BatchNorm1d(512)
        self.bn_fc2 = nn.BatchNorm1d(256)

        self.dropout = nn.Dropout(p=0.3)

    def _grid_to_points(self, burst):
        """
        Converts dense (B, T, C, H, W) -> Point Cloud (B, N, 5)
        Features: [x, y, z, normalized_time, intensity]
        """
        B, T, C, H, W = burst.shape
        device = burst.device

        # 1. Threshold to find active pixels (ignore noise)
        # Dynamic threshold: max(0.1, 10% of max signal) to be robust
        # For speed, we just use a hard clamp or simple > 0 check if data is sparse
        mask = burst > 1e-3

        # This part is tricky to vectorise perfectly across batches with variable point counts.
        # Strategy: We process batch-wise but pad/sample to fixed size N.

        # Pre-compute coordinate grids (cached if possible, but cheap to make)
        # Theta: 0 to 2pi
        theta_lin = torch.linspace(0, 2 * torch.pi, W, device=device)
        # Z: -148 to +148
        z_lin = torch.linspace(-self.Z_HALF, self.Z_HALF, H, device=device)

        # Time: 0 to T (Normalised)
        t_lin = torch.linspace(0, 1, T, device=device)

        # Mesh grids
        # We need indices to look up physical coordinates
        # nonzero() returns indices: [b, t, c, h, w]
        indices = torch.nonzero(mask, as_tuple=False)

        # If no points found (empty batch), return the dummy
        if indices.shape[0] == 0:
            return torch.zeros(B, 5, self.num_points, device=device)

        # Separate by batch index to handle variable N
        batch_point_list = []

        for b in range(B):
            # Get indices for this batch item
            idx_b = indices[indices[:, 0] == b]

            if idx_b.shape[0] == 0:
                # Pad with zeros if empty
                batch_point_list.append(torch.zeros(self.num_points, 5, device=device))
                continue

            # Extract attributes
            t_idx = idx_b[:, 1]
            c_idx = idx_b[:, 2]  # 0=Inner, 1=Outer, 2=DOI (ignore a DOI channel for loc, use intensity)
            h_idx = idx_b[:, 3]
            w_idx = idx_b[:, 4]

            # --- MAP TO PHYSICS ---
            # 1. Radius (Dependent on Channel)
            # Channel 0 = Inner, Channel 1 = Outer
            # We treat C=2 (DOI) as intensity info, not geometric location, or ignore.
            # Let's mask only C=0 and C=1 for geometry.
            valid_c = c_idx < 2
            if not valid_c.any():
                batch_point_list.append(torch.zeros(self.num_points, 5, device=device))
                continue

            t_idx, c_idx, h_idx, w_idx = t_idx[valid_c], c_idx[valid_c], h_idx[valid_c], w_idx[valid_c]

            r_val = torch.where(c_idx == 0,
                                torch.tensor(self.R_INNER, device=device),
                                torch.tensor(self.R_OUTER, device=device))

            # 2. Theta & Z
            theta_val = theta_lin[w_idx]
            z_val = z_lin[h_idx]

            # 3. Cartesian Conversion
            x_val = r_val * torch.cos(theta_val)
            y_val = r_val * torch.sin(theta_val)

            # 4. Time & Intensity
            time_val = t_lin[t_idx]
            # Gather intensity values from original tensor
            intensities = burst[b, t_idx, c_idx, h_idx, w_idx]

            # 5. Stack Features: [x, y, z, t, amp]
            points = torch.stack([x_val, y_val, z_val, time_val, intensities], dim=1)

            # --- SAMPLING (Critical) ---
            # We need exactly num_points.
            N_actual = points.shape[0]
            if N_actual >= self.num_points:
                # Random sampling (faster than FPS and sufficient for this)
                choice = torch.randperm(N_actual, device=device)[:self.num_points]
                points = points[choice]
            else:
                # Padding (repeat points)
                gap = self.num_points - N_actual
                choice = torch.randint(0, N_actual, (gap,), device=device)
                padding = points[choice]
                points = torch.cat([points, padding], dim=0)

            batch_point_list.append(points)

        return torch.stack(batch_point_list)  # (B, N, 5)

    def forward(self, x):
        # 1. Convert Voxel Grid to Point Cloud
        # Input: (B, T, 3, H, W) -> Output: (B, N, 5)
        # Note: We assume 'x' is log-normalized counts for intensity to be meaningful
        x = self._grid_to_points(x)

        # PointNet expects (B, Channels, N)
        x = x.transpose(2, 1)  # (B, 5, N)

        # 2. Feature Extraction (Shared MLP)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))  # (B, 1024, N)

        # 3. Global Max Pooling (Symmetric Function)
        # This grabs the most critical features regardless of point order
        x = torch.max(x, 2, keepdim=False)[0]  # (B, 1024)

        # 4. Regression Head
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)  # (B, 7)

        return x