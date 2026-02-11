import os
import torch
import torch.nn as nn
from src.utils.config import Geometry
from src.utils.utils import cartesian_to_polar, compute_angle_difference


class VirtualCylinderAwareLoss(nn.Module):
    def __init__(self, lambda_cls=0.01, w_polar=0.5, w_bound=0.2, hard_mining_frac=0.2):
        super().__init__()
        self.lambda_cls = lambda_cls
        self.w_polar = w_polar
        self.w_bound = w_bound
        self.hard_frac = hard_mining_frac

        self.log_normalise = os.getenv("LOG_NORMALISE", "False") == "True"
        self.scale_factor = float(os.getenv("LOG_SCALE_FACTOR", "1.0")) if self.log_normalise else 1.0

        self.bce = nn.BCEWithLogitsLoss()
        self.smooth_l1 = nn.SmoothL1Loss(reduction='none', beta=1.0)

        # Geometry constants (in mm)
        self.r_inner = Geometry.R_INNER
        self.r_outer = Geometry.R_OUTER
        self.z_half = Geometry.Z_HALF

    def _boundary_loss(self, p):
        """Trick A: Penalise points outside [R_in, R_out] and [-Z, +Z]."""
        # p is (N, 3) in scaled space -> convert to mm approx for check
        p_mm = p * self.scale_factor
        r = torch.sqrt(p_mm[:, 0] ** 2 + p_mm[:, 1] ** 2 + 1e-8)
        z = p_mm[:, 2].abs()

        # ReLU creates a gradient only when constraints are violated
        loss_r = torch.relu(self.r_inner - r ) + torch.relu(r - self.r_outer)
        loss_z = torch.relu(z - self.z_half)
        return (loss_r + loss_z).mean()

    @staticmethod
    def _polar_loss(p, t):
        # We calculate this primarily for the gradient signal
        r_p, th_p, z_p = cartesian_to_polar(p[:, 0], p[:, 1], p[:, 2])
        r_t, th_t, z_t = cartesian_to_polar(t[:, 0], t[:, 1], t[:, 2])

        # Arc error = Radius * Angle_Diff
        arc_err = r_t * torch.abs(compute_angle_difference(th_p, th_t))
        r_err = torch.abs(r_p - r_t)
        z_err = torch.abs(z_p - z_t)

        # Weighted combination (Emphasise Arc/Theta as it's often the hardest)
        return (1.5 * arc_err + 1.0 * r_err + 1.0 * z_err).mean()

    @staticmethod
    def _get_polar_diffs(p, t):
        """Helper to get absolute differences in (r, theta_arc, z) for metrics."""
        r_p, th_p, z_p = cartesian_to_polar(p[:, 0], p[:, 1], p[:, 2])
        r_t, th_t, z_t = cartesian_to_polar(t[:, 0], t[:, 1], t[:, 2])

        # Calculate Arc Length error (mm) instead of raw radians
        diff_arc = r_t * torch.abs(compute_angle_difference(th_p, th_t))
        diff_r = torch.abs(r_p - r_t)
        diff_z = torch.abs(z_p - z_t)

        return diff_r, diff_arc, diff_z

    def _weighted_trues_loss(self, loss_per_sample):
        """Trick F: Hard Example Mining - Upweight worst k% losses."""
        if loss_per_sample.numel() == 0:
            return loss_per_sample.sum()

        k = int(loss_per_sample.shape[0] * self.hard_frac)
        if k > 0:
            # Sort descending
            sorted_loss, _ = torch.sort(loss_per_sample, descending=True)
            # Upweight the top k worst examples by 2.0x
            hard_loss = sorted_loss[:k].mean() * 2.0
            easy_loss = sorted_loss[k:].mean() if k < loss_per_sample.shape[0] else 0.0
            return hard_loss + easy_loss
        return loss_per_sample.mean()

    def forward(self, pred, target):
        # 1. Unpack & Normalise Targets
        target_norm = target.clone()
        target_norm[:, :6] = target[:, :6] / self.scale_factor

        pred_p1 = pred[:, 0:3]
        pred_p2 = pred[:, 3:6]
        pred_logit = pred[:, 6]

        target_p1 = target_norm[:, 0:3]
        target_p2 = target_norm[:, 3:6]
        target_event_type = target[:, 6]

        # 2. Masks
        target_classification = torch.clamp(target_event_type, 0.0, 1.0)
        true_mask = target_classification == 1.0
        singles_mask = target_classification == 0.0

        # --- NEW: Calculate Radial Weights for Ground Truth ---
        # We assume the bias comes from the difficulty of the TARGET location

        # Loss Accumulators
        regression_loss_trues = torch.tensor(0.0, device=pred.device)
        regression_loss_singles = torch.tensor(0.0, device=pred.device)

        # Initialise metric placeholders
        loss_singles_p1 = torch.tensor(0.0, device=pred.device)
        loss_singles_p2 = torch.tensor(0.0, device=pred.device)
        log_p1_trues = torch.tensor(0.0, device=pred.device)
        log_p2_trues = torch.tensor(0.0, device=pred.device)

        # Metric Accumulators
        all_abs_errors = []  # For x, y, z
        all_euc_errors = []  # For Euclidean Distance
        all_polar_errors = []  # For r, theta, z

        # --- 3. TRUES Logic ---
        if true_mask.any():
            p1_t = pred_p1[true_mask]
            p2_t = pred_p2[true_mask]
            t1_t = target_p1[true_mask]
            t2_t = target_p2[true_mask]

            # Get weights for this subset

            # Permutation Loss with Radial Weighting applied to the raw L1 sums
            # Option A: P1->T1, P2->T2
            loss_a = (self.smooth_l1(p1_t, t1_t).sum(dim=1)+
                      self.smooth_l1(p2_t, t2_t).sum(dim=1))

            # Option B: P1->T2, P2->T1 (Swapped)
            loss_b = (self.smooth_l1(p1_t, t2_t).sum(dim=1)+
                      self.smooth_l1(p2_t, t1_t).sum(dim=1))

            min_loss_per_sample, indices = torch.min(torch.stack([loss_a, loss_b], dim=1), dim=1)

            # Apply Trick F (Hard Example Mining) ON TOP of Radial Weighting
            regression_loss_trues = self._weighted_trues_loss(min_loss_per_sample / 2.0)

            # Reconstruct Targets
            idx_expanded = indices.view(-1, 1).expand(-1, 3)

            # Gather correct T1 match
            t_stack_1 = torch.stack([t1_t, t2_t], dim=1)
            t1_matched = torch.gather(t_stack_1, 1, idx_expanded.unsqueeze(1)).squeeze(1)

            # Gather correct T2 match
            t_stack_2 = torch.stack([t2_t, t1_t], dim=1)
            t2_matched = torch.gather(t_stack_2, 1, idx_expanded.unsqueeze(1)).squeeze(1)

            # Auxiliary Losses
            aux_polar = (self._polar_loss(p1_t, t1_matched) + self._polar_loss(p2_t, t2_matched)) * 0.5
            aux_bound = (self._boundary_loss(p1_t) + self._boundary_loss(p2_t))
            regression_loss_trues += (self.w_polar * aux_polar + self.w_bound * aux_bound)

            # Metrics
            log_p1_trues = (min_loss_per_sample / 2.0).mean().detach()
            log_p2_trues = log_p1_trues

            # Collect Metrics (XYZ & Polar)
            with torch.no_grad():
                # XYZ
                all_abs_errors.append((p1_t - t1_matched).abs())
                all_abs_errors.append((p2_t - t2_matched).abs())

                # Euclidean
                all_euc_errors.append(torch.linalg.norm(p1_t - t1_matched, dim=1))
                all_euc_errors.append(torch.linalg.norm(p2_t - t2_matched, dim=1))

                # Polar (r, arc, z)
                dr1, darc1, dz1 = self._get_polar_diffs(p1_t, t1_matched)
                dr2, darc2, dz2 = self._get_polar_diffs(p2_t, t2_matched)

                # Stack as [N, 3] -> (r, arc, z)
                all_polar_errors.append(torch.stack([dr1, darc1, dz1], dim=1))
                all_polar_errors.append(torch.stack([dr2, darc2, dz2], dim=1))

        # --- 4. SINGLES Logic ---
        if singles_mask.any():
            p1_s = pred_p1[singles_mask]
            t1_s = target_p1[singles_mask]

            # Weighted Regression
            raw_loss_p1 = self.smooth_l1(p1_s, t1_s).sum(dim=1)
            loss_singles_p1 = (raw_loss_p1).mean()  # Apply weight

            loss_singles_p2 = torch.nn.functional.mse_loss(pred_p2[singles_mask], target_p2[singles_mask])

            aux_bound_s = self._boundary_loss(p1_s)
            aux_polar_s = self._polar_loss(p1_s, t1_s)

            regression_loss_singles = (loss_singles_p1 + loss_singles_p2 +
                                       self.w_bound * aux_bound_s +
                                       self.w_polar * aux_polar_s)

            # Collect Errors for Metrics (Only P1 is valid for singles)
            with torch.no_grad():
                # XYZ
                all_abs_errors.append((p1_s - t1_s).abs())
                # Euclidean
                all_euc_errors.append(torch.linalg.norm(p1_s - t1_s, dim=1))
                # Polar
                dr_s, darc_s, dz_s = self._get_polar_diffs(p1_s, t1_s)
                all_polar_errors.append(torch.stack([dr_s, darc_s, dz_s], dim=1))

        # --- 5. Total Loss ---
        loss_cls = self.bce(pred_logit, target_event_type.float())

        # 6. Total Loss
        total_loss = regression_loss_trues + regression_loss_singles + (self.lambda_cls * loss_cls)

        # --- 6. Aggregation ---
        sf = self.scale_factor

        # Defaults
        log_dx = log_dy = log_dz = torch.tensor(0.0, device=pred.device)
        log_dr = log_darc = torch.tensor(0.0, device=pred.device)
        log_euclidean = torch.tensor(0.0, device=pred.device)

        if len(all_abs_errors) > 0:
            # XYZ Mean
            all_xyz = torch.cat(all_abs_errors, dim=0).mean(dim=0)
            log_dx, log_dy, log_dz = all_xyz[0], all_xyz[1], all_xyz[2]

            # Polar Mean
            all_polar = torch.cat(all_polar_errors, dim=0).mean(dim=0)
            log_dr, log_darc = all_polar[0], all_polar[1]

            # Euclidean Mean
            log_euclidean = torch.cat(all_euc_errors, dim=0).mean()

        with torch.no_grad():
            pred_probs = torch.sigmoid(pred_logit)
            predicted_class = (pred_probs >= 0.5).float()
            accuracy = (predicted_class == target_event_type).float().mean()

        return {
            "total_loss": total_loss,
            "regression_loss": (regression_loss_trues + regression_loss_singles).detach() * sf,

            # Singles Metrics
            "regression_p1_singles": loss_singles_p1.detach() * sf,
            "regression_p2_singles": loss_singles_p2.detach() * sf,

            # Trues Metrics
            "regression_p1_trues": log_p1_trues * sf,
            "regression_p2_trues": log_p2_trues * sf,

            # Coordinate Decomposition (XYZ)
            "mean_error_x_mm": log_dx * sf,
            "mean_error_y_mm": log_dy * sf,
            "mean_error_z_mm": log_dz * sf,

            # Coordinate Decomposition (Polar) - NEW
            "mean_error_r_mm": log_dr * sf,
            "mean_error_arc_mm": log_darc * sf,  # Theta error in mm

            # Euclidean
            "mean_euclidean_error_mm": log_euclidean * sf,

            # Split Losses
            "regression_loss_trues": regression_loss_trues.detach() * sf,
            "regression_loss_singles": regression_loss_singles.detach() * sf,

            # Classification
            "classification_loss": loss_cls.detach(),
            "classification_accuracy": accuracy.detach()
        }