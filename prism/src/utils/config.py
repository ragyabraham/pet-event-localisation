# petnet/config.py
import os
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Geometry:
    R_INNER: float = 235.422  # mm
    R_OUTER: float = 278.296  # mm
    Z_HALF: float = 148.0  # mm (z in [-Z_HALF, +Z_HALF])
    CIRCUMFERENTIAL: float = 415.0
    AXIAL: float = 83.0
    R_THICK: float = R_OUTER - R_INNER
    R_MID: float = (R_OUTER + R_INNER) / 2


@dataclass
class TrainCfg:
    batch_size: int = 8
    num_workers: int = os.cpu_count()
    lr: float = 3e-4
    weight_decay: float = 1e-2
    max_epochs: int = 50
    clip_grad_norm: float = 0.0
    onecycle_max_lr: float = 2e-3
    amp: bool = True
    log_every: int = 50
    project_to_shell: bool = True  # project (x,y) to known radii
    lambda_value: float = 0.05
    normalisation_strategy: str = 'batch'
    model: str = "PetNetImproved3D".lower()
    seed: int = torch.seed()
    training_metrics = [
        "total_loss",
        "regression_loss",
        "classification_loss",
        "regression_loss_trues",
        "regression_loss_singles",
        "mean_error_x_mm",
        "mean_error_y_mm",
        "mean_error_z_mm",
        "mean_error_r_mm",
        "mean_error_arc_mm",
        "mean_euclidean_error_mm",
        "classification_loss_true",
        "classification_loss_singles",
        "classification_accuracy",
        "classification_accuracy_true",
        "classification_accuracy_singles",
        "regression_p1_singles",
        "regression_p2_singles",
        "regression_p1_trues",
        "regression_p2_trues"
    ]
