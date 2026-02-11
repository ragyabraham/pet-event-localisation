import logging
from argparse import Namespace

import torch.nn
import torch.nn as nn

from src.model.swin_unet_regressor import SwinUNetV2Regressor
from src.model.pointnet import PointNetPET
from src.model.pet_net_improved_3d import PetNetImproved3D
from src.model.pet_net_physics_3d import PetNetPhysics3D
from src.model.pet_net_physics_3d_polar import PetNetPhysics3DPolar
from src.training.losses import VirtualCylinderAwareLoss


def load_model(model_name: str, args: Namespace) -> nn.Module:
    model_name = model_name.lower()

    # 1. Instantiate the Architecture
    if model_name == "petnetimproved3d":
        return PetNetImproved3D(norm=args.normalisation_strategy)

    elif model_name == "petnetphysics3d":
        return PetNetPhysics3D()
    elif model_name == "petnetphysics3dpolar":
        return PetNetPhysics3DPolar()

    elif model_name == "pointnet":
        # num_points=1024 is a good balance for PET (enough for scatter tail)
        return PointNetPET(num_points=1024)
    elif model_name == "swin_unet":
        return SwinUNetV2Regressor()
    else:
        raise ValueError(f"Unknown model '{model_name}'.")


def create_model(model_name: str, args: Namespace) -> nn.Module:
    # 1. Build the empty shell (Random Weights)
    model = load_model(model_name, args)

    # 2. Load Weights (If a path is provided)
    if args.model_path:
        print(f"Loading checkpoint from: {args.model_path}")

        # Load to CPU first to avoid GPU OOM or device mismatch
        checkpoint = torch.load(args.model_path, map_location="cpu", weights_only=True)

        # Extract state dict
        if "model_state" in checkpoint:
            state_dict = checkpoint["model_state"]
        else:
            state_dict = checkpoint  # Handle cases where only state_dict was saved

        # Load into a model
        # strict=True (default) ensures we crash if keys/shapes don't match exactly.
        # This prevents the "Silent Failure".
        model.load_state_dict(state_dict, strict=True)

        print(f"✅ Successfully loaded model state for {model_name}")

    return model


def create_loss(args: Namespace):
    loss_name = args.loss.lower()
    logging.info(f"Loading '{loss_name}' loss function")
    match loss_name:
        case "smooth":
            return torch.nn.SmoothL1Loss()
        case "mse":
            return torch.nn.MSELoss()
        case "virtualcylinderaware":
            return VirtualCylinderAwareLoss()
        case _:
            raise ValueError(f"Invalid loss type provided: '{loss_name}'")
