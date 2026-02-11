import torch
import numpy as np
from src.data.dataset import PetBurstDataset
from src.data.utils import expand_train_input
from src.model.factory import create_model
from argparse import Namespace
import os

# --- SETUP ---
# Update these paths/params to match your setup
CHECKPOINT_PATH = os.getenv("MODEL_PATH")  # Update this
VAL_DIR = os.getenv("VAL")
MODEL_NAME = os.getenv("MODEL")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOG_NORMALISE = os.getenv("LOG_NORMALISE", "False") == "True"


def cartesian_to_cylindrical(x, y, z):
    r = torch.sqrt(x ** 2 + y ** 2)
    theta = torch.atan2(y, x)
    return r, theta, z


def main():
    # 1. Load Model
    args = Namespace(
        model=MODEL_NAME,
        model_path=CHECKPOINT_PATH,
        normalisation_strategy="batch",
    )

    model = create_model(MODEL_NAME, args).to(DEVICE)

    # Load Weights
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    # 2. Load Data (Singles Only for clarity)
    # We want to see how well it locates ONE point.
    ds = PetBurstDataset(expand_train_input(VAL_DIR), pattern="*_0.pt")

    errors_r = []
    errors_theta = []  # In mm (arc length)
    errors_z = []

    print("Running Inference...")
    with torch.no_grad():
        for i in range(min(len(ds), 500)):  # Test 500 samples
            sample = ds[i]
            burst = sample["burst"].unsqueeze(0).to(DEVICE)
            target = sample["target"].unsqueeze(0).to(DEVICE)  # [x1, y1, z1, ...]

            # Preprocessing (Match your training!)
            if LOG_NORMALISE:
                burst = torch.log1p(burst) / 10.0

            # Predict
            pred = model(burst)  # Normalised [0, 1] usually

            # Scale Prediction back to MM (Assuming you used scale_factor=300)
            pred_mm = pred * 300.0

            # Get P1 coordinates
            p1_pred = pred_mm[0, :3]
            p1_true = target[0, :3]

            # Convert to Cylindrical
            r_p, t_p, z_p = cartesian_to_cylindrical(*p1_pred)
            r_t, t_t, z_t = cartesian_to_cylindrical(*p1_true)

            # Calculate Errors
            diff_r = torch.abs(r_p - r_t).item()
            diff_z = torch.abs(z_p - z_t).item()

            # Arc Length Error for Theta: R * dTheta
            # Handle wrap-around (e.g. 359 vs 1 degree)
            diff_angle = torch.abs(t_p - t_t)
            diff_angle = torch.min(diff_angle, 2 * np.pi - diff_angle)
            diff_arc = (r_t * diff_angle).item()

            errors_r.append(diff_r)
            errors_theta.append(diff_arc)
            errors_z.append(diff_z)

    # 3. Results
    print("\n=== ERROR DECOMPOSITION ===")
    print(f"Radial Error (Depth):    {np.mean(errors_r):.2f} mm")
    print(f"Angular Error (XY Arc):  {np.mean(errors_theta):.2f} mm")
    print(f"Axial Error (Z):         {np.mean(errors_z):.2f} mm")

    total_euc = np.sqrt(np.mean(errors_r) ** 2 + np.mean(errors_theta) ** 2 + np.mean(errors_z) ** 2)
    print(f"Estimated Vector Error:  {total_euc:.2f} mm")


if __name__ == "__main__":
    main()
