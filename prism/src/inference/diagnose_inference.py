import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset


# IMPORT YOUR MODEL HERE
# from model import PetNetImproved3D_V2
# (Assuming the class we defined previously is in the same file or imported)

# ==========================================
# 1. METRIC CALCULATIONS
# ==========================================
def calculate_metrics(predictions, targets):
    """
    predictions: (N, 7) -> [x1, y1, z1, x2, y2, z2, class]
    targets: (N, 7)
    """
    # 1. Euclidean Distance for Interaction 1 (x1, y1, z1)
    # We assume the first 3 cols are coord 1
    diff = predictions[:, :3] - targets[:, :3]
    dist_sq = np.sum(diff ** 2, axis=1)
    euclidean_errors = np.sqrt(dist_sq)

    # 2. Bias (Mean Error per dimension)
    bias = np.mean(diff, axis=0)

    # 3. Dynamic Range Check (Crucial for 50mm error diagnosis)
    pred_min, pred_max = np.min(predictions[:, :3]), np.max(predictions[:, :3])
    target_min, target_max = np.min(targets[:, :3]), np.max(targets[:, :3])

    return {
        "mean_error_mm": np.mean(euclidean_errors),
        "median_error_mm": np.median(euclidean_errors),
        "std_error_mm": np.std(euclidean_errors),
        "max_error_mm": np.max(euclidean_errors),
        "bias_x": bias[0], "bias_y": bias[1], "bias_z": bias[2],
        "pred_range": (pred_min, pred_max),
        "target_range": (target_min, target_max),
        "raw_errors": euclidean_errors,
        "raw_diffs": diff,
        "raw_preds": predictions[:, :3],
        "raw_targets": targets[:, :3]
    }


# ==========================================
# 2. VISUALIZATION
# ==========================================
def plot_diagnostics(metrics, save_path="diagnosis.png"):
    preds = metrics['raw_preds']
    targets = metrics['raw_targets']
    errors = metrics['raw_errors']

    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"Diagnostic Report: Mean Error {metrics['mean_error_mm']:.2f}mm", fontsize=16)

    # Plot 1: Error Histogram
    sns.histplot(errors, bins=50, ax=axs[0, 0], color='red', kde=True)
    axs[0, 0].set_title("Distribution of Euclidean Errors")
    axs[0, 0].set_xlabel("Error (mm)")

    # Plot 2: Prediction vs Target (X-axis) - Linearity Check
    axs[0, 1].scatter(targets[:, 0], preds[:, 0], alpha=0.3, s=10)
    axs[0, 1].plot([targets[:, 0].min(), targets[:, 0].max()],
                   [targets[:, 0].min(), targets[:, 0].max()], 'r--')  # Perfect line
    axs[0, 1].set_title("Linearity Check (X-axis)")
    axs[0, 1].set_xlabel("True X")
    axs[0, 1].set_ylabel("Predicted X")

    # Plot 3: Prediction vs Target (Z-axis) - Depth Check
    axs[0, 2].scatter(targets[:, 2], preds[:, 2], alpha=0.3, s=10, color='green')
    axs[0, 2].plot([targets[:, 2].min(), targets[:, 2].max()],
                   [targets[:, 2].min(), targets[:, 2].max()], 'r--')
    axs[0, 2].set_title("Linearity Check (Z-axis)")
    axs[0, 2].set_xlabel("True Z")
    axs[0, 2].set_ylabel("Predicted Z")

    # Plot 4: Error Heatmap (X vs Z spatial distribution)
    # Where are the errors happening?
    sc = axs[1, 0].scatter(targets[:, 0], targets[:, 2], c=errors, cmap='hot', s=10)
    plt.colorbar(sc, ax=axs[1, 0], label='Error (mm)')
    axs[1, 0].set_title("Spatial Error Map (True X vs Z)")
    axs[1, 0].set_xlabel("True X")
    axs[1, 0].set_ylabel("True Z")

    # Plot 5: Prediction Distribution (Collapse Check)
    sns.kdeplot(preds[:, 0], ax=axs[1, 1], label='Pred X', fill=True, alpha=0.3)
    sns.kdeplot(targets[:, 0], ax=axs[1, 1], label='True X', fill=True, alpha=0.3)
    axs[1, 1].set_title("Distribution Overlap (Normalization Check)")
    axs[1, 1].legend()

    # Plot 6: Text Stats
    text_str = (
        f"Mean Err: {metrics['mean_error_mm']:.2f}\n"
        f"Median Err: {metrics['median_error_mm']:.2f}\n"
        f"Bias X: {metrics['bias_x']:.2f}\n"
        f"Bias Z: {metrics['bias_z']:.2f}\n\n"
        f"Pred Range: {metrics['pred_range']}\n"
        f"Target Range: {metrics['target_range']}"
    )
    axs[1, 2].text(0.1, 0.5, text_str, fontsize=12, family='monospace')
    axs[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Diagnostics saved to {save_path}")
    plt.show()


# ==========================================
# 3. INFERENCE LOOP
# ==========================================
def run_diagnosis(model, loader, device):
    model.eval()
    all_preds = []
    all_targets = []

    print("Running Inference...")
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)

            # Forward pass
            output = model(data)

            # Store results (Move to CPU numpy immediately to save GPU RAM)
            all_preds.append(output.cpu().numpy())
            all_targets.append(target.cpu().numpy())

            if batch_idx % 10 == 0:
                print(f"Processed batch {batch_idx}")

    # Concatenate all batches
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Analyze
    print("\nCalculating Metrics...")
    metrics = calculate_metrics(all_preds, all_targets)

    # Print Console Report
    print("-" * 30)
    print("DIAGNOSTIC REPORT")
    print("-" * 30)
    print(f"Mean Euclidean Error: {metrics['mean_error_mm']:.4f} mm")
    print(f"Median Euclidean Error: {metrics['median_error_mm']:.4f} mm")
    print(f"Prediction Range: {metrics['pred_range']}")
    print(f"Target Range:     {metrics['target_range']}")

    if abs(metrics['pred_range'][1]) < 2.0 and abs(metrics['target_range'][1]) > 50.0:
        print("\n[!!!] CRITICAL WARNING: SCALING ISSUE DETECTED")
        print("Model is predicting normalized values (approx -1 to 1) but targets are in mm.")
        print("You must un-normalize predictions before calculating error.")

    return metrics


# ==========================================
# 4. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # 1. SETUP DUMMY DATA (Replace this with your Real Loader)
    # B=16, T=10, C=2, H=32, W=32
    # Targets: 7 values (x1,y1,z1, x2,y2,z2, class)
    # We simulate real world coordinates (-100mm to 100mm)
    dummy_x = torch.randn(100, 10, 2, 32, 32)
    dummy_y = torch.rand(100, 7) * 200 - 100

    dataset = TensorDataset(dummy_x, dummy_y)
    test_loader = DataLoader(dataset, batch_size=16)

    # 2. LOAD MODEL
    # Ensure you define/import PetNetPhysics3D here
    try:
        from src.model.pet_net_physics_3d import PetNetPhysics3D

        model = PetNetPhysics3D()

        # Load weights if you have them
        model.load_state_dict(torch.load("path/to/weights.pt"))

    except ImportError:
        # Fallback if the model class isn't in a separate file yet
        print("Model class not imported, ensure PetNetPhysics3D is defined.")
        exit()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 3. RUN DIAGNOSIS
    metrics = run_diagnosis(model, test_loader, device)

    # 4. PLOT
    plot_diagnostics(metrics)