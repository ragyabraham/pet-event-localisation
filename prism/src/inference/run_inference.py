import logging

import torch
from torch.utils.data import DataLoader
from torch import nn

from src.model.factory import create_model
from src.data.dataset import PetBurstDataset
from src.data.utils import collate_with_radius_measure
from src.inference.visualisations import visualise_predictions, plot_error_distribution
from src.inference.data_exploration import check_data_scale


def load_model(
        checkpoint_path: str,
        device: torch.device,
        model_name: str = "petnetphysics3d",
        normalisation_strategy: str = "batch"
) -> nn.Module:
    """
    Load a trained model from a checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to load model on
        model_name: Name of the model architecture
        normalisation_strategy: Normalization strategy used during training

    Returns:
        Loaded model in eval mode
    """
    logging.info(f"Loading checkpoint from {checkpoint_path}")

    # Create dummy args namespace for model creation
    from argparse import Namespace
    args = Namespace(
        normalisation_strategy=normalisation_strategy,
        freeze_backbone_params=0,
        model_path=None
    )

    # Create model architecture
    model = create_model(model_name, args)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    # Extract model state dict (handle DDP wrapped models)
    if "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
    else:
        state_dict = checkpoint

    # Load state dict
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    logging.info(f"Model loaded successfully from epoch {checkpoint.get('epoch', 'unknown')}")
    if "val_loss" in checkpoint:
        logging.info(f"Checkpoint validation loss: {checkpoint['val_loss']:.4f}")

    return model


def remove_outliers_calculate_metrics(all_targets_p1, all_targets_p2, mean_error):
    # 1. Define Geometry Constants (Update if yours differ slightly)
    R_INNER = 235.422
    R_OUTER = 278.296
    R_MID = (R_INNER + R_OUTER) / 2.0

    # 2. Compute Radial Depth for Ground Truth (Physics Truth)
    # We filter based on where the event ACTUALLY happened (Targets), not where we guessed (Predictions)
    r_t1 = torch.sqrt(all_targets_p1[:, 0] ** 2 + all_targets_p1[:, 1] ** 2)
    r_t2 = torch.sqrt(all_targets_p2[:, 0] ** 2 + all_targets_p2[:, 1] ** 2)

    # --- DEFINE FILTERS HERE ---

    # A. Geometric Filter: "Inner Half Only"
    # Removes interactions that occurred deep in the scanner (Outer Surface)
    # We strictly require BOTH photons to be within the inner half to keep the event.
    mask_geometric = (r_t1 <= R_MID) & (r_t2 <= R_MID)

    # B. Sanity Filter: "Gross Outlier Removal"
    # Removes predictions that are physically impossible (e.g., error > 10cm)
    # This removes "failed" inferences that skew the mean but don't represent typical performance.
    mask_sanity = mean_error < 100.0

    # C. Statistical Filter (Optional): "99th Percentile"
    # Automatically chops off the worst 1% of errors
    threshold_99 = torch.quantile(mean_error, 0.99)
    mask_statistical = mean_error < threshold_99

    # 3. Combine Filters (Toggle using logic operators & etc.)
    # Current Goal: Test median error of ONLY the inner half, removing gross failures.
    final_mask = mask_geometric & mask_sanity

    # 4. Apply Mask
    cleaned_errors = mean_error[final_mask]

    # 5. Print Filtered Report
    print("\n" + "=" * 60)
    print(f"CLEANED / FILTERED METRICS")
    print(f"Kept {len(cleaned_errors)} of {len(mean_error)} events ({len(cleaned_errors) / len(mean_error):.1%})")
    print(f"Active Filters: R_true <= {R_MID:.1f}mm (Inner Half)")
    print("=" * 60)

    if len(cleaned_errors) > 0:
        print(f"Regression Errors (Filtered):")
        print(f"  Overall - Mean:   {cleaned_errors.mean():.2f} mm")
        print(f"  Overall - Median: {cleaned_errors.median():.2f} mm  <-- NEW RESULT")
        print(f"  Overall - Max:    {cleaned_errors.max():.2f} mm")

        # Show Delta (Improvement)
        delta_mean = mean_error.mean() - cleaned_errors.mean()
        delta_median = mean_error.median() - cleaned_errors.median()
        print(f"\nImpact of Filtering:")
        print(f"  Mean Improvement:   -{delta_mean:.2f} mm")
        print(f"  Median Improvement: -{delta_median:.2f} mm")
    else:
        print("  [WARNING] Filter was too strict! No events remained.")
    print("=" * 60)
    return cleaned_errors


def calculate_inference_metrics(
        all_predictions: torch.Tensor,
        all_targets: torch.Tensor,
        remove_outliers: bool = True
):
    cleaned_errors = None

    # Compute errors with permutation matching (same as training)
    all_predictions_p1 = all_predictions[:, :3]
    all_predictions_p2 = all_predictions[:, 3:6]
    all_targets_p1 = all_targets[:, :3]
    all_targets_p2 = all_targets[:, 3:6]

    # Permutation-aware error (same logic as loss function)
    error_a1 = torch.linalg.norm(all_predictions_p1 - all_targets_p1, dim=1)
    error_a2 = torch.linalg.norm(all_predictions_p2 - all_targets_p2, dim=1)
    error_b1 = torch.linalg.norm(all_predictions_p1 - all_targets_p2, dim=1)
    error_b2 = torch.linalg.norm(all_predictions_p2 - all_targets_p1, dim=1)

    # Take minimum permutation
    error_a = (error_a1 + error_a2) / 2
    error_b = (error_b1 + error_b2) / 2
    mean_error = torch.minimum(error_a, error_b)

    # We compute P90/P95 on the whole tensor 'mean_error', not the scalar median.
    p90 = torch.quantile(mean_error, 0.90)
    p95 = torch.quantile(mean_error, 0.95)

    print("\n" + "=" * 60)
    print("AGGREGATE METRICS")
    print("=" * 60)
    print(f"\nRegression Errors (3D distance in mm):")
    print(f"  Overall - Mean:   {mean_error.mean():.2f} ± {mean_error.std():.2f} mm")
    print(f"  Overall - Median: {mean_error.median():.2f} mm")
    print(f"  Overall - P90:    {p90:.2f} mm")
    print(f"  Overall - P95:    {p95:.2f} mm")
    print(f"  Overall - Min:    {mean_error.min():.2f} mm")
    print(f"  Overall - Max:    {mean_error.max():.2f} mm")
    print("=" * 60)

    if remove_outliers:
        cleaned_errors = remove_outliers_calculate_metrics(all_targets_p1, all_targets_p2, mean_error)

    return mean_error, cleaned_errors


def perform_inference(model, device: torch.device, dataloader: DataLoader):
    from tqdm import tqdm

    all_predictions = []
    all_targets = []

    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Running inference")):
            try:
                burst = batch["burst"].to(device)
                target = batch["target"]

                # Forward pass
                output = model(burst)
                pred_cart = output[:, :6]

                # Store results
                all_predictions.append(pred_cart.cpu())
                all_targets.append(target)

            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue
    all_predictions_tensor = torch.cat(all_predictions, dim=0)
    all_targets_tensor = torch.cat(all_targets, dim=0)
    return all_targets_tensor, all_predictions_tensor


def run_inference_job(
        data_directory: str,
        model_name: str,
        model_path: str,
        device: torch.device,
        output_directory: str = "./"
):
    model = load_model(
        checkpoint_path=model_path,
        device=device,
        model_name=model_name,
    )
    print("Model loaded successfully!")

    dataset = PetBurstDataset(
        data_directory,
        transform=None,
        regex_filter=r"\d+_1_[01].pt"
    )
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_with_radius_measure
    )

    all_targets, all_predictions = perform_inference(model, device, dataloader)

    if all_targets is not None and all_predictions is not None:
        if len(all_predictions) > 0 and len(all_targets) > 0:
            torch.save(all_predictions, f"{output_directory}/predictions.pt")
            torch.save(all_targets, f"{output_directory}/targets.pt")
            print(f"Dataset loaded: {len(dataset)} samples (geometry mask will be applied during iteration)")
            mean_error, cleaned_errors = calculate_inference_metrics(all_predictions, all_targets)
            logging.info("Error Distribution for raw errors")
            plot_error_distribution(mean_error)
            if cleaned_errors is not None:
                logging.info("Error Distribution for cleaned errors")
                plot_error_distribution(cleaned_errors)
            # Visualise the best error
            best_indices = torch.argsort(mean_error)[2:3]

            print("Visualizing samples with best errors:")
            for rank, idx in enumerate(best_indices, 1):
                print(f"\nRank {rank}: Sample {idx.item()} with error {mean_error[idx]:.2f} mm")

                # Get the sample
                sample = dataset[idx.item()]
                burst_single = sample["burst"].unsqueeze(0).to(device)  # Add batch dimension

                # Run inference
                with torch.no_grad():
                    output = model(burst_single)
                    pred_cart = output[:, :6]
                    class_prob = torch.sigmoid(output[:, 6])

                # Visualize
                visualise_predictions(
                    burst_single.cpu(),
                    pred_cart.cpu(),
                    sample["target"].unsqueeze(0),
                    sample["event_type"].unsqueeze(0),
                    class_prob.cpu(),
                    idx=0
                )
                # Added some data exploration metrics
                check_data_scale("", dataloader)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run inference on PET burst data")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model checkpoint file"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing the dataset"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="petnetphysics3d",
        help="Name of the model architecture (default: petnetphysics3d)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./",
        help="Directory to save output files (default: ./)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda", "mps"],
        help="Device to run inference on (default: auto-detect)"
    )

    args = parser.parse_args()

    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_inference_job(args.data_dir, args.model_name, args.model_path, device, args.output_dir)
