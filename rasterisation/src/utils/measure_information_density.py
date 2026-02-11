import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import random
# import seaborn as sns

def compute_entropy(frame: np.ndarray) -> float:
    total = np.sum(frame)
    if total == 0:
        return 0.0
    probs = frame / total
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))

def compute_info_density(entropy: float, num_pixels: int) -> float:
    """Compute information density in bits per pixel."""
    return entropy / num_pixels if num_pixels > 0 else 0.0

def analyze_multiple_resolutions(resolution_config: dict, sample_size: int = 500, output_csv: str = None):
    """
    Compute entropy and info density for random samples across multiple resolutions.

    Parameters:
        resolution_config (dict): {
            "12x12": {"path": "res_12x12.csv", "pixel_size": (12, 12)},
            ...
        }
        sample_size (int): Number of frames to randomly sample per resolution
        output_csv (str): Optional path to save combined results
    """
    all_results = []

    for label, config in resolution_config.items():
        csv_path = config["path"]
        pixel_w, pixel_h = config["pixel_size"]

        if not os.path.exists(csv_path):
            print(f"Warning: file {csv_path} not found. Skipping {label}.")
            continue

        df = pd.read_csv(csv_path, header=None)

        if df.shape[0] < sample_size:
            print(f"Warning: Only {df.shape[0]} frames found for {label}. Using all of them.")
            sampled_df = df
        else:
            sampled_df = df.sample(n=sample_size, random_state=42)

        for idx, row in sampled_df.iterrows():
            frame = row.iloc[1:].values.astype(float)  # skip index
            entropy = compute_entropy(frame)
            density = compute_info_density(entropy, len(frame))
            all_results.append({
                "Frame": int(row.iloc[0]),
                "Entropy (bits)": entropy,
                "Info Density (bits/pixel)": density,
                "Resolution": label
            })

    results_df = pd.DataFrame(all_results)

    if output_csv:
        results_df.to_csv(output_csv, index=False)
        print(f"Saved combined results to: {output_csv}")

    # Aggregate mean and standard deviation
    summary_df = results_df.groupby("Resolution")["Info Density (bits/pixel)"].agg(["mean", "std"]).reset_index()

    # Plot line graph with error bars (styled)
    plt.figure(figsize=(10, 6))
    plt.style.use("ggplot")

    # Sort summary_df for consistent ordering
    summary_df_sorted = summary_df.sort_values(by="Resolution")
    x_labels = summary_df_sorted["Resolution"]
    x_ticks = np.arange(len(x_labels))
    y_means = summary_df_sorted["mean"].values
    y_errors = summary_df_sorted["std"].values

    # Plot connected line with error bars
    plt.errorbar(
        x_ticks,
        y_means,
        yerr=y_errors,
        fmt='-o',
        capsize=6,
        linewidth=2,
        markersize=8,
        color='tab:blue',
        label='Mean ± SD'
    )


    # Replace xtick labels with resolution strings
    plt.xticks(x_ticks, x_labels)

    plt.title("Mean Information Density per Pixel across Resolutions", fontsize=14, weight='bold')
    plt.ylabel("Mean Information Density (bits/pixel)", fontsize=12)
    plt.xlabel("Frame Resolution", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig("information_density.svg")
    plt.show()

    return results_df


# ================== USAGE ==================

if __name__ == "__main__":
    # Define your resolution config
    base_path = "/Users/beast2/Documents/PhD/Github.nosync/matlab/src/output"
    resolution_config = {
        "207x41": {"path": f"{base_path}/8c0b6ae659b843fbb836efe31bbb2686/inner_random_sample.csv", "pixel_size": (12, 12)},
        "311x62": {"path": f"{base_path}/3ebc89cee3a5424a930a532e76689803/inner_random_sample.csv", "pixel_size": (6, 6)},
        "415x83": {"path": f"{base_path}/d5c2a047dc3748cfb0e2ae584223d35c/inner_random_sample.csv", "pixel_size": (3, 3)},
    }

    # Optional output CSV
    output_csv = "info_density_all_resolutions.csv"

    # Run analysis
    analyze_multiple_resolutions(resolution_config, sample_size=200, output_csv=output_csv)