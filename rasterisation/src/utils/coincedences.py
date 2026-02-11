import pandas as pd
import numpy as np

def isolate_coincidence_events(input_csv: str, output_csv: str):
    # Read the groundtruth CSV file
    df = pd.read_csv(input_csv, header=None)

    # Assume the last 6 columns correspond to x1, y1, z1, x2, y2, z2
    # Check if all 6 are non-zero
    coincidence_mask = (df.iloc[:, -6:] != 0).all(axis=1)

    # Filter the dataframe and keep only the last 6 columns
    coincidence_df = df[coincidence_mask].iloc[:, -6:]
    print(f"Final dataset has shape: {coincidence_df.shape}")

    # Save to new NumPy binary file
    np.save(output_csv, coincidence_df.to_numpy())


if __name__ == "__main__":
    base_path = "/Users/beast2/Documents/PhD/Github.nosync/matlab/src/output"
    # Example usage
    isolate_coincidence_events(
        input_csv=f"{base_path}/d5c2a047dc3748cfb0e2ae584223d35c/groundtruth.csv",
        output_csv=f"{base_path}/d5c2a047dc3748cfb0e2ae584223d35c/groundtruth_coincidence_only.npy"
    )