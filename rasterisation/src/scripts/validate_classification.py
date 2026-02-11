import argparse
import glob
import os
import torch
import sys
from collections import defaultdict
from tqdm import tqdm

# --- Geometry Constants (from your config) ---
R_INNER = 235.422
R_OUTER = 278.296
Z_HALF = 148.0
TOLERANCE = 1.0  # Strict tolerance for 0.0 checks


def is_point_valid(x, y, z):
    """
    Checks if a point is non-zero.
    (We assume missing points are exactly 0.0, 0.0, 0.0 or extremely close)
    """
    return (abs(x) > 1e-3 or abs(y) > 1e-3 or abs(z) > 1e-3)


def main():
    parser = argparse.ArgumentParser(description="Validate True vs Single Classification")
    parser.add_argument("root_dir", help="Path to pt_bursts directory")
    args = parser.parse_args()

    if not os.path.exists(args.root_dir):
        print(f"Error: Directory {args.root_dir} does not exist.")
        sys.exit(1)

    files = glob.glob(os.path.join(args.root_dir, "**", "*.pt"), recursive=True)
    print(f"🔍 Scanning {len(files)} files in {args.root_dir}...\n")

    # Metrics
    # Keys: (Label, Physical_Points_Count)
    matrix = defaultdict(int)
    errors = []

    for fpath in tqdm(files, unit="file"):
        try:
            # Load lightweight (map_location=cpu)
            data = torch.load(fpath, map_location="cpu")

            # Extract metadata
            eid = int(data.get("event_id", -1))
            label = int(data["event_type"])
            target = data["target"]  # [x1, y1, z1, x2, y2, z2]

            # Count valid physical points
            p1_valid = is_point_valid(target[0], target[1], target[2])
            p2_valid = is_point_valid(target[3], target[4], target[5])

            point_count = int(p1_valid) + int(p2_valid)

            # Update Matrix
            matrix[(label, point_count)] += 1

            # --- CRITICAL VALIDATION LOGIC ---

            # ERROR 1: Label 1 (True) but missing spatial points
            if label == 1 and point_count < 2:
                errors.append(f"CRITICAL: Event {eid} is Label 1 (True) but has {point_count} points.")

            # Note: We do NOT flag Label 0 + 2 Points as an error, because those are Randoms.
            # Note: We do NOT flag Label 0 + 1 Point as an error, because those are Singles.

        except Exception as e:
            print(f"Error reading {fpath}: {e}")

    # --- PRINT REPORT ---
    print("\n" + "=" * 50)
    print("      CLASSIFICATION VALIDATION REPORT")
    print("=" * 50)

    print("\n--- MATRIX [Label vs Physical Points] ---")
    print(f"{'Label':<10} | {'Physical Pts':<15} | {'Count':<10} | {'Verdict'}")
    print("-" * 55)

    # Sort keys for nice printing
    for (lbl, pts), count in sorted(matrix.items()):
        if lbl == 1:
            lbl_str = "1 (True)"
            if pts == 2:
                verdict = "✅ OK"
            else:
                verdict = "❌ INVALID (Broken True)"
        else:
            lbl_str = "0 (Bkg)"
            if pts == 1:
                verdict = "✅ OK (Single)"
            elif pts == 2:
                verdict = "✅ OK (Random)"
            else:
                verdict = "✅ OK (Noise)"

        print(f"{lbl_str:<10} | {pts:<15} | {count:<10,} | {verdict}")

    print("-" * 55)

    # --- SUMMARY ---
    total_class_1 = sum(v for (k, v) in matrix.items() if k[0] == 1)
    total_class_0 = sum(v for (k, v) in matrix.items() if k[0] == 0)

    correct_trues = matrix[(1, 2)]
    broken_trues = matrix[(1, 1)] + matrix[(1, 0)]

    singles = matrix[(0, 1)]
    randoms = matrix[(0, 2)]

    print(f"\nTotal Class 1 (True): {total_class_1:,}")
    print(f"  - Valid (2 pts):    {correct_trues:,}")
    print(f"  - Invalid (<2 pts): {broken_trues:,}  <-- MUST BE 0")

    print(f"\nTotal Class 0 (Bkg):  {total_class_0:,}")
    print(f"  - Singles (1 pt):   {singles:,}")
    print(f"  - Randoms (2 pts):  {randoms:,}")

    if len(errors) > 0:
        print(f"\n❌ FOUND {len(errors)} CRITICAL ERRORS:")
        for e in errors[:5]:  # Print first 5
            print(f"  {e}")
        if len(errors) > 5: print(f"  ... and {len(errors) - 5} more.")
        sys.exit(1)
    else:
        print("\n✅ SUCCESS: All Trues have 2 points and all Singles are correctly Class 0.")
        sys.exit(0)


if __name__ == "__main__":
    main()