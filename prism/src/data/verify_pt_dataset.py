#!/usr/bin/env python3
"""
verify_pt_dataset.py

Scan train/ and val/ directories of .pt files and print a health report:
- Number of events, duplicate IDs, overlap between splits
- Burst tensor shape consistency (expects [T, 2, H, W])
- Value stats (min/max/mean/std), dtype
- T (frames per burst) distribution
- Fraction of events with missing outer point (x2,y2,z2 == 0)
- Optional geometry check: |||(x1,y1)|| - R_INNER| and |||(x2,y2)|| - R_OUTER|

Usage:
  python verify_pt_dataset.py --root pt_bursts --sample 10 --check_geometry
"""

import os
import math
import argparse
from glob import glob
from collections import Counter, defaultdict

import torch

# ---- Adjust to your scanner geometry if you use --check_geometry ----
R_INNER = 235.422  # mm
R_OUTER = 278.296  # mm
R_TOL   = 10.0     # mm tolerance window for a quick sanity check

def list_pt_files(d):
    return sorted([p for p in glob(os.path.join(d, "*.pt"))])

def safe_load(path):
    # map_location='cpu' to avoid GPU pulls; weights_only not applicable to plain dicts
    return torch.load(path, map_location="cpu")

def vec_norm2(x, y):
    return math.sqrt(float(x) * float(x) + float(y) * float(y))

def summarize_folder(folder, sample_n=10, check_geometry=False):
    files = list_pt_files(folder)
    n = len(files)
    if n == 0:
        return {"count": 0, "ok": False, "msg": f"No .pt files in {folder}"}

    # Running tallies
    T_hist = Counter()
    H_set, W_set, C_set = set(), set(), set()
    dtype_set = set()
    ids = set()
    n_bad_shape = 0
    n_nan_inf = 0
    n_bad_keys = 0
    n_missing_outer = 0  # (x2,y2,z2) == 0
    n_missing_inner = 0  # (x1,y1,z1) == 0 (should be rare)
    geo_errs_inner = []
    geo_errs_outer = []

    # value stats (streaming-ish)
    vmin = float("inf")
    vmax = float("-inf")
    vsum = 0.0
    vsqsum = 0.0
    vcount = 0

    # sample a few detailed prints
    sample_paths = files[::max(1, len(files)//max(1, sample_n))][:sample_n]

    for i, path in enumerate(files, 1):
        try:
            data = safe_load(path)
        except Exception as e:
            n_bad_shape += 1
            continue

        if not all(k in data for k in ("burst", "target")):
            n_bad_keys += 1
            continue

        burst = data["burst"]
        target = data["target"]
        event_id = int(data.get("event_id", -1))
        if event_id >= 0:
            ids.add(event_id)

        # Basic shape checks
        if burst.ndim != 4:
            n_bad_shape += 1
            continue
        T, C, H, W = burst.shape
        C_set.add(int(C)); H_set.add(int(H)); W_set.add(int(W))
        T_hist[int(T)] += 1
        dtype_set.add(str(burst.dtype))

        if C != 2:
            n_bad_shape += 1
            continue
        if target.numel() != 6:
            n_bad_shape += 1
            continue

        # NaN/Inf check
        b_ok = torch.isfinite(burst).all().item()
        t_ok = torch.isfinite(target).all().item()
        if not (b_ok and t_ok):
            n_nan_inf += 1

        # Value stats (approximate: sample up to 2 frames to keep things snappy)
        # You can comment this sampling out to scan all values (slower).
        with torch.no_grad():
            if T > 0:
                # take first frame of each side
                sample_vals = burst[0].flatten()  # (2, H, W) -> flat
                vmin = min(vmin, float(sample_vals.min()))
                vmax = max(vmax, float(sample_vals.max()))
                vsum += float(sample_vals.float().sum())
                vsqsum += float((sample_vals.float() ** 2).sum())
                vcount += sample_vals.numel()

        # Missing-point tallies
        x1, y1, z1, x2, y2, z2 = [float(v) for v in target.tolist()]
        if abs(x2) < 1e-12 and abs(y2) < 1e-12 and abs(z2) < 1e-12:
            n_missing_outer += 1
        if abs(x1) < 1e-12 and abs(y1) < 1e-12 and abs(z1) < 1e-12:
            n_missing_inner += 1

        # Optional geometry checks
        if check_geometry:
            r1 = vec_norm2(x1, y1)
            r2 = vec_norm2(x2, y2)
            geo_errs_inner.append(abs(r1 - R_INNER))
            geo_errs_outer.append(abs(r2 - R_OUTER))

    # Aggregate stats
    mean_val = vsum / max(1, vcount)
    var_val = vsqsum / max(1, vcount) - mean_val * mean_val
    std_val = math.sqrt(max(0.0, var_val))

    report = {
        "count": n,
        "ids_count": len(ids),
        "dtype": sorted(dtype_set),
        "C_set": sorted(C_set),
        "H_set": sorted(H_set),
        "W_set": sorted(W_set),
        "T_hist_top": T_hist.most_common(10),
        "n_bad_shape": n_bad_shape,
        "n_bad_keys": n_bad_keys,
        "n_nan_inf": n_nan_inf,
        "value_min": vmin if vcount > 0 else None,
        "value_max": vmax if vcount > 0 else None,
        "value_mean_est": mean_val if vcount > 0 else None,
        "value_std_est": std_val if vcount > 0 else None,
        "missing_outer_frac": n_missing_outer / max(1, n),
        "missing_inner_frac": n_missing_inner / max(1, n),
    }

    if check_geometry and len(geo_errs_inner) > 0:
        import statistics as st
        report.update({
            "inner_radius_err_mm": {
                "mean": st.mean(geo_errs_inner),
                "median": st.median(geo_errs_inner),
                "p95": st.quantiles(geo_errs_inner, n=100)[94],
                "over_tol_frac": sum(e > R_TOL for e in geo_errs_inner)/len(geo_errs_inner),
            },
            "outer_radius_err_mm": {
                "mean": st.mean(geo_errs_outer),
                "median": st.median(geo_errs_outer),
                "p95": st.quantiles(geo_errs_outer, n=100)[94],
                "over_tol_frac": sum(e > R_TOL for e in geo_errs_outer)/len(geo_errs_outer),
            },
        })

    report["ok"] = (report["n_bad_shape"] == 0 and report["n_bad_keys"] == 0)
    return report, set(ids)

def print_report(name, rep):
    print(f"\n=== [{name}] ===")
    print(f"files: {rep['count']} | unique_ids: {rep['ids_count']}")
    print(f"bad_shape: {rep['n_bad_shape']} | bad_keys: {rep['n_bad_keys']} | nan/inf: {rep['n_nan_inf']}")
    print(f"dtype(s): {rep['dtype']}")
    print(f"C values (should be [2]): {rep['C_set']}")
    print(f"H set: {rep['H_set']}")
    print(f"W set: {rep['W_set']}")
    print(f"T (frames) top-10: {rep['T_hist_top']}")
    print(f"value ~ min/mean±std/max: {rep['value_min']:.3f} / {rep['value_mean_est']:.3f} ± {rep['value_std_est']:.3f} / {rep['value_max']:.3f}" if rep['value_min'] is not None else "value stats: (skipped)")
    print(f"missing_outer_frac: {rep['missing_outer_frac']:.3f} | missing_inner_frac: {rep['missing_inner_frac']:.3f}")
    if "inner_radius_err_mm" in rep:
        ir, or_ = rep["inner_radius_err_mm"], rep["outer_radius_err_mm"]
        print(f"inner | mean={ir['mean']:.2f} med={ir['median']:.2f} p95={ir['p95']:.2f} over_tol={ir['over_tol_frac']:.2%}")
        print(f"outer | mean={or_['mean']:.2f} med={or_['median']:.2f} p95={or_['p95']:.2f} over_tol={or_['over_tol_frac']:.2%}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Root with train/ and val/ subfolders")
    ap.add_argument("--sample", type=int, default=10, help="How many files to sample for quick stats")
    ap.add_argument("--check_geometry", action="store_true", help="Check that (x1,y1)~R_INNER and (x2,y2)~R_OUTER")
    args = ap.parse_args()

    train_dir = os.path.join(args.root, "train")
    val_dir   = os.path.join(args.root, "val")

    train_rep, train_ids = summarize_folder(train_dir, sample_n=args.sample, check_geometry=args.check_geometry)
    val_rep,   val_ids   = summarize_folder(val_dir,   sample_n=args.sample, check_geometry=args.check_geometry)

    print_report("TRAIN", train_rep)
    print_report("VAL",   val_rep)

    # Cross-split sanity check
    overlap = train_ids & val_ids
    print(f"\n=== SPLIT CHECK ===")
    print(f"overlap_ids: {len(overlap)} (should be 0)")
    if len(overlap) > 0:
        sample = list(sorted(overlap))[:10]
        print("example overlaps:", sample)

    # Final verdict
    ok = train_rep["ok"] and val_rep["ok"] and len(overlap) == 0
    print(f"\nVERDICT: {'OK ✅' if ok else 'ISSUES FOUND ⚠️'}")

if __name__ == "__main__":
    main()