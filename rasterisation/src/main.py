import concurrent.futures
import pathlib
import time
import logging
import numpy as np
import pandas as pd
import os
import sys
import torch
import csv
import hashlib
from threading import Lock
from typing import List, Optional, Dict
from functools import partial
from tqdm import tqdm
from dotenv import load_dotenv

# Imports from your project
from utils.gamma import gamma_class_temporal
from utils.sipm_binning import sipm_binning
from utils.os_utils import create_working_dir, get_dat_files


# -----------------------------------------------------------------------------
# UTILITIES FROM CSV_TO_PT
# -----------------------------------------------------------------------------

def deterministic_val_split(event_id: int, val_split: float, seed: int) -> bool:
    """Deterministic assignment: hash(seed,event_id) -> [0,1)."""
    key = f"{seed}:{event_id}".encode("utf-8")
    h = hashlib.sha1(key).hexdigest()
    frac = int(h[:8], 16) / 0xFFFFFFFF
    return frac < val_split


def safe_save(data, path: pathlib.Path, retries=3):
    """Atomic write to fix inline_container errors."""
    tmp_path = path.with_suffix(f".tmp_{os.getpid()}_{time.time_ns()}")
    for i in range(retries):
        try:
            torch.save(data, tmp_path)
            tmp_path.rename(path)
            return
        except Exception as e:
            if tmp_path.exists():
                tmp_path.unlink()
            if i == retries - 1:
                raise e
            time.sleep(0.1 * (i + 1))


def apply_burst_augmentation(burst: torch.Tensor, target: torch.Tensor | None = None) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Applies Z-flip augmentation."""
    if target is None:
        return None, None

    # Flip image along H (Z-axis)
    burst = torch.flip(burst, dims=[-2])

    # Flip target Z coordinates
    x1, y1, z1 = target[0], target[1], target[2]
    x2, y2, z2 = target[3], target[4], target[5]

    target = torch.stack([x1, y1, -z1, x2, y2, -z2]).to(target.dtype)
    return burst, target


# -----------------------------------------------------------------------------
# CORE LOGIC
# -----------------------------------------------------------------------------

def process_single_burst(
        event_id: int,
        start_frame_idx: int,
        gt_row: np.ndarray,
        H_slice: np.ndarray,
        radius_slice: np.ndarray,
        frame_length: float,
        mid_r: float,
        energy_optical: float,
        fixed_t: int,
        target_w: int,
        z_edges: np.ndarray,
        theta_edges_inner: np.ndarray,
        theta_edges_outer: np.ndarray,
        inner_shape: tuple,  # (H, W)
        outer_shape: tuple,  # (H, W)
        out_dir_train: pathlib.Path,
        out_dir_val: pathlib.Path,
        val_split: float,
        seed: int,
        cut_off: int
) -> tuple[int, int, int]:
    """
    Processes a single event:
    1. Extracts hits for T frames.
    2. Bins them into images.
    3. Stacks into tensor.
    4. Saves to disk.
    """
    try:
        # 1. Prepare Burst Container
        # Shape: [T, 2, H, W] -> 2 channels (Inner, Outer)
        T = fixed_t
        H_in, W_in = inner_shape
        H_out, W_out = outer_shape

        # We will resize everything to target_w later, but first bin naturally
        burst_frames_in = []
        burst_frames_out = []

        # 2. Iterate through the frames of this burst
        # H_slice contains ALL hits for the duration of the burst (sorted by time)
        # We need to sub-slice it per frame

        # Calculate start time of the event
        t0_event = start_frame_idx * frame_length

        # Pre-calculate frame indices for the slice relative to the event start
        # This avoids repeatedly searching the whole H_slice
        if len(H_slice) > 0:
            # Local frame index: 0, 1, 2...
            local_frame_indices = np.floor((H_slice[:, 0] - t0_event) / frame_length).astype(int)
        else:
            local_frame_indices = np.array([])

        for t in range(T):
            # Extract hits for this specific frame t
            mask = local_frame_indices == t
            if not np.any(mask):
                # Empty frame (Zero padding)
                im_in = np.zeros((H_in, W_in), dtype=np.float32)
                im_out = np.zeros((H_out, W_out), dtype=np.float32)
            else:
                frame_hits = H_slice[mask]
                frame_r = radius_slice[mask]

                # --- Bin Inner ---
                mask_in = (frame_hits[:, 1] <= energy_optical) & (frame_r < mid_r)
                innerDet = frame_hits[mask_in]

                if len(innerDet) > 0:
                    theta_in = np.arctan2(innerDet[:, 8], innerDet[:, 7])
                    z_in = innerDet[:, 9]
                    c_in = (theta_in + 2 * np.pi) % (2 * np.pi)

                    hist, _, _ = np.histogram2d(z_in, c_in, bins=[z_edges, theta_edges_inner])
                    im_in = hist if hist.shape == (H_in, W_in) else np.zeros((H_in, W_in))
                else:
                    im_in = np.zeros((H_in, W_in), dtype=np.float32)

                # --- Bin Outer ---
                mask_out = (frame_hits[:, 1] <= energy_optical) & (frame_r > mid_r)
                outerDet = frame_hits[mask_out]

                if len(outerDet) > 0:
                    theta_out = np.arctan2(outerDet[:, 8], outerDet[:, 7])
                    z_out = outerDet[:, 9]
                    c_out = (theta_out + 2 * np.pi) % (2 * np.pi)

                    hist, _, _ = np.histogram2d(z_out, c_out, bins=[z_edges, theta_edges_outer])
                    im_out = hist if hist.shape == (H_out, W_out) else np.zeros((H_out, W_out))
                else:
                    im_out = np.zeros((H_out, W_out), dtype=np.float32)

            burst_frames_in.append(im_in)
            burst_frames_out.append(im_out)

        # 3. Stack and Align Widths
        # Convert to arrays: [T, H, W]
        stack_in = np.stack(burst_frames_in, axis=0)
        stack_out = np.stack(burst_frames_out, axis=0)

        # Circular Resample to target_w if needed
        def resample(x, tw):
            curr_w = x.shape[2]
            if curr_w == tw: return x
            # Simple circular linear interp
            src_pos = np.arange(tw, dtype=np.float32) * (curr_w / float(tw))
            i0 = np.floor(src_pos).astype(np.int64)
            frac = src_pos - i0
            i1 = (i0 + 1) % curr_w
            return (1.0 - frac) * x[..., i0] + frac * x[..., i1]

        stack_in = resample(stack_in, target_w)
        stack_out = resample(stack_out, target_w)

        # Final Burst: [T, 2, H, W]
        burst = np.stack([stack_in, stack_out], axis=1).astype(np.float32)

        # 4. Sparsity Check
        if cut_off > 0:
            if np.count_nonzero(burst) < cut_off:
                return 0, 0, 1  # Skipped

        # 5. Prepare Target
        # gt_row structure: [event_id, trk_id, event_type, is_pe, x1, y1, z1, x2, y2, z2] (based on gamma.py output)
        # Note: adjust indices based on your exact gamma.py output format
        # Assuming gamma.py writes: [idx, eid, trk, type, is_pe, x1,y1,z1,x2,y2,z2]
        event_type = int(gt_row[3])
        is_pe = int(gt_row[4])
        coords = gt_row[5:11].astype(np.float32)

        target_tensor = torch.from_numpy(coords)
        burst_tensor = torch.from_numpy(burst)

        # 6. Save (Train/Val Split)
        is_val = deterministic_val_split(event_id, val_split, seed)
        save_dir = out_dir_val if is_val else out_dir_train

        # Save Standard
        sample = {
            "burst": burst_tensor,
            "target": target_tensor,
            "event_type": torch.tensor(event_type),
            "is_pe": torch.tensor(is_pe),
            "event_id": event_id
        }
        safe_save(sample, save_dir / f"{event_id}_{event_type}_{is_pe}.pt")

        saved_train = 1 if not is_val else 0
        saved_val = 1 if is_val else 0

        # 7. Augment (Only for True events in Train/Val)
        # if event_type == 1:
            # b_aug, t_aug = apply_burst_augmentation(burst_tensor, target_tensor)
            # if b_aug is not None:
            #     sample_aug = {
            #         "burst": b_aug,
            #         "target": t_aug,
            #         "event_type": torch.tensor(event_type),
            #         "is_pe": torch.tensor(is_pe),
            #         "event_id": event_id
            #     }
            #     safe_save(sample_aug, save_dir / f"{event_id}_{event_type}_{is_pe}_aug.pt")
            #     if is_val:
            #         saved_val += 1
            #     else:
            #         saved_train += 1

        return saved_train, saved_val, 0

    except Exception as e:
        logging.error(f"Error processing event {event_id}: {e}")
        return 0, 0, 0  # treat error as skip for counter-simplicity


# -----------------------------------------------------------------------------
# WRAPPER FOR THREADPOOL
# -----------------------------------------------------------------------------
def task_wrapper(args):
    return process_single_burst(*args)


# -----------------------------------------------------------------------------
# MAIN PIPELINE
# -----------------------------------------------------------------------------

def integrated_pipeline(
        hits: np.ndarray,
        frame_length: float,
        base_directory: str,
        num_workers: int
):
    load_dotenv()

    # --- Configuration ---
    energy_optical = 5e-6
    energy_cutoff = float(os.getenv("OPTICAL_CUTOFF", 0.00001))

    num_pix_Z = int(os.getenv("NUM_PIX_Z", 1))
    num_pix_C = int(os.getenv("NUM_PIX_C", 1))
    dim_pix = [int(os.getenv("PIX_DIM", 12)), int(os.getenv("PIX_DIM", 12))]

    fixed_t = int(os.getenv("FIXED_T", 3))
    target_w = int(os.getenv("TARGET_W", 207))
    val_split = float(os.getenv("VAL_SPLIT", 0.2))
    seed = int(os.getenv("SEED", 1337))
    cut_off = int(os.getenv("BURST_MIN_NONZERO", 200))

    # --- Geometry & Binning ---
    num_sipm_Z = 21
    num_sipm_out = 124
    num_sipm_in = 104
    border_width = 1
    sipm_space_out = 0.1
    sipm_space_in = 0.221
    spacing_in = border_width + sipm_space_in / 2
    spacing_out = border_width + sipm_space_out / 2

    side_length_inner = num_pix_C * dim_pix[1] + 2 * spacing_in
    side_length_outer = num_pix_C * dim_pix[1] + 2 * spacing_out

    length_C_in = num_sipm_in * side_length_inner
    length_C_out = num_sipm_out * side_length_outer
    length_Z = num_sipm_Z * (num_pix_Z * dim_pix[0] + 2 * spacing_out)

    number_of_edges_along_the_inner_circumference = sipm_binning(num_sipm_in, num_pix_C, dim_pix[1], length_C_in, spacing_in)
    number_of_edges_along_the_outer_circumference = sipm_binning(num_sipm_out, num_pix_C, dim_pix[1], length_C_out, spacing_out)
    number_of_sipm_edges_along_the_axial_length = sipm_binning(num_sipm_Z, num_pix_Z, dim_pix[0], length_Z, spacing_out)

    inner_circumference_length = len(number_of_edges_along_the_inner_circumference) - 1
    outer_circumference_length = len(number_of_edges_along_the_outer_circumference) - 1
    inner_axial_length = outer_axial_length = len(number_of_sipm_edges_along_the_axial_length) - 1

    z_edges_global = number_of_sipm_edges_along_the_axial_length
    theta_edges_inner_global = np.linspace(0.0, 2.0 * np.pi, inner_circumference_length + 1)
    theta_edges_outer_global = np.linspace(0.0, 2.0 * np.pi, outer_circumference_length + 1)

    inner_shape = (inner_axial_length, inner_circumference_length)
    outer_shape = (outer_axial_length, outer_circumference_length)

    # --- Setup Directories ---
    out_dir_root = pathlib.Path(base_directory) / "pt_bursts"
    out_dir_train = out_dir_root / "train"
    out_dir_val = out_dir_root / "val"
    out_dir_train.mkdir(parents=True, exist_ok=True)
    out_dir_val.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Sorting ---
    logging.info("Sorting hits by timestamp...")
    sort_idx = np.argsort(hits[:, 0])
    hits = hits[sort_idx]

    # --- Step 2: Ground Truth Generation ---
    logging.info("Generating Ground Truth (Events)...")
    # This writes groundtruth.csv. We let it write to disk because it's metadata (small)
    # and useful to have a record of.
    _ = gamma_class_temporal(hits, frame_length, energy_cutoff, base_directory,
                             inner_circumference_length, inner_axial_length, as_image=False, debug=False)

    gt_path = f'{base_directory}/groundtruth.csv'
    if not os.path.exists(gt_path):
        raise RuntimeError("Ground truth file was not created.")

    # Read GT back into memory
    # CSV Format: [frame_idx, eid, trk, type, is_pe, x1, y1, z1, x2, y2, z2]
    # We want a list of events to process
    logging.info("Loading Ground Truth...")
    gt_data = []
    with open(gt_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if row:
                gt_data.append(np.array(row, dtype=np.float32))

    logging.info(f"Found {len(gt_data)} events to process.")

    # --- Step 3: Pre-calc Radius & Midpoint ---
    radius_all = np.sqrt(np.square(hits[:, 7]) + np.square(hits[:, 8]))
    mid_r = np.mean(radius_all)

    # --- Step 4: Parallel Burst Processing ---
    # We iterate over events in GT. For each event, we need the slice of hits corresponding to T frames.

    # Helper to find global indices for time windows
    all_times = hits[:, 0]

    def arg_generator():
        for row in gt_data:
            start_frame_idx = int(row[0])
            event_id = int(row[1])  # Or usually row[0] is used as ID if unique. Adjust as needed.
            # Ideally event_id in GT matches the frame_idx or is unique.
            # Your separate_true_events.py logic uses frame_idx as the ID often.
            # Let's use the first column (frame_idx) as the unique ID for file naming
            unique_id = int(row[0])

            start_time = start_frame_idx * frame_length
            end_time = (start_frame_idx + fixed_t) * frame_length

            # Binary search for the start and end indices of this burst window
            # Since we are processing sorted hits, searchsorted is O(log N)
            idx_start = np.searchsorted(all_times, start_time, side='left')
            idx_end = np.searchsorted(all_times, end_time, side='left')

            # Slice O(1)
            h_slice = hits[idx_start:idx_end]
            r_slice = radius_all[idx_start:idx_end]

            yield (
                unique_id,
                start_frame_idx,
                row,
                h_slice,
                r_slice,
                frame_length,
                mid_r,
                energy_optical,
                fixed_t,
                target_w,
                z_edges_global,
                theta_edges_inner_global,
                theta_edges_outer_global,
                inner_shape,
                outer_shape,
                out_dir_train,
                out_dir_val,
                val_split,
                seed,
                cut_off
            )

    logging.info(f"Starting Burst Generation with {num_workers} workers...")

    total_train = 0
    total_val = 0
    total_skipped = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        for n_train, n_val, n_skip in tqdm(executor.map(task_wrapper, arg_generator(), chunksize=32),
                                           total=len(gt_data),
                                           desc="Building Bursts"):
            total_train += n_train
            total_val += n_val
            total_skipped += n_skip

    logging.info(f"Complete. Train: {total_train}, Val: {total_val}, Skipped: {total_skipped}")


# -----------------------------------------------------------------------------
# ENTRY POINT
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

    out_dir = os.getenv("OUT_DIR", "./src/output")
    num_workers = int(os.getenv("NUMBER_WORKERS", os.cpu_count()))

    num_pix_Z = int(os.getenv("NUM_PIX_Z", 1))
    num_pix_C = int(os.getenv("NUM_PIX_C", 1))

    start_time = time.time()

    path = sys.argv[1] if len(sys.argv) > 1 else os.getenv("FILE_PATH")
    if not path:
        raise RuntimeError("Input file path required.")


    def run_job(p):
        logging.info(f"Processing: {p}")
        directory = create_working_dir(out_dir, num_pix_C, num_pix_Z)
        hits = pd.read_csv(p, comment='#', delimiter=',', header=0).to_numpy()
        integrated_pipeline(hits, 50e-9, directory, num_workers)


    if pathlib.Path(path).is_dir():
        files = get_dat_files(path)
        for f in files:
            run_job(f)
    else:
        run_job(path)

    logging.info(f"Total Runtime: {round(time.time() - start_time, 2)}s")