import math
import random
from pathlib import Path
from typing import Iterator, List

import torch
from torch import distributed as dist
from torch.utils.data import Sampler
from src.utils.config import Geometry


def expand_train_input(train_dir: str, data_type: str = "train") -> List[str]:
    """Expand TRAIN env to a list of directories containing .pt files.

    Accepts either:
      - Comma-separated list of directories
      - A single directory that contains many experiment subdirectories, each
        with a `train/` or `trues/` child that holds `.pt` files
      - A path directly pointing at a `train/` or `trues/` directory itself

    Args:
        train_dir: Path or comma-separated paths to directories
        data_type: Which directories to search for. Options:
                   - "train": Only search for train/ directories
                   - "trues": Only search for trues/ directories
                   - "both": Search for both train/ and trues/ directories (default)

    Returns a sorted list of absolute directory paths to pass to PetBurstDataset.
    Raises ValueError if nothing valid is found.
    """
    # Validate data_type
    if data_type not in ["train", "trues"]:
        raise ValueError(f"data_type must be 'train', 'trues'', got: {data_type}")

    # Determine which directory names to search for
    search_dirs = [data_type]

    # Fast path: comma-separated list
    if "," in train_dir:
        paths = [p.strip() for p in train_dir.split(",") if p.strip()]
        return [str(Path(p).resolve()) for p in paths]

    root = Path(train_dir).expanduser().resolve()
    if not root.exists():
        raise ValueError(f"TRAIN path does not exist: {root}")

    # If the provided path is already a target directory with .pt files, use it
    if root.is_dir() and root.name in search_dirs:
        has_pts = any(child.suffix == ".pt" for child in root.glob("*.pt"))
        if has_pts:
            return [str(root)]

    # If it's a directory, scan its immediate children for matching directories
    if root.is_dir():
        candidates: List[Path] = []
        for child in root.iterdir():
            if not child.is_dir():
                continue
            # Check specified directory types
            for dirname in search_dirs:
                tdir = child / dirname
                if tdir.is_dir() and any(f.suffix == ".pt" for f in tdir.glob("*.pt")):
                    candidates.append(tdir.resolve())

        # If none found at depth=1, do a recursive search
        if not candidates:
            for dirname in search_dirs:
                for tdir in root.rglob(dirname):
                    if tdir.is_dir() and any(f.suffix == ".pt" for f in tdir.glob("*.pt")):
                        candidates.append(tdir.resolve())

        if candidates:
            # de-duplicate & sort
            uniq = sorted({str(p) for p in candidates})
            return uniq

    # If it's a file path or a dir without a matching structure
    raise ValueError(
        f"No valid '{'/'.join(search_dirs)}' directories with .pt files were found under: {str(root)}"
    )


class DistributedWeightedRandomSampler(Sampler[int]):
    """
    DDP-aware variant of WeightedRandomSampler.

    - Balances classes using per-sample weights.
    - Each rank gets a different subset of indices.
    - Uses torch.distributed.scatter internally.

    With backend="nccl", all collective ops must run on CUDA tensors.
    With backend="gloo", they run on CPU tensors.
    """

    def __init__(
            self,
            weights: torch.Tensor,
            num_samples: int | None = None,
            replacement: bool = True
    ) -> None:
        super().__init__()
        if weights.dim() != 1:
            raise ValueError("weights must be a 1D tensor")

        if not (dist.is_available() and dist.is_initialized()):
            raise RuntimeError(
                "DistributedWeightedRandomSampler requires torch.distributed to be initialized"
            )

        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()

        # Store weights on CPU; we’ll move them to the right device inside __iter__
        self.weights = weights.detach().clone().to(dtype=torch.double, device="cpu")
        self.replacement = replacement

        # Number of samples *per rank* per epoch
        if num_samples is None:
            total_len = len(self.weights)
            self.num_samples = (total_len + self.world_size - 1) // self.world_size
        else:
            self.num_samples = num_samples

        # Decide which device to use for collectives
        backend = dist.get_backend()
        if str(backend).lower() == "nccl":
            # NCCL requires CUDA tensors
            if not torch.cuda.is_available():
                raise RuntimeError("NCCL backend requires CUDA, but no CUDA device is available")
            self.device = torch.device("cuda", torch.cuda.current_device())
        else:
            # gloo / mpi etc can work on CPU
            self.device = torch.device("cpu")

    def __iter__(self) -> Iterator[int]:
        # Total number of samples drawn *globally* this epoch
        total_samples = self.num_samples * self.world_size

        # rank 0 draws the global index set
        if self.rank == 0:
            # Move weights to correct device for multinomial
            weights_dev = self.weights.to(self.device)

            # global_indices: (total_samples,) on self.device
            global_indices = torch.multinomial(
                weights_dev,
                total_samples,
                self.replacement,
            )

            # Split into world_size equal chunks
            chunks: List[torch.Tensor] = []
            for r in range(self.world_size):
                start = r * self.num_samples
                end = start + self.num_samples
                # clone so chunk tensors are independent
                chunks.append(global_indices[start:end].clone())
        else:
            chunks = None

        # Allocate output buffer on the correct device
        out = torch.empty(self.num_samples, dtype=torch.long, device=self.device)

        # Scatter chunks from rank 0 to all ranks
        # scatter_list must be a list of tensors on src rank, and None/[] on others
        dist.scatter(
            out,
            scatter_list=chunks if self.rank == 0 else [],
            src=0,
        )

        # DataLoader expects CPU indices; move back to CPU and convert to Python ints
        indices = out.cpu().tolist()
        return iter(indices)

    def __len__(self) -> int:
        # length from perspective of a single rank
        return self.num_samples


def get_label_from_sample(sample) -> int:
    """
    Extract the event_type label from a PetBurstDataset sample.
    Adjust here if your dataset returns something slightly different.
    """
    # Expect a dict-like sample with "event_type"
    evt = sample["event_type"]
    if isinstance(evt, torch.Tensor):
        return int(evt.item())
    return int(evt)


def add_doi_channel(burst: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    """
    Calculates and appends a Depth-of-Interaction (DOI) map to the burst.

    Formula: DOI = (Inner - Outer) / (Inner + Outer + epsilon)

    Args:
        burst: Tensor of shape (T, 2, H, W). Channel 0 is Inner, 1 is Outer.
        epsilon: Small constant to prevent division by zero.

    Returns:
        Tensor of shape (T, 3, H, W). The 3rd channel is the DOI map.
    """
    # Ensure input is a floating point for division
    if not burst.is_floating_point():
        burst = burst.float()

    # 1. Extract Inner and Outer channels
    # Keep dim=1 to allow concatenation later: (T, 1, H, W)
    inner = burst[:, 0:1, :, :]
    outer = burst[:, 1:2, :, :]

    # 2. Calculate DOI Map
    # Result ranges from +1 (Pure Inner) to -1 (Pure Outer). 0 is the Midline.
    numerator = inner - outer
    denominator = inner + outer + epsilon
    doi_map = numerator / denominator

    # 3. Concatenate along the channel dimension
    # (T, 2, H, W) + (T, 1, H, W) -> (T, 3, H, W)
    burst_with_doi = torch.cat([inner, outer, doi_map], dim=1)

    return burst_with_doi


def collate_with_radius_measure(batch):
    """Filter out samples with invalid target coordinates (r < 200mm)"""
    # Stack all samples
    bursts = torch.stack([item["burst"] for item in batch])
    targets = torch.stack([item["target"] for item in batch])
    event_types = torch.stack([item["event_type"] for item in batch])
    is_pes = torch.stack([item["is_pe"] for item in batch])
    ids = [item["id"] for item in batch]

    # Calculate radii for both points
    r1_target = torch.sqrt(targets[:, 0]**2 + targets[:, 1]**2)
    r2_target = torch.sqrt(targets[:, 3]**2 + targets[:, 4]**2)

    # Detector is at ~235mm, filter out bad labels (0,0,0) closer than 200mm
    valid_mask = (r1_target >= Geometry.R_INNER) & (r2_target >= Geometry.R_INNER)

    # Apply mask
    return {
        "burst": bursts[valid_mask],
        "target": targets[valid_mask],
        "event_type": event_types[valid_mask],
        "is_pe": is_pes[valid_mask],
        "id": [ids[i] for i, v in enumerate(valid_mask) if v]
    }

def get_batch_mask(event_type, is_pe, filter_mode):
    """
    Returns a boolean mask based on the training phase.
    """
    if filter_mode == "singles_pe":
        # Phase 1: Clean Singles (Photoelectric only)
        # Learn features on the highest quality single-point data
        return (event_type == 0) & (is_pe == 1)

    elif filter_mode == "trues_pe":
        # Phase 2: Clean Coincidences (Photoelectric only)
        # Learn to separate two points using high-quality data
        return (event_type == 1) & (is_pe == 1)

    elif filter_mode == "all":
        # Phase 3: Everything (Compton, Scatters, etc.)
        # Robustness training
        return torch.ones_like(event_type, dtype=torch.bool)

    else:
        # Fallback for "singles" or "trues" without PE restriction
        if filter_mode == "singles": return event_type == 0
        if filter_mode == "trues": return event_type == 1

        raise ValueError(f"Unknown filter mode: {filter_mode}")


class RandomCylindricalRoll:
    def __init__(self, prob=0.8, width_pixels=Geometry.CIRCUMFERENTIAL):
        self.prob = prob
        self.width_pixels = width_pixels

    def __call__(self, burst, target):
        if random.random() < self.prob:
            shift = random.randint(0, self.width_pixels - 1)
            burst = torch.roll(burst, shifts=shift, dims=-1)

            # Rotate Target (x,y)
            angle_rad = (shift / self.width_pixels) * 2 * math.pi
            c, s = math.cos(angle_rad), math.sin(angle_rad)

            # P1
            x1, y1 = target[0].item(), target[1].item()
            target[0] = x1 * c - y1 * s
            target[1] = x1 * s + y1 * c

            # P2 (rotate regardless of being 0 or real)
            x2, y2 = target[3].item(), target[4].item()
            target[3] = x2 * c - y2 * s
            target[4] = x2 * s + y2 * c
        return burst, target