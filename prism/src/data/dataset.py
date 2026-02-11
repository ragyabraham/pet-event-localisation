# petnet/dataset.py
import os
import glob
import re
from typing import Dict, List, Union

import torch
from torch.utils.data import Dataset

from src.data.utils import add_doi_channel


class PetBurstDataset(Dataset):
    """
    Expects one or more directories or glob patterns of .pt files; each file is a dict with:
      - "burst": FloatTensor [T, 2, H, W] (2 = inner, outer)
      - "target": FloatTensor [6] (x1,y1,z1, x2,y2,z2)
    You can pass a single directory, a single glob string, a path to a single .pt file,
    or a list of any mix of these. Set `recursive=True` to search subdirectories.
    """

    def __init__(
            self,
            roots: Union[str, List[str]],
            transform=None,
            glob_pattern: str = "*.pt",  # 1. Broad search (finds all candidates)
            regex_filter: str = None,  # 2. Precise filter (your specific regex)
            recursive: bool = True,
            transfer_learning: bool = False
    ):
        # Normalise roots to a list
        root_list: List[str] = [roots] if isinstance(roots, str) else list(roots)

        # Expand all roots into a flat, sorted, deduplicated list of files
        files: List[str] = []
        for r in root_list:
            if os.path.isdir(r):
                # Use the broad glob pattern (e.g., "*.pt") to get candidates
                search = os.path.join(r, "**", glob_pattern) if recursive else os.path.join(r, glob_pattern)
                files.extend(glob.glob(search, recursive=recursive))
            elif os.path.isfile(r):
                # For direct file paths, check if they match the extension
                if r.endswith(glob_pattern.replace("*", "")):
                    files.append(r)
            else:
                files.extend(glob.glob(r, recursive=True))

        # Canonicalise + de-duplicate
        files = sorted({os.path.abspath(p) for p in files})

        # --- NEW: Apply Regex Filtering ---
        if regex_filter:
            compiled_re = re.compile(regex_filter)
            # Keep a file only if the filename (not the full path) matches the regex
            files = [f for f in files if compiled_re.search(os.path.basename(f))]

        if len(files) == 0:
            raise FileNotFoundError(
                f"No files found for inputs: {root_list} with glob='{glob_pattern}' and regex='{regex_filter}'"
            )

        self.files = files
        self.transform = transform
        self.tf = transfer_learning

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        path = self.files[idx]
        data = torch.load(path, map_location="cpu")
        burst: torch.Tensor = data["burst"].float()  # [T, 2, H, W]
        target: torch.Tensor = data["target"].float()  # [6]
        event_type: torch.Tensor = data["event_type"].long()  # scalar
        is_pe: torch.Tensor = data["is_pe"]

        if self.transform is not None:
            # Allow transforms that operate on both burst and target
            try:
                burst, target = self.transform(burst, target)
            except TypeError:
                burst = self.transform(burst)

        burst = add_doi_channel(burst)

        # Enforce shape guarantees (B not included here)
        assert burst.dim() == 4 and burst.size(1) == 3, (
            "Burst must be [T, 2, H, W]"
        )
        assert target.numel() == 6, (
            "Target must be 6 floats: x1,y1,z1,x2,y2,z2"
        )

        return {
            "burst": burst,  # [T, 2, H, W]
            "target": target,  # [6]
            "event_type": event_type,  # scalar
            "is_pe": is_pe,
            "id": os.path.basename(path),
        }


