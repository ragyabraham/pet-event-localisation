# PET Point Localiser

Deep learning techniques for event localisation in a continuous shell monolithic PET scintillator.

---

## Table of Contents
- [Problem Statement](#problem-statement)
- [Scanner Geometry](#scanner-geometry)
- [Physics Background](#physics-background)
- [Data Structure](#data-structure)
- [Pipeline Overview](#pipeline-overview)
- [Model Architectures](#model-architectures)
- [Training](#training)
- [Usage](#usage)

---

## Problem Statement

This project aims to **localise the first point of interaction of gamma photons** in a continuous monolithic PET scintillator using deep learning. The model predicts interaction coordinates in polar format (r, θ) for both sides of the scanner, along with event type classification (true coincidence vs. singles/random).

**Objective**: Given optical photon shower patterns captured by silicon photomultipliers, predict within a few millimeters tolerance:
- First interaction point on inner detector: `(r1, θ1)`
- First interaction point on outer detector: `(r2, θ2)`
- Event type: `true coincidence (1)` or `singles/random (0)`

---

## Scanner Geometry

The PET scanner is simulated using **OpenGate** (based on Geant4) with the following specifications:

| Parameter | Value |
|-----------|-------|
| Material | Nano-composite scintillator |
| Outer Diameter | 278.296 mm |
| Inner Diameter | 235.422 mm |
| Shell Thickness | 42.874 mm |
| Axial Length | 296 mm (z ∈ [-148, +148]) |
| Radioactive Source | F-18 in water phantom |

**Coordinate System**:
- Cylindrical geometry: scanner is unrolled to 2D frames
- Origin at scanner center
- z = 0 at axial midpoint

---

## Physics Background

### Gamma Interaction Types

When a 511 keV gamma photon interacts with the scintillator:

1. **Photoelectric Effect**: Complete energy absorption → large optical shower
2. **Compton Scatter**: Partial energy deposition → photon deflection
   - Photon may escape, undergo further Compton scattering, or photoelectric absorption

### Event Types

PET annihilation events are classified as:

1. **True Coincidence** (Class 1): Both gamma photons from same annihilation hit detector within coincidence window
2. **Singles** (Class 0): Only one gamma photon reaches detector
3. **Random Coincidence** (Class 0): Two photons from different annihilation events detected within coincidence window

---

## Data Structure

### Input: Burst Tensor

**Shape**: `(T, 2, H, W)`
- `T`: Temporal frames (10 ns acquisition windows)
- `2`: Inner and outer detector surfaces
- `H, W`: Spatial dimensions of unrolled cylinder

Each burst represents the complete optical shower from a gamma interaction:
- Initial flash at interaction point
- Temporal decay based on scintillator properties
- Captured by silicon photomultipliers on both detector surfaces

### Ground Truth

Each burst has associated labels:
- **Cartesian coordinates** (in data files): `(x1, y1, z1, x2, y2, z2)` in mm
- **Polar coordinates** (for training): `(r1, θ1, r2, θ2)`
- **Event type**: `1` for true coincidence, `0` for singles/random
- Missing interactions: coordinates set to `(0, 0, 0)`

### Data Format

Preprocessed data stored as `.pt` files containing:
```python
{
    "burst": torch.FloatTensor,    # [T, 2, H, W]
    "target": torch.FloatTensor,   # [6] -> (x1,y1,z1, x2,y2,z2)
    "event_type": torch.LongTensor # scalar: 0 or 1
}
```

---

## Pipeline Overview

### 1. Data Loading (`src/data/dataset.py`)

**`PetBurstDataset`**:
- Loads `.pt` files from specified directories
- Supports recursive search and glob patterns
- Optional data augmentation via transforms
- Returns bursts with targets and event types

**`expand_train_input()`**:
- Handles comma-separated directory lists
- Auto-discovers experiment subdirectories
- Expands training data paths

### 2. Coordinate Conversion (`src/model/utils.py`)

**`convert_targets_to_polar()`**:
- Converts Cartesian `(x, y, z)` → Polar `(r, θ)`
- Applied to both interaction points
- Preserves cylindrical geometry

```python
target_coord = convert_targets_to_polar(target)  # (B,6) → (B,4)
target = torch.cat([target_coord, event_type.unsqueeze(-1)], dim=-1)  # (B,5)
```

### 3. Model Forward Pass (`src/model/`)

Models process bursts through:

1. **Frame Encoding**: Extract spatial features from each temporal frame
2. **Temporal Aggregation**: Combine features across time (e.g., GRU)
3. **Dual-Stream Processing**: Separate pathways for inner/outer detectors
4. **Prediction Heads**:
   - Polar coordinate regression for each detector
   - Event type classification from combined features

### 4. Loss Computation (`src/training/losses.py`)

**Available Loss Functions**:

- **`MixedRegressionClassificationLoss`** (current):
  - `SmoothL1Loss` for polar coordinates
  - `BCELoss` for event type classification
  - Weighted combination with per-class metrics

- **`PhysicsInformedEuclideanLoss`**:
  - Euclidean distance in mm
  - Arc-length error (radius × |Δθ|)
  - Radial and axial errors
  - Boundary penalties for geometric constraints

### 5. Training Loop (`main.py`)

**Key Features**:
- Distributed Data Parallel (DDP) support
- Mixed precision training (AMP)
- OneCycleLR scheduler
- TensorBoard logging
- Checkpointing with validation monitoring
- Early stopping

**Training Step**:
```python
1. Load batch: burst (B,T,2,H,W), target (B,6), event_type (B)
2. Convert target to polar: (B,6) → (B,4) → concat event_type → (B,5)
3. Forward pass: model(burst) → predictions (B,5)
4. Compute loss: criterion(predictions, target)
5. Backward and optimize
6. Log metrics: regression loss, classification loss, accuracy
```

### 6. Validation (`validate()`)

- Evaluates on validation set without gradients
- Aggregates metrics across distributed processes
- Logs to TensorBoard
- Returns validation loss for checkpointing

---

## Model Architectures

### SwinUNetV2Regressor (Recommended)

**Architecture**: Twin-stream Swin Transformer V2 with U-Net decoder

**Key Features**:
- **Swin Transformer V2 backbone**: Hierarchical vision transformer with shifted windows
- **Circular padding**: Handles cylindrical geometry (left/right edges continuous)
- **Temporal GRU**: Aggregates frame features across time
- **Dual regression heads**: Separate MLPs for inner/outer polar coordinates
- **Event classifier**: Combined feature classification head

**Forward Flow**:
```
Input: (B, T, 2, H, W)
  ↓
Split into inner/outer streams
  ↓
For each stream:
  - Per-frame encoding via Swin-UNet (with circular padding)
  - Temporal aggregation via GRU
  - Dropout regularization
  ↓
Inner stream → MLP → (r1, θ1)
Outer stream → MLP → (r2, θ2)
Combined features → MLP → event_type
  ↓
Output: (B, 5) = [r1, θ1, r2, θ2, event_type]
```

**Why SwinUNetV2 is Best**:
1. **Multi-scale spatial features**: Swin's hierarchical windows capture both local and global patterns
2. **Temporal modeling**: Explicit GRU handles decay dynamics
3. **Pretrained backbone**: Transfer learning from ImageNet
4. **Cylindrical awareness**: Circular padding for wraparound geometry
5. **Regularization**: Feature and head dropout for robust generalization

### Other Architectures

- **`PetNetImproved3D`**: 3D CNN with batch/instance/group normalization
- **`PetNetImproved3DSlim`**: Lighter 3D CNN variant
- **`PetPointLocaliser`**: Transfer learning based architecture

---

## Training

### Configuration

Set environment variables in `.env` file:

```bash
# Data paths
TRAIN=/path/to/train/data
VAL=/path/to/val/data

# Model
MODEL=swinunetv2
LOSS=mixed

# Hyperparameters
BS=16
EPOCHS=100
LR=1e-4
MAX_LR=1e-3
WD=1e-4
CLIP=1.0

# Training settings
WORKERS=8
NO_AMP=0  # Use mixed precision
SEED=42

# Logging
LOGDIR=runs/experiment_name
```

### Running Training

```bash
# Single GPU
python main.py

# Multi-GPU (DDP)
torchrun --nproc_per_node=4 main.py

# Custom .env file
python main.py --env /path/to/custom.env
```

### Monitoring

```bash
tensorboard --logdir runs/
```

**Logged Metrics**:
- Training: loss, lr, regression_loss, classification_loss, accuracy (overall, true, singles)
- Validation: same as training, computed over full validation set

### Checkpoints

Best model saved to:
```
{LOGDIR}/checkpoints/best_epoch{epoch}_loss{val_loss}.pt
```

Contains:
- Model state dict
- Optimizer state
- Validation loss
- Training arguments

---

## Usage

### Model Factory

```python
from src.model.factory import create_model, create_loss

# Create model
model = create_model("swinunetv2", args)

# Create loss
criterion = create_loss(args)
```

### Inference

```python
import torch
from src.data.dataset import PetBurstDataset
from src.model.factory import create_model

# Load model
checkpoint = torch.load("path/to/checkpoint.pt")
model = create_model("swinunetv2", args)
model.load_state_dict(checkpoint["model_state"])
model.eval()

# Load data
dataset = PetBurstDataset("path/to/test/data")
burst = dataset[0]["burst"].unsqueeze(0)  # Add batch dimension

# Predict
with torch.no_grad():
    prediction = model(burst)  # (1, 5)
    r1, theta1, r2, theta2, event_type = prediction[0]
```

### Coordinate Conversion

```python
from src.model.utils import convert_targets_to_polar, polar_to_cartesian

# Cartesian → Polar
polar = convert_targets_to_polar(cartesian_coords)  # (B,6) → (B,4)

# Polar → Cartesian (if needed for visualization)
x, y = r * torch.cos(theta), r * torch.sin(theta)
```

---

## Project Structure

```
pet-point-localiser/
├── src/
│   ├── data/
│   │   ├── dataset.py           # PetBurstDataset
│   │   └── csv_to_pt.py         # Data preprocessing
│   ├── model/
│   │   ├── factory.py           # Model/loss creation
│   │   ├── swin_unet_v2.py      # SwinUNetV2Regressor
│   │   ├── heads.py             # MLP heads (polar, xyz)
│   │   ├── pet_net_improved_3d.py
│   │   └── pet_point_localiser.py
│   ├── training/
│   │   └── losses.py            # Loss functions
│   └── utils/
│       ├── config.py            # Training configuration
│       └── utils.py             # Helper functions
├── main.py                       # Training script
├── requirements.txt
└── README.md
```

---

## Key Implementation Details

### Circular Padding

The cylindrical detector is unrolled into 2D frames, making left/right edges spatially continuous. `SwinUNetV2Regressor` uses circular padding in the horizontal (width) dimension:

```python
# In ConvBNAct forward():
x = F.pad(x, (pad_w, pad_w, 0, 0), mode='circular')  # Horizontal
x = F.pad(x, (0, 0, pad_h, pad_h), mode='constant')  # Vertical
```

This ensures the model learns continuous patterns across the θ=0°/360° boundary.

### Missing Endpoints

When only one gamma photon is detected:
- Second endpoint coordinates: `(x2, y2, z2) = (0, 0, 0)`
- Model still predicts `(r2, θ2)` but these are ignored in loss
- Event type prediction identifies singles vs. coincidence

### Distributed Training

Supports PyTorch DDP for multi-GPU training:
- `DistributedSampler` ensures unique samples per GPU
- Gradient synchronisation across processes
- Validation metrics aggregated via `all_reduce`

---

## Citation

```
PhD Thesis: "Deep learning techniques for event localisation in a
continuous shell monolithic PET scintillator"
```

