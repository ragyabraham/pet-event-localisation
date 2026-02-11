# PET Event Localisation

Deep learning techniques for event localisation in a continuous shell monolithic PET scintillator. This project forms part of a PhD thesis focused on physics-informed CNNs for high-resolution PET imaging.

## Project Overview

The goal of this project is to accurately localise the first point of interaction (POI) of gamma photons within a continuous monolithic PET detector. By leveraging deep learning models (e.g., Transformers, 3D CNNs), we predict interaction coordinates from the optical photon shower patterns captured by silicon photomultipliers (SiPMs).

### Key Features
- **Physics-Informed Deep Learning**: Custom loss functions and architectures tailored to PET physics.
- **Cylindrical Geometry Support**: Models utilize circular padding to handle the wraparound nature of the detector.
- **End-to-End Pipeline**: From GATE simulation to data rasterisation and neural network training.
- **State-of-the-Art Architectures**: Includes SwinUNetV2, Swin-Transformer, and 3D CNN variants.

---

## Project Structure

The repository is divided into three main modules:

### 1. [gate-simulation](./gate-simulation/)
Contains the **OpenGate** (Geant4-based) simulation configurations. This module generates the raw physics data, simulating gamma interactions, optical photon transport, and detection in a nano-composite scintillator shell.
- **Output**: `.root` files (typically converted to `.dat` or `.csv` for further processing).

### 2. [rasterisation](./rasterisation/)
Responsible for data preprocessing and feature engineering. It converts raw simulation hits into a format suitable for machine learning.
- **Process**:
  - Bins SiPM hits into temporal frames (bursts).
  - Generates 2D histograms (images) of optical photon distributions.
  - Matches optical bursts with ground truth interaction points from the simulation.
- **Output**: `.pt` (PyTorch) files containing burst tensors and labels.

### 3. [prism](./prism/)
The core machine learning library for training and evaluating localisation models.
- **Models**: SwinUNetV2, PetNetPhysics3D, PointNet, etc.
- **Training**: Supports Distributed Data Parallel (DDP), Mixed Precision (AMP), and OneCycleLR scheduling.
- **Inference**: Scripts for model evaluation, error analysis, and visualization.

---

## Getting Started

### Prerequisites
- Python 3.10+
- PyTorch 2.0+
- CUDA-enabled GPU (recommended for training)

### Workflow

1.  **Simulation**: Run the GATE simulation to generate raw data.
2.  **Preprocessing**: Use the `rasterisation` module to convert raw hits into training-ready tensors.
    ```bash
    cd rasterisation
    python src/main.py --input /path/to/simulation/data --output /path/to/processed/data
    ```
3.  **Training**: Configure your environment in `prism` and start training.
    ```bash
    cd prism
    python main.py --env .env
    ```
4.  **Inference**: Evaluate the trained model on test data.
    ```bash
    cd prism
    python src/inference/run_inference.py --checkpoint /path/to/model.pt --data /path/to/test/data
    ```

---

## Data Pipeline

| Stage | Input | Tool | Output |
| :--- | :--- | :--- | :--- |
| **Simulation** | GATE Config | `OpenGate` | `hits.root` |
| **Extraction** | `hits.root` | `read_root` | `hits.dat` |
| **Rasterisation** | `hits.dat` | `rasterisation/src/main.py` | `.pt` Bursts |
| **Training** | `.pt` Bursts | `prism/main.py` | Trained Model |

---

## Documentation

For detailed information on each module, please refer to their respective READMEs:
- [Rasterisation README](./rasterisation/README.md)
- [PRISM (Machine Learning) README](./prism/README.md)

---

## Citation
If you use this code in your research, please cite the associated PhD Thesis:
> *Physics informed CNN for event localisation in monolithic PET*
