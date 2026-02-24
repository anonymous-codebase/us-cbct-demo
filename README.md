# Deformation-Aware CBCT Updating

This repository provides a inference demo for **ultrasound-based deformation estimation** and **deformation-aware CBCT slice warping**.


## Overview

Included scripts:

- **`infer_us_flow.py`**: bidirectional ultrasound deformation inference
- **`infer_cbct_warp_from_us.py`**: CBCT slice warping using deformation estimated from ultrasound
- **`infer_utils.py`**: shared inference utilities (I/O, preprocessing, model loading, warping)
- **`stucorunet.py`**: network definition used for inference


## Checkpoint Selection

The `models/` directory contains three checkpoints corresponding to different motion regimes (base / probe-induced / external-induced), consistent with the paper's model setup and fine-tuning strategy.

### Ultrasound-only demos (`examples/us/*`)
Use the following checkpoints:

- `examples/us/1` → `models/base.pth`
- `examples/us/2` → `models/base.pth`
- `examples/us/3` → `models/probe_induced.pth`
- `examples/us/4` → `models/probe_induced.pth`
- `examples/us/5` → `models/base.pth`

### CBCT warping demos (`examples/us_and_cbct/*`)
Use:

- `models/probe_induced.pth`


## Download Checkpoints (GitHub Releases)

Pretrained checkpoints are provided via **GitHub Releases** as a single archive:

- `models.zip`

> Note: Release assets are **not downloaded automatically** when cloning the repository.

Please download `models.zip`, extract it at the repository root, and make sure it creates:

models/
  - `base.pth`
  - `probe_induced.pth`
  - `external_induced.pth`

## Requirements

- Python 3.10+
- PyTorch
- OpenCV (`opencv-python`)
- NumPy
