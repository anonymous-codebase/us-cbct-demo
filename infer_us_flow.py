#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
infer_us_flow.py

Bidirectional ultrasound deformation inference.

Functionality
-------------
- Read ultrasound pair:
    I0.png (reference)
    I1.png (deformed)
- Build StuCorUNet 5-channel input:
    X = [I0, I1, (I1-I0), |∇I0|, |∇I1|]
- Predict bidirectional flow:
    F01, F10 = model(X)
- Warp images:
    I0' = warp(I1, F01)  # I1 -> I0 grid
    I1' = warp(I0, F10)  # I0 -> I1 grid
- Save:
    I0_pred.png, I1_pred.png

"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from infer_utils import (
    autocast_context,
    build_stucor_input_5ch,
    ensure_paths_exist,
    load_student_model,
    log_info,
    make_base_grid,
    read_gray_float01,
    save_gray_float01_png,
    setup_inference_device,
    warp_tensor,
)


# =============================================================================
# Hard-coded configuration (public demo style)
# =============================================================================
@dataclass(frozen=True)
class Config:
    """Configuration for US deformation inference."""

    in_dir: Path = Path("examples/us/1")
    ckpt_path: Path = Path("models/base.pth")

    use_amp: bool = True
    align_corners: bool = True

    input_i0_name: str = "I0.png"
    input_i1_name: str = "I1.png"

    output_i0_pred_name: str = "I0_pred.png"
    output_i1_pred_name: str = "I1_pred.png"


CFG = Config()


# =============================================================================
# Inference
# =============================================================================
@torch.inference_mode()
def run_bidirectional_us_warp(
    *,
    model: torch.nn.Module,
    device: torch.device,
    i0_01,
    i1_01,
    use_amp: bool,
    align_corners: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Predict bidirectional flow and warp the input ultrasound pair.

    Parameters
    ----------
    model : torch.nn.Module
        Loaded StuCorUNet model.
    device : torch.device
        Inference device.
    i0_01 : np.ndarray
        Reference image, shape (H, W), float32 [0,1].
    i1_01 : np.ndarray
        Deformed image, shape (H, W), float32 [0,1].
    use_amp : bool
        Whether to enable autocast on CUDA.
    align_corners : bool
        align_corners setting for grid_sample.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        (I0_prime, I1_prime) as tensors with shape (1,1,H,W).
    """
    if i0_01.shape != i1_01.shape:
        raise ValueError(f"Input shape mismatch: I0={i0_01.shape}, I1={i1_01.shape}")

    height, width = i0_01.shape
    base_grid = make_base_grid(height, width, device)

    x_np = build_stucor_input_5ch(i0_01, i1_01)  # (5, H, W)
    x = torch.from_numpy(x_np).unsqueeze(0).to(device, non_blocking=True)  # (1, 5, H, W)

    i0_t = torch.from_numpy(i0_01).unsqueeze(0).unsqueeze(0).to(device, non_blocking=True)
    i1_t = torch.from_numpy(i1_01).unsqueeze(0).unsqueeze(0).to(device, non_blocking=True)

    with autocast_context(device=device, enabled=use_amp, dtype=torch.float16):
        f01, f10 = model(x)
        i0_prime = warp_tensor(i1_t, f01, base_grid, align_corners=align_corners)  # I1 -> I0 grid
        i1_prime = warp_tensor(i0_t, f10, base_grid, align_corners=align_corners)  # I0 -> I1 grid

    return i0_prime, i1_prime


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    input_i0_path = CFG.in_dir / CFG.input_i0_name
    input_i1_path = CFG.in_dir / CFG.input_i1_name
    output_i0_pred_path = CFG.in_dir / CFG.output_i0_pred_name
    output_i1_pred_path = CFG.in_dir / CFG.output_i1_pred_name

    ensure_paths_exist([input_i0_path, input_i1_path, CFG.ckpt_path])

    i0_01, _ = read_gray_float01(input_i0_path)
    i1_01, _ = read_gray_float01(input_i1_path)

    if i0_01.shape != i1_01.shape:
        raise RuntimeError(f"Shape mismatch: I0={i0_01.shape}, I1={i1_01.shape}")

    device = setup_inference_device()
    model = load_student_model(CFG.ckpt_path, device, in_ch=5, strict=False)

    i0_prime, i1_prime = run_bidirectional_us_warp(
        model=model,
        device=device,
        i0_01=i0_01,
        i1_01=i1_01,
        use_amp=CFG.use_amp,
        align_corners=CFG.align_corners,
    )

    save_gray_float01_png(output_i0_pred_path, i0_prime)
    save_gray_float01_png(output_i1_pred_path, i1_prime)

    log_info(f"Saved: {output_i0_pred_path}")
    log_info(f"Saved: {output_i1_pred_path}")


if __name__ == "__main__":
    main()