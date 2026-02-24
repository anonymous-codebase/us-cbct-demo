#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
infer_cbct_warp_from_us.py

Single-case inference for deformation-aware CBCT slice warping.

Pipeline
--------
1) Read:
   - US_I0.png (reference ultrasound)
   - US_I1.png (deformed ultrasound)
   - CBCT_I0.png (source CBCT slice)
2) Build StuCorUNet input from the ultrasound pair
3) Predict bidirectional flow (F01, F10)
4) Warp CBCT_I0 to target time using F10 (defined on I1 grid)
5) Save:
   - CBCT_I0_pred.png


Flow convention reminder
------------------------
F10 is the backward flow (t1 -> t0) defined on the t1 grid.
To synthesize CBCT at t1 from CBCT at t0, we warp CBCT_t0 with F10.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from infer_utils import (
    autocast_context,
    build_stucor_input_5ch,
    ensure_paths_exist,
    load_student_model,
    log_info,
    log_warn,
    make_base_grid,
    pad_images_to_common_hw,
    read_gray_float01,
    save_float01_like_dtype_png,
    setup_inference_device,
    warp_tensor,
)


# =============================================================================
# Hard-coded configuration (public demo style)
# =============================================================================
@dataclass(frozen=True)
class Config:
    """Configuration for single-case CBCT warping inference."""

    case_dir: Path = Path("examples/us_and_cbct/2")
    ckpt_path: Path = Path("models/probe_induced.pth")

    use_amp: bool = True
    align_corners: bool = True

    us_t0_name: str = "US_I0.png"
    us_t1_name: str = "US_I1.png"
    ct_t0_name: str = "CBCT_I0.png"

    out_ct_t1_pred_name: str = "CBCT_I0_pred.png"

    # If CT and US have different size before padding, resize CT to US_t0 size first.
    resize_ct_to_us: bool = True


CFG = Config()


# =============================================================================
# Inference
# =============================================================================
@torch.inference_mode()
def run_student_ct_warp(
    *,
    model: torch.nn.Module,
    device: torch.device,
    us0_01: np.ndarray,
    us1_01: np.ndarray,
    ct0_01: np.ndarray,
    use_amp: bool,
    align_corners: bool,
) -> np.ndarray:
    """
    Predict deformation from ultrasound pair and warp CBCT_t0 -> CBCT_t1.

    Parameters
    ----------
    model : torch.nn.Module
        Loaded StuCorUNet model.
    device : torch.device
        Inference device.
    us0_01 : np.ndarray
        Ultrasound at t0, shape (H, W), float32 [0,1].
    us1_01 : np.ndarray
        Ultrasound at t1, shape (H, W), float32 [0,1].
    ct0_01 : np.ndarray
        CBCT slice at t0, shape (H, W), float32 [0,1].
    use_amp : bool
        Whether to enable autocast on CUDA.
    align_corners : bool
        align_corners setting for grid_sample.

    Returns
    -------
    np.ndarray
        Predicted CBCT_t1 in float32 [0,1], shape (H, W).
    """
    if not (us0_01.shape == us1_01.shape == ct0_01.shape):
        raise ValueError(
            "All inputs must share the same shape before inference. "
            f"Got us0={us0_01.shape}, us1={us1_01.shape}, ct0={ct0_01.shape}"
        )

    height, width = us0_01.shape
    base_grid = make_base_grid(height, width, device)

    x_np = build_stucor_input_5ch(us0_01, us1_01)  # (5, H, W)
    x = torch.from_numpy(x_np).unsqueeze(0).to(device, non_blocking=True)  # (1, 5, H, W)
    ct0_t = torch.from_numpy(ct0_01).unsqueeze(0).unsqueeze(0).to(device, non_blocking=True)  # (1,1,H,W)

    with autocast_context(device=device, enabled=use_amp, dtype=torch.float16):
        f01, f10 = model(x)

        # Warp CBCT_t0 -> t1 using backward flow defined on t1 grid.
        ct1_pred = warp_tensor(ct0_t, f10, base_grid, align_corners=align_corners)

    out = ct1_pred.squeeze().clamp(0.0, 1.0).to(torch.float32).cpu().numpy()
    return np.ascontiguousarray(out, dtype=np.float32)


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    us_t0_path = CFG.case_dir / CFG.us_t0_name
    us_t1_path = CFG.case_dir / CFG.us_t1_name
    ct_t0_path = CFG.case_dir / CFG.ct_t0_name
    out_ct_t1_pred_path = CFG.case_dir / CFG.out_ct_t1_pred_name

    ensure_paths_exist([us_t0_path, us_t1_path, ct_t0_path, CFG.ckpt_path])

    # Read and normalize grayscale images
    us0_01, _ = read_gray_float01(us_t0_path)
    us1_01, _ = read_gray_float01(us_t1_path)
    ct0_01, ct_dtype = read_gray_float01(ct_t0_path)

    # Optionally resize CT to match US_t0 before padding
    if CFG.resize_ct_to_us and ct0_01.shape != us0_01.shape:
        target_h, target_w = us0_01.shape
        ct0_01 = cv2.resize(ct0_01, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        ct0_01 = np.ascontiguousarray(ct0_01, dtype=np.float32)
        log_warn(f"Resized CT to match US_t0 shape: {(target_h, target_w)}")

    # Pad all images to common shape (bottom/right padding, top-left aligned)
    (us0_01, us1_01, ct0_01), target_h, target_w = pad_images_to_common_hw(
        [us0_01, us1_01, ct0_01],
        pad_value=0.0,
    )

    # Safety check (should already be matched)
    if not (us0_01.shape == us1_01.shape == ct0_01.shape == (target_h, target_w)):
        raise RuntimeError(
            "Internal shape alignment failed: "
            f"us0={us0_01.shape}, us1={us1_01.shape}, ct0={ct0_01.shape}, target={(target_h, target_w)}"
        )

    device = setup_inference_device()
    model = load_student_model(CFG.ckpt_path, device, in_ch=5, strict=False)

    ct1_pred_01 = run_student_ct_warp(
        model=model,
        device=device,
        us0_01=us0_01,
        us1_01=us1_01,
        ct0_01=ct0_01,
        use_amp=CFG.use_amp,
        align_corners=CFG.align_corners,
    )

    save_float01_like_dtype_png(out_ct_t1_pred_path, ct1_pred_01, ct_dtype)

    log_info(f"Saved: {out_ct_t1_pred_path}")


if __name__ == "__main__":
    main()