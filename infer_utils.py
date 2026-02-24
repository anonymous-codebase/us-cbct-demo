#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
infer_utils.py

Shared utilities for public inference scripts.

This module centralizes:
- image I/O and grayscale conversion
- normalization to float32 [0, 1]
- PNG saving helpers
- shape padding utilities
- 5-channel StuCorUNet input construction
- flow-based warping with torch.grid_sample
- model checkpoint loading
- device/runtime setup

The goal is to keep inference scripts concise, consistent, and easy to audit.
"""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from stucorunet import StudentCorrUNetGN


# =============================================================================
# Logging helpers
# =============================================================================
def log_info(message: str) -> None:
    """Print an info-level message with a consistent prefix."""
    print(f"[INFO] {message}")


def log_warn(message: str) -> None:
    """Print a warning-level message with a consistent prefix."""
    print(f"[WARN] {message}")


# =============================================================================
# Image I/O and normalization
# =============================================================================
def read_image_any(path: Path) -> np.ndarray:
    """
    Read an image using OpenCV without altering the bit depth.

    Parameters
    ----------
    path : Path
        Image path.

    Returns
    -------
    np.ndarray
        Image array as a contiguous NumPy array.

    Raises
    ------
    FileNotFoundError
        If the file cannot be read by OpenCV.
    """
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return np.ascontiguousarray(image)


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert an image to a single-channel grayscale array.

    Supported inputs:
    - (H, W)
    - (H, W, 1)
    - (H, W, 3) (interpreted as BGR from OpenCV)
    - (H, W, 4) (BGRA -> BGR -> gray)

    Parameters
    ----------
    image : np.ndarray
        Input image.

    Returns
    -------
    np.ndarray
        Grayscale image with shape (H, W).
    """
    if image.ndim == 2:
        return image

    if image.ndim == 3 and image.shape[2] == 1:
        return image[..., 0]

    if image.ndim == 3 and image.shape[2] == 4:
        image = image[..., :3]

    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    raise ValueError(f"Unsupported image shape for grayscale conversion: {image.shape}")


def _integer_dtype_max(dtype: np.dtype) -> float:
    """Return the maximum representable value for an integer dtype."""
    if np.issubdtype(dtype, np.integer):
        return float(np.iinfo(dtype).max)
    return 1.0


def image_to_float01(gray: np.ndarray) -> tuple[np.ndarray, np.dtype]:
    """
    Convert a grayscale image to float32 in [0, 1].

    Integer images are normalized using dtype max (e.g., uint8->255, uint16->65535).
    Floating-point images are cast to float32 and clipped to [0, 1].

    Parameters
    ----------
    gray : np.ndarray
        Single-channel image.

    Returns
    -------
    tuple[np.ndarray, np.dtype]
        (normalized_image_float32, original_dtype)
    """
    if gray.ndim != 2:
        raise ValueError(f"image_to_float01 expects a 2D grayscale image, got shape={gray.shape}")

    original_dtype = gray.dtype

    if gray.dtype == np.uint8:
        x = gray.astype(np.float32) / 255.0
    elif gray.dtype == np.uint16:
        x = gray.astype(np.float32) / 65535.0
    elif np.issubdtype(gray.dtype, np.integer):
        denom = _integer_dtype_max(gray.dtype)
        x = gray.astype(np.float32) / (denom if denom > 0 else 1.0)
    else:
        x = gray.astype(np.float32)

    x = np.clip(x, 0.0, 1.0)
    return np.ascontiguousarray(x, dtype=np.float32), original_dtype


def read_gray_float01(path: Path) -> tuple[np.ndarray, np.dtype]:
    """
    Read an image, convert to grayscale, and normalize to float32 [0, 1].

    Parameters
    ----------
    path : Path
        Input image path.

    Returns
    -------
    tuple[np.ndarray, np.dtype]
        (float32 grayscale image in [0,1], original dtype before normalization)
    """
    raw = read_image_any(path)
    gray = to_grayscale(raw)
    return image_to_float01(gray)


def save_gray_float01_png(path: Path, tensor_or_array: torch.Tensor | np.ndarray) -> None:
    """
    Save a grayscale image in [0, 1] as uint8 PNG.

    Parameters
    ----------
    path : Path
        Output PNG path.
    tensor_or_array : torch.Tensor | np.ndarray
        Image data. Accepts shape (H, W), (1, H, W), or (1, 1, H, W).
    """
    if isinstance(tensor_or_array, torch.Tensor):
        x = tensor_or_array.detach().squeeze().clamp(0.0, 1.0).to(torch.float32).cpu().numpy()
    else:
        x = np.asarray(tensor_or_array, dtype=np.float32).squeeze()
        x = np.clip(x, 0.0, 1.0)

    if x.ndim != 2:
        raise ValueError(f"save_gray_float01_png expects a 2D grayscale image after squeeze, got {x.shape}")

    path.parent.mkdir(parents=True, exist_ok=True)
    u8 = (x * 255.0 + 0.5).astype(np.uint8)
    ok = cv2.imwrite(str(path), u8)
    if not ok:
        raise IOError(f"Failed to write PNG: {path}")


def save_float01_like_dtype_png(path: Path, x01: np.ndarray, reference_dtype: np.dtype) -> None:
    """
    Save float32 [0,1] image as uint8 or uint16 PNG depending on reference dtype.

    Parameters
    ----------
    path : Path
        Output image path.
    x01 : np.ndarray
        Grayscale float32 image in [0,1].
    reference_dtype : np.dtype
        If uint16 -> save uint16 PNG; otherwise save uint8 PNG.
    """
    if x01.ndim != 2:
        raise ValueError(f"save_float01_like_dtype_png expects shape (H, W), got {x01.shape}")

    x01 = np.clip(np.asarray(x01, dtype=np.float32), 0.0, 1.0)
    path.parent.mkdir(parents=True, exist_ok=True)

    if reference_dtype == np.uint16:
        out = (x01 * 65535.0 + 0.5).astype(np.uint16)
    else:
        out = (x01 * 255.0 + 0.5).astype(np.uint8)

    ok = cv2.imwrite(str(path), out)
    if not ok:
        raise IOError(f"Failed to write image: {path}")


# =============================================================================
# Shape utilities
# =============================================================================
def pad_to_hw(arr: np.ndarray, height: int, width: int, value: float | int = 0) -> np.ndarray:
    """
    Pad an array to (height, width) at bottom/right, keeping top-left aligned.

    Supports:
    - (H, W)
    - (H, W, C)

    Parameters
    ----------
    arr : np.ndarray
        Input array.
    height : int
        Target height.
    width : int
        Target width.
    value : float | int
        Padding value.

    Returns
    -------
    np.ndarray
        Padded array.
    """
    if arr.ndim == 2:
        h, w = arr.shape
        out = np.full((height, width), value, dtype=arr.dtype)
        out[:h, :w] = arr
        return out

    if arr.ndim == 3:
        h, w, c = arr.shape
        out = np.full((height, width, c), value, dtype=arr.dtype)
        out[:h, :w, :] = arr
        return out

    raise ValueError(f"pad_to_hw only supports 2D/3D arrays, got shape={arr.shape}")


def pad_images_to_common_hw(
    images: Sequence[np.ndarray],
    pad_value: float | int = 0,
) -> tuple[list[np.ndarray], int, int]:
    """
    Pad a list of 2D grayscale images to a common shape (max H, max W).

    Parameters
    ----------
    images : Sequence[np.ndarray]
        List/tuple of images with shape (H, W).
    pad_value : float | int
        Padding value.

    Returns
    -------
    tuple[list[np.ndarray], int, int]
        (padded_images, target_height, target_width)
    """
    if len(images) == 0:
        raise ValueError("pad_images_to_common_hw received an empty image list.")

    for i, img in enumerate(images):
        if img.ndim != 2:
            raise ValueError(f"Expected 2D image at index {i}, got shape={img.shape}")

    target_h = max(img.shape[0] for img in images)
    target_w = max(img.shape[1] for img in images)
    padded = [pad_to_hw(img, target_h, target_w, value=pad_value) for img in images]
    return padded, target_h, target_w


# =============================================================================
# StuCorUNet input construction
# =============================================================================
def np_grad_mag(image01: np.ndarray) -> np.ndarray:
    """
    Compute simple forward-difference gradient magnitude for a float image in [0,1].

    Parameters
    ----------
    image01 : np.ndarray
        2D float32 image.

    Returns
    -------
    np.ndarray
        Gradient magnitude (float32), same shape as input.
    """
    if image01.ndim != 2:
        raise ValueError(f"np_grad_mag expects a 2D image, got shape={image01.shape}")

    image01 = np.ascontiguousarray(image01, dtype=np.float32)

    gy = np.zeros_like(image01, dtype=np.float32)
    gx = np.zeros_like(image01, dtype=np.float32)

    gy[1:, :] = image01[1:, :] - image01[:-1, :]
    gx[:, 1:] = image01[:, 1:] - image01[:, :-1]

    return np.sqrt(np.maximum(gx * gx + gy * gy, 0.0)).astype(np.float32)


def build_stucor_input_5ch(i0_01: np.ndarray, i1_01: np.ndarray) -> np.ndarray:
    """
    Build the 5-channel input used by StuCorUNet.

    Channel order (kept identical to the original scripts):
        [I0, I1, (I1 - I0), |∇I0|, |∇I1|]

    Parameters
    ----------
    i0_01 : np.ndarray
        Reference image, float32 in [0,1], shape (H, W).
    i1_01 : np.ndarray
        Deformed image, float32 in [0,1], shape (H, W).

    Returns
    -------
    np.ndarray
        Input tensor as NumPy array, shape (5, H, W), dtype float32.
    """
    if i0_01.shape != i1_01.shape:
        raise ValueError(f"Input shape mismatch: i0={i0_01.shape}, i1={i1_01.shape}")

    i0_01 = np.ascontiguousarray(i0_01, dtype=np.float32)
    i1_01 = np.ascontiguousarray(i1_01, dtype=np.float32)

    diff = (i1_01 - i0_01).astype(np.float32)
    g0 = np_grad_mag(i0_01)
    g1 = np_grad_mag(i1_01)

    x = np.stack([i0_01, i1_01, diff, g0, g1], axis=0)
    return np.ascontiguousarray(x, dtype=np.float32)


# =============================================================================
# Warping utilities
# =============================================================================
def make_base_grid(height: int, width: int, device: torch.device) -> torch.Tensor:
    """
    Create a normalized sampling grid for torch.grid_sample.

    Returns grid with shape (1, H, W, 2), where:
    - grid[..., 0] is x in [-1, 1]
    - grid[..., 1] is y in [-1, 1]
    """
    ys = torch.linspace(-1.0, 1.0, height, device=device)
    xs = torch.linspace(-1.0, 1.0, width, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx, yy], dim=-1).unsqueeze(0)


def warp_tensor(
    src: torch.Tensor,
    flow_px: torch.Tensor,
    base_grid: torch.Tensor,
    *,
    align_corners: bool = True,
    padding_mode: str = "zeros",
    mode: str = "bilinear",
) -> torch.Tensor:
    """
    Warp a source image using a dense flow field in pixel units.

    Parameters
    ----------
    src : torch.Tensor
        Source image tensor, shape (B, C, H, W).
    flow_px : torch.Tensor
        Flow tensor in pixel units, shape (B, 2, H, W), order (dx, dy).
    base_grid : torch.Tensor
        Base normalized grid, shape (1, H, W, 2) or (B, H, W, 2).
    align_corners : bool
        Passed to torch.grid_sample.
    padding_mode : str
        grid_sample padding mode.
    mode : str
        grid_sample interpolation mode.

    Returns
    -------
    torch.Tensor
        Warped tensor, shape (B, C, H, W).
    """
    if src.ndim != 4:
        raise ValueError(f"src must be (B,C,H,W), got {tuple(src.shape)}")
    if flow_px.ndim != 4 or flow_px.shape[1] != 2:
        raise ValueError(f"flow_px must be (B,2,H,W), got {tuple(flow_px.shape)}")

    b, _, h, w = flow_px.shape
    fx = flow_px[:, 0] * (2.0 / max(w - 1, 1))
    fy = flow_px[:, 1] * (2.0 / max(h - 1, 1))
    grid = base_grid + torch.stack([fx, fy], dim=-1)

    return F.grid_sample(
        src,
        grid,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners,
    )


# =============================================================================
# Runtime / model loading
# =============================================================================
def setup_inference_device() -> torch.device:
    """
    Select an inference device and apply lightweight CUDA runtime settings.

    Returns
    -------
    torch.device
        CUDA device if available, otherwise CPU.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        log_info("Using CUDA for inference.")
    else:
        log_warn("CUDA is not available. Falling back to CPU.")
    return device


def autocast_context(
    *,
    device: torch.device,
    enabled: bool,
    dtype: torch.dtype = torch.float16,
):
    """
    Return a context manager for autocast when supported and enabled.

    Notes
    -----
    - Autocast is applied only on CUDA.
    - On CPU or when disabled, a no-op context is returned.
    """
    if enabled and device.type == "cuda":
        return torch.amp.autocast("cuda", dtype=dtype)
    return nullcontext()


def _safe_torch_load_weights(path: str | Path):
    """
    Load a checkpoint with compatibility across PyTorch versions.

    Tries `weights_only=True` first (newer PyTorch), then falls back.
    """
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_student_model(
    ckpt_path: str | Path,
    device: torch.device,
    *,
    in_ch: int = 5,
    strict: bool = False,
) -> torch.nn.Module:
    """
    Load StudentCorrUNetGN checkpoint with common key-format handling.

    Supported checkpoint formats:
    - raw state_dict
    - {"model": state_dict}
    - {"state_dict": state_dict}
    - DataParallel prefixes ("module.")

    Parameters
    ----------
    ckpt_path : str | Path
        Checkpoint path.
    device : torch.device
        Target device.
    in_ch : int
        Number of input channels (default: 5).
    strict : bool
        strict flag passed to load_state_dict.

    Returns
    -------
    torch.nn.Module
        Model in eval mode on target device.
    """
    model = StudentCorrUNetGN(in_ch=in_ch).to(device).eval()

    ckpt = _safe_torch_load_weights(ckpt_path)
    state = ckpt.get("model", ckpt.get("state_dict", ckpt)) if isinstance(ckpt, dict) else ckpt

    if not isinstance(state, dict):
        raise RuntimeError(f"Unexpected checkpoint format: {ckpt_path}")

    if any(k.startswith("module.") for k in state.keys()):
        state = {k[7:]: v for k, v in state.items()}

    missing_keys, unexpected_keys = model.load_state_dict(state, strict=strict)

    if len(missing_keys) > 0:
        preview = ", ".join(missing_keys[:10])
        suffix = " ..." if len(missing_keys) > 10 else ""
        log_warn(f"Missing checkpoint keys ({len(missing_keys)}): {preview}{suffix}")

    if len(unexpected_keys) > 0:
        preview = ", ".join(unexpected_keys[:10])
        suffix = " ..." if len(unexpected_keys) > 10 else ""
        log_warn(f"Unexpected checkpoint keys ({len(unexpected_keys)}): {preview}{suffix}")

    return model


# =============================================================================
# Validation helpers
# =============================================================================
def ensure_paths_exist(paths: Iterable[Path]) -> None:
    """Raise FileNotFoundError if any path in the iterable does not exist."""
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"Missing file: {p}")