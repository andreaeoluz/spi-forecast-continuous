"""spatial.py - Spatial helpers shared across the continuous SPI pipeline.

Binary-mask-specific post-processing (small-object/hole removal on
thresholded drought masks) has been removed - the model output is a
continuous field, so there is nothing to morphologically clean up.
"""

import numpy as np
from typing import Tuple

# Re-exported from data.preprocessing so there is a single source of truth
# for these functions instead of two independent implementations.
from data.preprocessing import (  # noqa: F401
    downsample_block_mean,
    build_valid_mask,
    apply_domain_mask,
)


def get_valid_pixel_indices(
    mask: np.ndarray,
    return_flat: bool = True
) -> np.ndarray:
    """
    Return the indices of valid pixels in a boolean mask.

    Args:
        mask: Boolean mask (H, W).
        return_flat: If True, return flat indices; otherwise 2D indices.

    Returns:
        Indices of valid pixels.
    """
    if return_flat:
        return np.where(mask.flatten())[0]
    else:
        return np.where(mask)


def create_train_val_mask(
    mask: np.ndarray,
    validation_split: float = 0.2,
    seed: int = 42,
    spatial_blocks: bool = False,
    block_size: int = 8
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create train/validation masks from a domain mask.

    Args:
        mask: Domain mask (H, W).
        validation_split: Fraction of pixels reserved for validation.
        seed: Random seed for reproducibility.
        spatial_blocks: If True, split by spatial blocks instead of pixels.
        block_size: Block size used when spatial_blocks is True.

    Returns:
        (train_mask, val_mask)
    """
    np.random.seed(seed)

    if spatial_blocks:
        H, W = mask.shape
        train_mask = np.zeros_like(mask, dtype=bool)
        val_mask = np.zeros_like(mask, dtype=bool)

        blocks_h = H // block_size
        blocks_w = W // block_size

        block_indices = np.arange(blocks_h * blocks_w)
        np.random.shuffle(block_indices)

        n_val_blocks = int(blocks_h * blocks_w * validation_split)
        val_blocks = set(block_indices[:n_val_blocks])

        for bi in range(blocks_h):
            for bj in range(blocks_w):
                h_start = bi * block_size
                h_end = min(h_start + block_size, H)
                w_start = bj * block_size
                w_end = min(w_start + block_size, W)

                block_idx = bi * blocks_w + bj
                if block_idx in val_blocks:
                    val_mask[h_start:h_end, w_start:w_end] = True
                else:
                    train_mask[h_start:h_end, w_start:w_end] = True

        train_mask = train_mask & mask
        val_mask = val_mask & mask

    else:
        valid_indices = get_valid_pixel_indices(mask, return_flat=True)

        n_valid = len(valid_indices)
        n_val = int(n_valid * validation_split)

        shuffled = np.random.permutation(valid_indices)
        val_indices = shuffled[:n_val]
        train_indices = shuffled[n_val:]

        train_mask = np.zeros_like(mask, dtype=bool)
        val_mask = np.zeros_like(mask, dtype=bool)

        train_mask.flat[train_indices] = True
        val_mask.flat[val_indices] = True

    return train_mask, val_mask
