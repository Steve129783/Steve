#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import torch
import csv
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import cv2

# -------- Configuration --------
group_n    = '1_2_3_4'
CACHE_FILE = rf'E:\Project_SNV\1N\1_Pack\{group_n}\data_cache.pt'
CSV_OUT    = rf'E:\Project_SNV\2N\c2_cache\{group_n}\defect_ratios_m.csv'
BATCH_SIZE = 16
W = 5
ksize = 2*W + 1
C     = 10/255

# If you want to visualize a specific path, put it here; otherwise keep None
SHOW_PATH = rf"E:\Project_SNV\0S\6_patch\1\0_6_4.png"  # set to a full path or a dataset path string

# -------- Dataset Definition --------
class CachedImageDataset(Dataset):
    def __init__(self, cache_path):
        cache = torch.load(cache_path, map_location='cpu')
        self.images = cache['images']        # Tensor [N,1,H,W], float32
        self.paths  = cache.get(
            'paths',
            [f"idx_{i}" for i in range(len(self.images))]
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx].squeeze().numpy().astype(np.float32)
        # Normalize to [0,1] if it’s not already
        mi, ma = img.min(), img.max()
        if ma > 1.0 or mi < 0.0:
            img = (img - mi) / (ma - mi)
        path = str(self.paths[idx])
        return img, path

# -------- Float-based Gaussian Adaptive Thresholding --------
def defect_mask_float_gauss(gray_f32, ksize=13, C=0.05):
    """
    Apply Gaussian-weighted local thresholding on a float32 grayscale image:
      - gray_f32: float32 grayscale image (range [0,1])
      - ksize:    Gaussian kernel size (odd number)
      - C:        constant subtracted from the local mean (same unit as gray_f32)
    Returns:
      local_mean: Gaussian weighted local mean image
      mask:       binary mask (0 or 1), dtype=uint8
    """
    # 1) Compute local Gaussian weighted mean (float32 supported)
    local_mean = cv2.GaussianBlur(gray_f32, (ksize, ksize), (ksize-1)/6)
    # 2) Foreground: gray < mean - C
    mask = (gray_f32 < (local_mean - C)).astype(np.uint8)
    return local_mean, mask

# -------- Visualization in float domain (with local mean) --------
def visualize_with_mean(gray_f32, local_mean, mask, path):
    """
    Visualize with Matplotlib:
      - gray_f32:   float32 grayscale image, range [0,1]
      - local_mean: Gaussian local mean image
      - mask:       binary mask, 0/1
      - path:       image identifier string
    """
    plt.figure(figsize=(16,3))

    # Original
    ax = plt.subplot(1,4,1)
    ax.imshow(gray_f32, cmap='gray', vmin=0, vmax=1)
    ax.set_title('Gray (float32)', fontsize=8)
    ax.axis('off')

    # Local mean
    ax = plt.subplot(1,4,2)
    ax.imshow(local_mean, cmap='gray', vmin=0, vmax=1)
    ax.set_title('Local Mean', fontsize=8)
    ax.axis('off')

    # Binary mask
    ax = plt.subplot(1,4,3)
    ax.imshow(mask, cmap='gray', vmin=0, vmax=1)
    ax.set_title('Mask', fontsize=8)
    ax.axis('off')

    # Overlay
    ax = plt.subplot(1,4,4)
    ax.imshow(gray_f32, cmap='gray', vmin=0, vmax=1)
    ax.imshow(
        np.ma.masked_where(mask == 0, mask),
        cmap='Reds', alpha=0.3, vmin=0, vmax=1
    )
    ax.set_title('Overlay', fontsize=8)
    ax.axis('off')

    plt.suptitle(os.path.basename(path), fontsize=10)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# -------- Main Routine --------
def main():
    ds = CachedImageDataset(CACHE_FILE)

    # Single image visualization
    if SHOW_PATH is not None:
        if SHOW_PATH not in ds.paths:
            print(f"Path '{SHOW_PATH}' not found in dataset.")
            return
        idx = ds.paths.index(SHOW_PATH)
        gray_f32, _ = ds[idx]

        # Compute local mean and binary mask
        local_mean, mask = defect_mask_float_gauss(gray_f32, ksize=ksize, C=C)

        # Visualize with local mean
        visualize_with_mean(gray_f32, local_mean, mask, SHOW_PATH)
        return

    # Batch processing and CSV export
    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda x: x
    )
    os.makedirs(os.path.dirname(CSV_OUT), exist_ok=True)
    with open(CSV_OUT, 'w', newline='') as fp:
        writer = csv.writer(fp)
        writer.writerow(['path', 'defect_ratio'])
        for batch in loader:
            for gray_f32, path in batch:
                _, mask = defect_mask_float_gauss(gray_f32, ksize=ksize, C=C)
                ratio = mask.sum() / mask.size
                writer.writerow([path, f"{ratio:.6f}"])
    print(f"CSV written to: {CSV_OUT}")

if __name__ == '__main__':
    main()
