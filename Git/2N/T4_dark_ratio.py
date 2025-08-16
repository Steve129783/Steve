#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import torch
import csv
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

# -------- Configuration --------
group_n    = '1_2_3_4'
CACHE_FILE = rf'E:\Project_SNV\1N\1_Pack\{group_n}\data_cache.pt'
CSV_OUT    = rf'E:\Project_SNV\2N\c1_cache\{group_n}\defect_ratios_o.csv'
BATCH_SIZE = 16

SHOW_PATH = None  # If set to a file path, only visualize this image

# -------- Dataset Definition --------
class CachedImageDataset(Dataset):
    """
    Dataset wrapper for cached image patches stored in a .pt file.
    Each item returns (float32 grayscale image, absolute path).
    """
    def __init__(self, cache_path):
        cache = torch.load(cache_path, map_location='cpu')
        imgs = cache['images']  # Tensor [N,1,H,W], float32
        paths = cache.get('paths', [f"idx_{i}" for i in range(len(imgs))])
        # Normalize paths to absolute paths
        self.paths = [os.path.abspath(p) for p in paths]
        self.images = imgs

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx].squeeze().numpy().astype(np.float32)
        # Normalize to [0,1] if not already
        mi, ma = img.min(), img.max()
        if ma > 1.0 or mi < 0.0:
            img = (img - mi) / (ma - mi)
        path = self.paths[idx]
        return img, path

# -------- Float-based Otsu Thresholding --------
def otsu_from_floats(img, nbins=256):
    """
    Compute Otsu's threshold for a float32 grayscale image (range [0,1]).
    Returns the threshold value.
    """
    # 1) Histogram
    hist, bin_edges = np.histogram(img.ravel(), bins=nbins)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # 2) Class weights
    w0 = np.cumsum(hist)           # weight of class 0
    total = hist.sum()
    w1 = total - w0                # weight of class 1

    # 3) Cumulative weighted sums
    cum_sum = np.cumsum(hist * centers)
    total_sum = cum_sum[-1]

    # 4) Class means
    m0 = cum_sum / np.where(w0 == 0, 1, w0)
    m1 = (total_sum - cum_sum) / np.where(w1 == 0, 1, w1)

    # 5) Maximize between-class variance
    var_between = w0[:-1] * w1[:-1] * (m0[:-1] - m1[:-1])**2
    idx = np.nanargmax(var_between)
    return centers[idx]

def defect_mask_otsu_float(gray):
    """
    Given a float32 grayscale image [0,1], apply Otsu thresholding.
    Returns:
        threshold, binary image (float32), binary mask (uint8)
    """
    ret = otsu_from_floats(gray, nbins=256)
    mask = (gray <= ret)  # boolean mask of "defect" regions
    bw = mask.astype(np.float32)
    mask_u8 = mask.astype(np.uint8)
    return ret, bw, mask_u8

# -------- Visualization --------
def visualize_single(gray, bw, mask, path):
    """
    Show original grayscale, binary, mask, and overlay for one image.
    """
    plt.figure(figsize=(12,3))

    # Original
    ax = plt.subplot(1,4,1)
    ax.imshow(gray, cmap='gray', vmin=0, vmax=1)
    ax.set_title('Gray (float32)')
    ax.axis('off')

    # Binary Otsu
    ax = plt.subplot(1,4,2)
    ax.imshow(bw, cmap='gray', vmin=0, vmax=1)
    ax.set_title('Binary (Otsu float)')
    ax.axis('off')

    # Mask
    ax = plt.subplot(1,4,3)
    ax.imshow(mask, cmap='gray', vmin=0, vmax=1)
    ax.set_title('Mask (0/1)')
    ax.axis('off')

    # Overlay
    ax = plt.subplot(1,4,4)
    ax.imshow(gray, cmap='gray', vmin=0, vmax=1)
    ax.imshow(np.ma.masked_where(mask==0, mask),
              cmap='Reds', alpha=0.3, vmin=0, vmax=1)
    ax.set_title('Overlay')
    ax.axis('off')

    plt.suptitle(os.path.basename(path))
    plt.tight_layout(rect=[0,0.03,1,0.95])
    plt.show()

# -------- Main Routine --------
def main():
    ds = CachedImageDataset(CACHE_FILE)

    # Single image visualization mode
    if SHOW_PATH:
        show_abs = os.path.abspath(SHOW_PATH)
        if show_abs not in ds.paths:
            print(f"Path '{show_abs}' not found in dataset.")
            return
        idx = ds.paths.index(show_abs)
        gray, _ = ds[idx]
        ret, bw, mask = defect_mask_otsu_float(gray)
        visualize_single(gray, bw, mask, show_abs)
        return

    # Batch processing: compute defect ratios for all images
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda x:x)
    os.makedirs(os.path.dirname(CSV_OUT), exist_ok=True)
    with open(CSV_OUT, 'w', newline='') as fp:
        writer = csv.writer(fp)
        writer.writerow(['path','defect_ratio'])
        for batch in loader:
            for gray, path in batch:
                ret, bw, mask = defect_mask_otsu_float(gray)
                ratio = mask.sum() / mask.size
                writer.writerow([path, f"{ratio:.6f}"])
    print(f"CSV written to: {CSV_OUT}")

if __name__ == '__main__':
    main()
