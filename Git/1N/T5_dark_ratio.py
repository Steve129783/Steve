#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import torch
import csv
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

# -------- Configuration --------
FG_RATIO_MIN = 0
FG_RATIO_MAX = 1
SHOW_PATH    = rf"E:\Project_SNV\0S\6_patch\1\0_6_4.png" # Specify a .png path to visualize, or None
group_n      = '1_2_3_4'
CACHE_FILE   = rf'E:\Project_SNV\1N\1_Pack\{group_n}\data_cache.pt'
CSV_OUT      = rf'E:\Project_SNV\1N\c1_cache\{group_n}\defect_ratios_o.csv'
BATCH_SIZE   = 16

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
        # Normalize to [0,1] if not already in this range
        mi, ma = img.min(), img.max()
        if ma > 1.0 or mi < 0.0:
            img = (img - mi) / (ma - mi)
        path = str(self.paths[idx])
        return img, path

# -------- Manual Otsu Implementation --------
def otsu_from_floats(img, nbins=256):
    hist, bin_edges = np.histogram(img.ravel(), bins=nbins)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    w0 = np.cumsum(hist)
    w1 = hist.sum() - w0
    m0 = np.cumsum(hist * centers) / np.where(w0 == 0, 1, w0)
    m1 = ((hist * centers).sum() - np.cumsum(hist * centers)) / np.where(w1 == 0, 1, w1)
    var_between = w0[:-1] * w1[:-1] * (m0[:-1] - m1[:-1])**2
    idx = np.nanargmax(var_between)
    return centers[idx]

def defect_mask_simple_float(gray):
    ret = otsu_from_floats(gray, nbins=256)
    mask = (gray <= ret).astype(np.uint8)
    p = mask.mean()
    if p < FG_RATIO_MIN or p > FG_RATIO_MAX:
        return ret, np.zeros_like(mask, dtype=np.float32), np.zeros_like(mask, dtype=np.float32)
    bw = mask.astype(np.float32)
    return ret, bw, mask.astype(np.float32)

# -------- Visualization (Option A) --------
def visualize_single_with_hist(gray, bw, mask, ret, path):
    gray_rgb = np.stack([gray]*3, axis=-1)
    mask_rgb = np.zeros_like(gray_rgb)
    mask_rgb[..., 0] = mask
    overlay = gray_rgb * 0.7 + mask_rgb * 0.3

    titles = ['Original', 'Binary', 'Mask', 'Overlay']
    images = [gray, bw, mask, overlay]

    plt.figure(figsize=(12,5))
    for i, (im, t) in enumerate(zip(images, titles),1):
        ax = plt.subplot(2,4,i)
        cmap = 'gray' if im.ndim==2 else None
        ax.imshow(im, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(t, fontsize=8)
        ax.axis('off')

    hist, bins = np.histogram(gray.ravel(), bins=256)
    ax_hist = plt.subplot(2,1,2)
    ax_hist.plot(bins[:-1], hist)
    ax_hist.axvline(ret, color='r', linestyle='--', label=f'T={ret:.3f}')
    ax_hist.set_xlim(0,1)
    ax_hist.set_xlabel('Gray level')
    ax_hist.set_ylabel('Pixel count')
    ax_hist.legend(fontsize=8)

    plt.suptitle(os.path.basename(path))
    plt.tight_layout(rect=[0,0.03,1,0.95])
    plt.show()

# -------- Main --------
def main():
    ds = CachedImageDataset(CACHE_FILE)

    if SHOW_PATH:
        # First try exact path match
        SHOW = SHOW_PATH if SHOW_PATH in ds.paths else None
        if SHOW is None:
            # Try matching by filename only
            fn = os.path.basename(SHOW_PATH)
            cands = [p for p in ds.paths if os.path.basename(p)==fn]
            if len(cands)==1:
                SHOW = cands[0]
            elif len(cands)>1:
                SHOW = cands[0]  # multiple matches, take the first
            else:
                print(f"Path '{SHOW_PATH}' not found. Available samples:")
                print(ds.paths[:10], "...")
                return
        idx = ds.paths.index(SHOW)
        gray, _ = ds[idx]
        ret, bw, mask = defect_mask_simple_float(gray)
        visualize_single_with_hist(gray, bw, mask, ret, SHOW_PATH)
        return

    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda x:x)
    os.makedirs(os.path.dirname(CSV_OUT), exist_ok=True)
    with open(CSV_OUT,'w',newline='') as fp:
        writer = csv.writer(fp)
        writer.writerow(['path','defect_ratio'])
        for batch in loader:
            for gray, path in batch:
                _, bw, mask = defect_mask_simple_float(gray)
                ratio = mask.sum()/mask.size
                writer.writerow([path, f"{ratio:.6f}"])
    print(f"CSV written to: {CSV_OUT}")

if __name__=='__main__':
    main()
