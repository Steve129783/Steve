#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import os
import re
import random
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from scipy.stats import spearmanr
from pathlib import Path
from c2_CNN_model import CachedImageDataset, CNN

# =========================
# 1. Configuration
# =========================
file_name     = '1_2_3_4'
cache_file    = rf'E:\Project_SNV\1N\1_Pack\{file_name}\data_cache.pt'
model_path    = rf'E:\Project_SNV\1N\c2_cache\{file_name}\best_model.pth'
ratio_csv     = rf'E:\Project_SNV\1N\c2_cache\{file_name}\defect_ratios_m.csv'
seed          = 42
batch_size    = 32
num_workers   = 0
in_shape      = (50, 50)
device        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
max_per_group = 1500

# Adaptive threshold params (Gaussian kernel size must be odd)
W     = 5
ksize = 2 * W + 1
C     = 10 / 255.0

# —— Load gid→name mapping —— #
def load_group_names(cache_path):
    info = {}
    p = re.compile(r'^\s*(\d+)\s*:\s*([^\s(]+)')
    info_path = Path(cache_path).parent / 'info.txt'
    if info_path.exists():
        with open(info_path, encoding='utf-8') as f:
            for line in f:
                m = p.match(line)
                if m:
                    info[int(m.group(1))] = m.group(2)
    return info

group_names = load_group_names(cache_file)

# =========================
# 2. Mask + Overlay utilities
# =========================
def defect_mask_float_gauss(gray_f32, ksize=ksize, C=C):
    """Input gray must be float32 in [0,1]. Returns binary float mask (0/1)."""
    local_mean = cv2.GaussianBlur(gray_f32, (ksize, ksize), (ksize - 1) / 6)
    return (gray_f32 < (local_mean - C)).astype(np.float32)

def make_overlay(gray, mask, alpha=0.3):
    """Overlay binary mask (red, semi-transparent) onto grayscale image."""
    gray = np.clip(gray, 0, 1).astype(np.float32)
    m = (mask > 0).astype(np.float32)
    rgb = np.dstack([gray, gray, gray])
    red = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    idx = m.astype(bool)
    if np.any(idx):
        rgb[idx] = (1 - alpha) * rgb[idx] + alpha * red
    return np.clip(rgb, 0, 1)

# =========================
# 3. Main: Spearman’s ρ + Histograms + Interactive Scatter
# =========================
def extract_and_plot_defect_ratio():
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load defect ratios
    df = pd.read_csv(ratio_csv, dtype={'path': str, 'defect_ratio': float})
    ratio_dict = dict(zip(df['path'], df['defect_ratio']))

    # Dataset & model
    ds = CachedImageDataset(cache_file, transform=None, return_path=True)

    # Load checkpoint & infer number of classes
    state = torch.load(model_path, map_location=device)
    linear_keys = [k for k, v in state.items() if v.ndim == 2]
    last_key = sorted(linear_keys)[-1]
    n_classes = state[last_key].shape[0]

    model = CNN(in_shape, n_classes).to(device)
    model.load_state_dict(state)
    model.eval()

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    all_h, all_grp, all_paths = [], [], []
    with torch.no_grad():
        for x, y, paths in loader:
            out = model(x.to(device).float())
            # Ensure model forward returns (logits, hidden)
            if isinstance(out, (list, tuple)) and len(out) >= 2:
                h = out[1]
            else:
                raise RuntimeError("Your CNN.forward must return (logits, hidden).")
            all_h.append(h.cpu().numpy())
            all_grp.extend(y.numpy())
            all_paths.extend(paths)
    H = np.vstack(all_h)
    grp = np.array(all_grp)

    # Map paths → defect ratios (may contain NaN)
    defect = np.array([ratio_dict.get(p, np.nan) for p in all_paths], dtype=np.float32)

    # PCA → PC1, PC2
    pcs = PCA(n_components=2, random_state=seed).fit_transform(H)
    pc1, pc2 = pcs[:, 0], pcs[:, 1]

    # Per-group subsampling
    idxs = []
    for g in np.unique(grp):
        inds = np.where(grp == g)[0]
        if len(inds) > max_per_group:
            inds = np.random.RandomState(seed).choice(inds, max_per_group, replace=False)
        idxs.extend(inds.tolist())
    idxs = np.array(idxs, dtype=int)

    # Remove NaN ratios
    valid_mask = ~np.isnan(defect)
    idxs = idxs[valid_mask[idxs]]

    # Spearman’s rho (only valid samples)
    rho, pval = spearmanr(pc1[idxs], defect[idxs])

    # Three histograms (PC1, PC2, Defect Ratio)
    for data, title, xlabel in [
        (pc1[idxs], 'PC1', 'PC1'),
        (pc2[idxs], 'PC2', 'PC2'),
        (defect[idxs], 'Defect Ratio', 'Ratio')
    ]:
        plt.figure(figsize=(6, 4))
        plt.hist(data, bins=30)
        plt.title(f"Distribution of {title}")
        plt.xlabel(xlabel)
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()

    # Interactive scatter: colored by group
    fig, ax = plt.subplots(figsize=(6, 6))
    sc = ax.scatter(pc1[idxs], defect[idxs], c=grp[idxs], cmap='tab10',
                    s=25, alpha=0.7, picker=5)
    ax.set_title(f"PC1 vs Defect Ratio  (Spearman ρ={rho:.3f}, p={pval:.3f})")
    ax.set_xlabel('PC1')
    ax.set_ylabel('Defect Ratio')

    # Colorbar with group names
    ticks = np.unique(grp[idxs])
    cbar = fig.colorbar(sc, ax=ax, label='Group')
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([group_names.get(int(t), str(int(t))) for t in ticks])

    highlights = {}

    def on_pick(event):
        i0 = event.ind[0]
        ii = idxs[i0]

        # Toggle highlight
        if ii in highlights:
            highlights[ii].remove()
            highlights.pop(ii)
            fig.canvas.draw_idle()
            return
        hl, = ax.plot(pc1[ii], defect[ii], marker='o', markersize=12,
                      markerfacecolor='none', markeredgecolor='red', linewidth=2)
        highlights[ii] = hl
        fig.canvas.draw_idle()

        # Load image and normalize to [0,1]
        img_t, gid, path = ds[ii]
        gray = img_t.squeeze().cpu().numpy().astype(np.float32)
        mi, ma = gray.min(), gray.max()
        if ma > 1.0 or mi < 0.0:
            gray = (gray - mi) / (ma - mi + 1e-8)

        # Mask + overlay (red semi-transparent)
        mask = defect_mask_float_gauss(gray, ksize=ksize, C=C)
        overlay = make_overlay(gray, mask, alpha=0.3)

        # Popup: Original / Mask / Overlay
        win = plt.figure(figsize=(9, 3))
        ax1 = win.add_subplot(1, 3, 1)
        ax1.imshow(gray, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
        ax1.set_title('Original (gray)', fontsize=9)
        ax1.axis('off')

        ax2 = win.add_subplot(1, 3, 2)
        ax2.imshow(mask, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
        ax2.set_title('Binary Mask', fontsize=9)
        ax2.axis('off')

        ax3 = win.add_subplot(1, 3, 3)
        ax3.imshow(overlay, vmin=0, vmax=1, interpolation='nearest')
        ax3.set_title('Overlay', fontsize=9)
        ax3.axis('off')

        base = os.path.basename(path) if path is not None else ''
        win.suptitle(f"{group_names.get(int(gid), str(int(gid)))} — {base}", fontsize=9)
        plt.tight_layout()

        def on_close(_ev):
            art = highlights.pop(ii, None)
            if art:
                art.remove()
                fig.canvas.draw_idle()

        win.canvas.mpl_connect('close_event', on_close)
        plt.show()

    fig.canvas.mpl_connect('pick_event', on_pick)
    plt.show()

if __name__ == "__main__":
    extract_and_plot_defect_ratio()
