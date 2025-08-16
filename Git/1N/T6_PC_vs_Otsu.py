#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import random
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from pathlib import Path
from c2_CNN_model import CachedImageDataset, CNN

# =========================
# 1. Configuration
# =========================
file_name     = '1_2_3_4'
cache_file    = rf'E:\Project_SNV\1N\1_Pack\{file_name}\data_cache.pt'
model_path    = rf'E:\Project_SNV\1N\c1_cache\{file_name}\best_model.pth'
ratio_csv     = rf'E:\Project_SNV\1N\c1_cache\{file_name}\defect_ratios_o.csv'
seed          = 42
batch_size    = 32
num_workers   = 0
in_shape      = (50, 50)
device        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
max_per_group = 1500

# Foreground ratio thresholds
dtype = np.uint8
FG_RATIO_MIN = 0
FG_RATIO_MAX = 1

# —— Load gid→name mapping ——
def load_group_names(cache_path):
    info = {}
    p = re.compile(r'^\s*(\d+)\s*:\s*([^\s(]+)')
    info_path = Path(cache_path).parent / 'info.txt'
    if not info_path.exists():
        return {}
    with open(info_path, encoding='utf-8') as f:
        for line in f:
            m = p.match(line)
            if m:
                info[int(m.group(1))] = m.group(2)
    return info

group_names = load_group_names(cache_file)

# Custom Otsu threshold implementation
def otsu_threshold(img, nbins=256):
    hist, bin_edges = np.histogram(img.ravel(), bins=nbins)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    w0 = np.cumsum(hist).astype(float)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * centers) / np.maximum(w0, 1)
    m1 = (hist * centers).sum() - np.cumsum(hist * centers)
    m1 /= np.maximum(w1, 1)
    var_between = w0 * w1 * (m0 - m1)**2
    idx = np.nanargmax(var_between)
    return centers[idx]

# Generate defect mask
def compute_mask(gray):
    th = otsu_threshold(gray)
    mask = (gray <= th).astype(np.uint8)
    bw = mask.astype(np.float32)
    return th, bw, mask

# =========================
# 2. Main Routine
# =========================
def extract_and_plot_from_csv():
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    df = pd.read_csv(ratio_csv)
    ratio_dict = dict(zip(df['path'], df['defect_ratio']))

    ds = CachedImageDataset(cache_file, transform=None, return_path=True)

    # Dynamically infer number of classes from checkpoint
    state = torch.load(model_path, map_location=device)
    linear_keys = [k for k,v in state.items() if v.ndim == 2]
    last_key = sorted(linear_keys)[-1]
    n_classes = state[last_key].shape[0]

    model = CNN(in_shape, n_classes).to(device)
    model.load_state_dict(state)
    model.eval()

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    feats_list, grp_list, path_list = [], [], []
    with torch.no_grad():
        for imgs, labels, paths in loader:
            feats = model(imgs.to(device).float())[1].cpu().numpy()
            feats_list.append(feats)
            grp_list.extend(labels.numpy())
            path_list.extend(paths)
    H = np.vstack(feats_list)

    defect_ratios = np.array([ratio_dict.get(p, np.nan) for p in path_list])
    grp = np.array(grp_list)

    # PCA to 1D
    pc1 = PCA(n_components=1, random_state=seed).fit_transform(H).ravel()

    # Limit number of samples per group
    idxs = []
    for g in np.unique(grp):
        inds = np.where(grp == g)[0]
        if len(inds) > max_per_group:
            inds = np.random.RandomState(seed).choice(inds, max_per_group, replace=False)
        idxs.extend(inds.tolist())
    idxs = np.array(idxs)

    # Scatter plot
    fig, ax = plt.subplots(figsize=(6,6))
    sc = ax.scatter(pc1[idxs], defect_ratios[idxs], c=grp[idxs], cmap='tab10',
                    s=20, alpha=0.7, picker=5)
    ax.set_title('PC1 vs Defect Ratio')
    ax.set_xlabel('PC1'); ax.set_ylabel('Defect Ratio')

    # Align colorbar
    ticks = np.unique(grp[idxs])
    cbar = fig.colorbar(sc, ax=ax, label='Group')
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([group_names.get(int(t), str(int(t))) for t in ticks])

    # Fit regression line
    m, b = np.polyfit(pc1[idxs], defect_ratios[idxs], 1)
    xs = np.linspace(pc1[idxs].min(), pc1[idxs].max(), 100)
    ax.plot(xs, m*xs + b, 'r-', label=f'y={m:.3f}x+{b:.3f}')
    ax.legend(loc='upper left')

    highlights = {}
    def on_pick(event):
        sel = event.ind[0]
        idx = idxs[sel]
        if idx in highlights:
            highlights[idx].remove()
            del highlights[idx]
            fig.canvas.draw_idle()
            return
        hl, = ax.plot(pc1[idx], defect_ratios[idx], 'o', ms=12, mfc='none', mec='r', mew=2)
        highlights[idx] = hl
        fig.canvas.draw_idle()

        img_tensor, _, path = ds[idx]
        gray = img_tensor.squeeze().cpu().numpy()
        th, bw, mask = compute_mask(gray)

        titles = ['Original', 'Binary', 'Mask', 'Overlay']
        orig3 = np.stack([gray]*3, axis=-1)
        overlay = orig3.copy(); overlay[mask==1] = [1,0,0]
        images = [gray, bw, mask.astype(np.float32), overlay]

        popup = plt.figure(figsize=(8,2))
        for i,(im, t) in enumerate(zip(images, titles), 1):
            axp = popup.add_subplot(1,4,i)
            cmap = 'gray' if im.ndim==2 else None
            axp.imshow(im, cmap=cmap, vmin=0, vmax=1, interpolation='nearest')
            axp.set_title(f'{t}\n{os.path.basename(path)}', fontsize=8)
            axp.axis('off')
        plt.tight_layout()
        def on_close(evt):
            if idx in highlights:
                highlights[idx].remove()
                del highlights[idx]
                fig.canvas.draw_idle()
        popup.canvas.mpl_connect('close_event', on_close)
        plt.show()

    fig.canvas.mpl_connect('pick_event', on_pick)
    plt.show()

if __name__ == '__main__':
    extract_and_plot_from_csv()
