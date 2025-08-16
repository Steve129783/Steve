#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from PIL import Image
from c2_CNN_model import CachedImageDataset, CNN

# =========================
# 1. Configuration
# =========================
file_name     = 'h_c_l'
cache_file    = rf'E:\Project_SNV\1N\1_Pack\{file_name}\data_cache.pt'
model_path    = rf'E:\Project_SNV\1N\c2_cache\{file_name}\best_model.pth'
seed          = 42
batch_size    = 32
num_workers   = 0
in_shape      = (50, 50)
device        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
max_per_group = 1500

# —— Load gid → name mapping —— #
def load_group_names(cache_path):
    info = {}
    p = re.compile(r'^\s*(\d+)\s*:\s*([^\s(]+)')
    path = os.path.join(os.path.dirname(cache_path), 'info.txt')
    with open(path, encoding='utf-8') as f:
        for line in f:
            m = p.match(line)
            if m:
                info[int(m.group(1))] = m.group(2)
    return info

group_names = load_group_names(cache_file)

# =========================
# Main routine: Extract features and plot PC1 vs NMB (contrast)
# with interactive picking
# =========================
def extract_and_plot_contrast():
    # Fix random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Load dataset (enable return_path=True to get original file paths)
    ds = CachedImageDataset(cache_file, return_path=True)

    # Load checkpoint & infer n_classes automatically
    ckpt = torch.load(model_path, map_location=device)
    linear_keys = [k for k, v in ckpt.items() if k.endswith('weight') and v.dim() == 2]
    last_lin = sorted(linear_keys)[-1]
    n_classes = ckpt[last_lin].shape[0]
    print(f"Detected {n_classes} classes from checkpoint key '{last_lin}'")

    # Build model
    model = CNN(in_shape, n_classes).to(device)
    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    if missing or unexpected:
        print("Missing keys:", missing)
        print("Unexpected keys:", unexpected)
    model.eval()

    # DataLoader
    loader = DataLoader(ds, batch_size=batch_size,
                        shuffle=False, num_workers=num_workers)

    all_h, all_contrast, all_g = [], [], []
    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device)
            _, h = model(x)
            all_h.append(h.cpu().numpy())

            # Compute contrast: (mean-min) / (max-min)
            imgs = x.cpu().numpy().astype(np.float32)
            for im in imgs:
                gray = im.squeeze()
                mn, mx = gray.min(), gray.max()
                contrast = (gray.mean() - mn) / (mx - mn) if mx != mn else 0.0
                all_contrast.append(contrast)

            all_g.extend(y.cpu().numpy())

    H        = np.vstack(all_h)
    contrast = np.array(all_contrast)
    grp      = np.array(all_g)

    # PCA → PC1
    pc1 = PCA(n_components=1, random_state=seed).fit_transform(H).flatten()

    # Downsample if a group has more than max_per_group samples
    idxs = []
    for g in np.unique(grp):
        gi = np.where(grp == g)[0]
        if len(gi) > max_per_group:
            gi = np.random.RandomState(seed).choice(gi, max_per_group, replace=False)
        idxs.extend(gi.tolist())
    idxs = np.array(idxs)

    # Plot
    fig, ax = plt.subplots(figsize=(7,7))
    sc = ax.scatter(pc1[idxs], contrast[idxs],
                    c=grp[idxs], cmap='tab10',
                    s=25, alpha=0.7, picker=5)
    ax.set_title("PC1 vs NMB")
    ax.set_xlabel("PC1")
    ax.set_ylabel("NMB")
    cbar = fig.colorbar(sc, ax=ax, label="Group")
    cbar.set_ticks(sorted(group_names.keys()))
    cbar.set_ticklabels([group_names[int(t)] for t in cbar.get_ticks()])

    # Interactive picking
    highlights = {}
    def on_pick(event):
        ind0 = event.ind[0]
        sel  = idxs[ind0]
        x0, y0 = pc1[sel], contrast[sel]

        # Highlight selected point
        hl, = ax.plot(x0, y0, marker='o', markersize=12,
                      markerfacecolor='none', markeredgecolor='red', linewidth=2)
        highlights[sel] = hl
        fig.canvas.draw_idle()

        # Popup: left = original, right = processed
        img_t, gid, path = ds[sel]
        proc_img = img_t.squeeze().numpy()
        orig_img = np.array(Image.open(path))  # Read original uint16 image

        fig2, (ax1, ax2) = plt.subplots(1,2, figsize=(8,4))
        ax2.imshow(proc_img, cmap='gray', vmin=0, vmax=1)
        ax2.set_title(f"Processed\nGroup {group_names[int(gid)]}")
        ax2.axis('off')
        ax1.imshow(orig_img, cmap='gray',
                   vmin=0, vmax=np.iinfo(orig_img.dtype).max)
        ax1.set_title("Original (uint16)")
        ax1.axis('off')
        plt.tight_layout()
        plt.show()

        # Remove highlight when popup closes
        def on_close(evt):
            art = highlights.pop(sel, None)
            if art:
                art.remove()
                fig.canvas.draw_idle()
        fig2.canvas.mpl_connect('close_event', on_close)

    fig.canvas.mpl_connect('pick_event', on_pick)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    extract_and_plot_contrast()
