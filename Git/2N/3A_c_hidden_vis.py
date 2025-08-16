#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import random
import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, random_split
from c1_frozen import CachedImageDataset, CNN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# =========================
# 1. Configuration
# =========================
file_name     = 'h_c_l'
cha           = 11
cache_file    = rf'E:\Project_SNV\1N\1_Pack\{file_name}\data_cache.pt'
pretrained    = rf'E:\Project_SNV\2N\c2_cache\{file_name}\{cha}\best_model_{cha}.pth'
splits        = (0.7, 0.15, 0.15)
seed          = 42
batch_size    = 32
num_workers   = 0
in_shape      = (50, 50)
device        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
max_per_group = 1500
keep_channels = [cha]       # list of channels to keep

# —— Read gid→name mapping from info.txt ——#
def load_group_names(cache_path):
    info_path = os.path.join(os.path.dirname(cache_path), 'info.txt')
    pattern   = re.compile(r'^\s*(\d+)\s*:\s*([^\s(]+)')
    mapping   = {}
    with open(info_path, 'r', encoding='utf-8') as f:
        for line in f:
            m = pattern.match(line)
            if m:
                mapping[int(m.group(1))] = m.group(2)
    return mapping

group_names = load_group_names(cache_file)

# =========================
# 2. Main routine: load frozen model + PCA visualization + interaction
# =========================
def extract_and_plot_all():
    # Fix random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Dataset reading & splitting
    ds = CachedImageDataset(cache_file)
    N  = len(ds)
    n1 = int(splits[0] * N)
    n2 = int((splits[0] + splits[1]) * N)
    train_ds, val_ds, test_ds = random_split(
        ds, [n1, n2-n1, N-n2],
        generator=torch.Generator().manual_seed(seed)
    )
    combined = ConcatDataset([train_ds, val_ds, test_ds])

    # Load frozen model
    model = CNN(in_shape=in_shape, n_classes=len(group_names), keep_channels=keep_channels)
    state = torch.load(pretrained, map_location=device)
    # Filter out parameters that do not match the current model (e.g., fc2)
    own_state = model.state_dict()
    filtered = {k: v for k, v in state.items() if k in own_state and v.size() == own_state[k].size()}
    own_state.update(filtered)
    model.load_state_dict(own_state)
    model.to(device).eval()

    # Hook into penultimate features (bn2 output)
    feats, labels = [], []
    def hook_fn(module, inp, out):
        feats.append(out.detach().cpu().numpy())
    handle = model.bn2.register_forward_hook(hook_fn)

    loader = DataLoader(combined, batch_size=batch_size,
                        shuffle=False, num_workers=num_workers)
    with torch.no_grad():
        for x, y in loader:
            _ = model(x.to(device))
            labels.extend(y.numpy())
    handle.remove()

    H = np.vstack(feats)  # [N, hidden_dim]
    G = np.array(labels)  # [N]

    # PCA down to 2D, also print explained variance ratio of top 5 PCs
    pca = PCA(n_components=5, random_state=seed).fit(H)
    ratios = [round(r, 4) for r in pca.explained_variance_ratio_]
    print("Explained variance ratio of first 5 PCs:", ratios)
    H_pca = pca.transform(H)[:, :2]

    # Per-group subsampling (limit number of points per group)
    idxs = []
    for grp in np.unique(G):
        grp_idx = np.where(G == grp)[0]
        if len(grp_idx) > max_per_group:
            grp_idx = np.random.RandomState(seed).choice(
                grp_idx, max_per_group, replace=False)
        idxs.extend(grp_idx.tolist())
    idxs = np.array(idxs)

    # Scatter plot: PC1 vs PC2
    fig, ax = plt.subplots(figsize=(8, 7))
    sc = ax.scatter(
        H_pca[idxs, 0], H_pca[idxs, 1],
        c=G[idxs], cmap='tab10', s=30, picker=True, alpha=0.8
    )
    ax.set_title("PCA of Frozen-Model Hidden Features (max 1500/group)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

    # Colorbar: group mapping
    cbar = fig.colorbar(sc, ax=ax, label="Group")
    ticks = sorted(group_names.keys())
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([group_names[k] for k in ticks])

    # Interactive highlight & popup image view
    def on_pick(event):
        ind0 = event.ind[0]
        ind  = idxs[ind0]
        x0, y0 = H_pca[ind]
        hl, = ax.plot(x0, y0,
                      marker='o', markersize=12,
                      markerfacecolor='none',
                      markeredgecolor='red', linewidth=2)
        fig.canvas.draw_idle()

        img_t, gid = combined[ind]
        fig_img = plt.figure(figsize=(4, 4))
        plt.imshow(img_t.squeeze().numpy(), cmap='gray', vmin=0, vmax=1)
        plt.title(f"Group {group_names[int(gid)]}")
        plt.axis('off')

        def on_close(evt):
            hl.remove()
            fig.canvas.draw_idle()
        fig_img.canvas.mpl_connect('close_event', on_close)
        plt.show()

    fig.canvas.mpl_connect('pick_event', on_pick)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    extract_and_plot_all()
