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
from sklearn.manifold import TSNE
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
keep_channels = [cha]        # can be list(range(16)) or a single channel [cha]

# —— Load gid→name mapping —— #
def load_group_names(cache_path):
    info_path = os.path.join(os.path.dirname(cache_path), 'info.txt')
    pattern   = re.compile(r'^\s*(\d+)\s*:\s*([^\s(]+)')
    mapping   = {}
    if os.path.exists(info_path):
        with open(info_path, 'r', encoding='utf-8') as f:
            for line in f:
                m = pattern.match(line)
                if m:
                    mapping[int(m.group(1))] = m.group(2)
    return mapping

group_names = load_group_names(cache_file)

# =========================
# 2. Main pipeline: Load frozen model + t-SNE visualization + interaction
# =========================
def extract_and_plot_all():
    # Fix random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Dataset loading and splitting
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
    # Filter out mismatched parameters (e.g., fc2)
    own_state = model.state_dict()
    filtered = {k: v for k, v in state.items() if k in own_state and v.size() == own_state[k].size()}
    own_state.update(filtered)
    model.load_state_dict(own_state)
    model.to(device).eval()

    # Hook to capture penultimate features (output of bn2)
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

    # —— Subsample per group (limit before t-SNE) —— #
    idxs = []
    rng = np.random.RandomState(seed)
    for grp in np.unique(G):
        grp_idx = np.where(G == grp)[0]
        if len(grp_idx) > max_per_group:
            grp_idx = rng.choice(grp_idx, max_per_group, replace=False)
        idxs.extend(grp_idx.tolist())
    idxs = np.array(idxs)
    H_sub = H[idxs]     # [n_points, D]
    G_sub = G[idxs]     # [n_points]

    # —— Pre-reduction (PCA → ≤50 dims) to denoise and speed up —— #
    pca_dim = int(min(50, H_sub.shape[1]))
    H_pca = PCA(n_components=pca_dim, random_state=seed).fit_transform(H_sub)

    # —— Run t-SNE to 2D (compatible with old sklearn versions) —— #
    n_points = H_pca.shape[0]
    perp = int(max(5, min(50, round(n_points / 100))))  # adaptive perplexity
    tsne_kwargs = dict(
        n_components=2,
        perplexity=perp,
        learning_rate=200.0,     # numeric value for compatibility (instead of 'auto')
        init='random',
        metric='euclidean',
        random_state=seed,
        verbose=1
        # method='barnes_hut',   # uncomment if needed; for large data use 'exact'
    )
    try:
        H_emb = TSNE(n_iter=1000, **tsne_kwargs).fit_transform(H_pca)
    except TypeError:
        # Old sklearn may not support n_iter: fallback to default (usually 1000)
        H_emb = TSNE(**tsne_kwargs).fit_transform(H_pca)

    # —— Plot: t-SNE-1 vs t-SNE-2 —— #
    fig, ax = plt.subplots(figsize=(8, 7))
    sc = ax.scatter(
        H_emb[:, 0], H_emb[:, 1],
        c=G_sub, cmap='tab10', s=30, picker=True, alpha=0.8
    )
    ax.set_title(f"t-SNE of Frozen-Model Hidden Features (perplexity={perp})")
    ax.set_xlabel("t-SNE-1")
    ax.set_ylabel("t-SNE-2")

    # Colorbar: only groups appearing in the subset
    uniq = sorted(np.unique(G_sub).tolist())
    cbar = fig.colorbar(sc, ax=ax, label="Group")
    cbar.set_ticks(uniq)
    cbar.set_ticklabels([group_names.get(k, str(k)) for k in uniq])

    # Interactive highlight & popup (show selected patch)
    def on_pick(event):
        ind0 = int(event.ind[0])     # index in subset
        global_ind = int(idxs[ind0]) # map back to combined dataset index
        x0, y0 = float(H_emb[ind0, 0]), float(H_emb[ind0, 1])

        hl, = ax.plot(x0, y0,
                      marker='o', markersize=12,
                      markerfacecolor='none',
                      markeredgecolor='red', linewidth=2)
        fig.canvas.draw_idle()

        img_t, gid = combined[global_ind]
        fig_img = plt.figure(figsize=(4, 4))
        plt.imshow(np.asarray(img_t.squeeze()), cmap='gray', vmin=0, vmax=1)
        plt.title(f"Group {group_names.get(int(gid), str(int(gid)))}")
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
