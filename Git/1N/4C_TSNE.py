#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import re
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torchvision import transforms
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from PIL import Image
from c2_CNN_model import CachedImageDataset, CNN

# =========================
# 1. Configuration
# =========================
file_name     = '1_2_3_4'
cache_file    = rf'E:\Project_SNV\1N\1_Pack\{file_name}\data_cache.pt'
model_path    = rf'E:\Project_SNV\1N\c2_cache\{file_name}\best_model.pth'
splits        = (0.7, 0.15, 0.15)
seed          = 42
batch_size    = 32
num_workers   = 0   # ensure determinism
in_shape      = (50, 50)
device        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
max_per_group = 1500  # maximum points per group for visualization

# =============== Utility: read gid → name mapping ===============
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
# 2. Main routine
# =========================
def extract_and_plot_all():
    # ---- Deterministic settings ----
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # ---- Dataset loading and splitting ----
    ds = CachedImageDataset(cache_file, transform=None, return_path=True)
    N  = len(ds)
    n1 = int(splits[0] * N)
    n2 = int((splits[0] + splits[1]) * N)
    train_ds, val_ds, test_ds = random_split(
        ds, [n1, n2-n1, N-n2],
        generator=torch.Generator().manual_seed(seed)
    )

    # ---- Load model ----
    state = torch.load(model_path, map_location=device)
    linear_keys = [k for k,v in state.items() if k.endswith('weight') and v.dim()==2]
    last_key = sorted(linear_keys)[-1]
    n_classes = state[last_key].shape[0]
    model = CNN(in_shape, n_classes).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()

    # ---- Merge datasets and extract hidden features ----
    combined = ConcatDataset([train_ds, val_ds, test_ds])
    loader = DataLoader(combined, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    all_h, all_g, all_paths = [], [], []
    with torch.no_grad():
        for x,y,p in loader:
            # Assume CNN.forward returns (logits, hidden)
            _, h = model(x.to(device))
            all_h.append(h.float().cpu().numpy())
            all_g.extend(y.numpy())
            all_paths.extend(p)
    H = np.vstack(all_h)     # [N, D]
    G = np.array(all_g)      # [N]

    # ---- Subsample by group (limit to max_per_group per class) ----
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

    # ---- Pre-reduction (PCA → up to 50 dims) for speed & denoising ----
    pca_dim = int(min(50, H_sub.shape[1]))
    H_pca = PCA(n_components=pca_dim, random_state=seed).fit_transform(H_sub)

    # ---- t-SNE to 2D (supporting older sklearn versions) ----
    n_points = H_pca.shape[0]
    perp = int(max(5, min(50, round(n_points / 100))))  # adaptive perplexity
    tsne_kwargs = dict(
        n_components=2,
        perplexity=perp,
        learning_rate=200.0,   # numeric value is more compatible than 'auto'
        init='random',
        metric='euclidean',
        random_state=seed,
        verbose=1
        # method='barnes_hut', # uncomment if forced method is needed
    )
    try:
        # newer sklearn supports n_iter
        H_emb = TSNE(n_iter=1000, **tsne_kwargs).fit_transform(H_pca)
    except TypeError:
        # older version does not support n_iter → fallback
        H_emb = TSNE(**tsne_kwargs).fit_transform(H_pca)

    # ---- Plot (with clickable image preview) ----
    fig, ax = plt.subplots(figsize=(8, 7))
    sc = ax.scatter(H_emb[:, 0], H_emb[:, 1],
                    c=G_sub, cmap='tab10',
                    s=30, picker=True, alpha=0.8)
    ax.set_title(f"t-SNE of Hidden Features (perplexity={perp})")
    ax.set_xlabel("t-SNE-1"); ax.set_ylabel("t-SNE-2")

    # Colorbar only for groups present in current subset
    uniq = sorted(np.unique(G_sub).tolist())
    cbar = fig.colorbar(sc, ax=ax, label="Group")
    cbar.set_ticks(uniq)
    cbar.set_ticklabels([group_names.get(k, str(k)) for k in uniq])

    highlights = {}
    def on_pick(event):
        ind0 = int(event.ind[0])   # index in subset
        global_ind = int(idxs[ind0])  # map back to global index
        x0, y0 = float(H_emb[ind0, 0]), float(H_emb[ind0, 1])

        # Highlight selected point
        hl, = ax.plot(x0, y0,
                      marker='o', markersize=12,
                      markerfacecolor='none',
                      markeredgecolor='red',
                      linewidth=2)
        fig.canvas.draw_idle()

        # Popup: original vs processed
        img_tensor, gid, path = combined[global_ind]
        proc_img = np.asarray(img_tensor.squeeze())
        orig_img = np.array(Image.open(path))

        pf, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
        ax1.imshow(orig_img, cmap='gray',
                   vmin=0, vmax=np.iinfo(orig_img.dtype).max)
        ax1.set_title('Original (u16)')
        ax1.axis('off')
        ax2.imshow(proc_img, cmap='gray', vmin=0, vmax=1)
        ax2.set_title('Processed')
        ax2.axis('off')
        plt.tight_layout()
        plt.show()

        def on_close(evt):
            art = highlights.pop(pf, None)
            if art:
                art.remove()
                fig.canvas.draw_idle()
        highlights[pf] = hl
        pf.canvas.mpl_connect('close_event', on_close)

    fig.canvas.mpl_connect('pick_event', on_pick)
    plt.tight_layout()
    plt.show()

if __name__=='__main__':
    extract_and_plot_all()
