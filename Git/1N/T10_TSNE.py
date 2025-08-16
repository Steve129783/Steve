#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, random
import numpy as np
import torch
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader, random_split, ConcatDataset
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from PIL import Image

from c2_CNN_model import CachedImageDataset, CNN  # consistent with training

# =========================
# 1. Configuration
# =========================
file_name     = 'h_c_l'
cache_file    = rf'E:/Project_SNV/1N/1_Pack/{file_name}/data_cache.pt'
model_path    = rf'E:/Project_SNV/1N/c2_cache/{file_name}/best_model.pth'
ratio_csv     = rf'E:/Project_SNV/1N/c2_cache/{file_name}/defect_ratios_m.csv'
splits        = (0.7, 0.15, 0.15)
seed          = 42
batch_size    = 32
num_workers   = 0          # ensure determinism
in_shape      = (50, 50)
device        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
max_per_group = 1500       # max points shown per group
W, C          = 5, 10/255
ksize         = 2*W + 1

# —— Read gid→name mapping ——
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

# —— Build path→gid mapping from cache (for per-group sampling/colorbar labels) ——
def build_path2gid(cache_path):
    d = torch.load(cache_path, map_location='cpu')
    paths = d.get('paths', None)
    gids  = d.get('group_ids', None)
    if paths is None or gids is None:
        raise RuntimeError("cache_file is missing 'paths' or 'group_ids'")
    return {p: int(g) for p, g in zip(paths, gids)}

# =========================
# 2. Mask & Overlay utilities (unchanged functionality)
# =========================
def defect_mask_float_gauss(gray_f32, ksize=ksize, C=C):
    local_mean = cv2.GaussianBlur(gray_f32, (ksize, ksize), (ksize - 1) / 6)
    mask = (gray_f32 < (local_mean - C)).astype(np.float32)
    return mask

def make_overlay(gray, mask, alpha=0.3):
    gray = np.clip(gray, 0, 1).astype(np.float32)
    m = (mask > 0).astype(np.float32)
    rgb = np.dstack([gray, gray, gray])
    red = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    idx = m == 1
    if np.any(idx):
        rgb[idx] = (1 - alpha) * rgb[idx] + alpha * red
    return np.clip(rgb, 0, 1)

# =========================
# 3. Main: use the same dataset split procedure as the “standard code”
# =========================
def extract_and_plot_defect_ratio():
    # Fix seeds
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    # Read defect_ratio
    df = pd.read_csv(ratio_csv)
    ratio_dict = dict(zip(df['path'], df['defect_ratio']))

    # === Alignment with standard code: load/split in the same way ===
    ds = CachedImageDataset(cache_file, transform=None, return_path=True)
    N  = len(ds)
    n1 = int(splits[0] * N)
    n2 = int((splits[0] + splits[1]) * N)
    train_ds, val_ds, test_ds = random_split(
        ds, [n1, n2-n1, N-n2],
        generator=torch.Generator().manual_seed(seed)
    )
    combined = ConcatDataset([train_ds, val_ds, test_ds])
    loader   = DataLoader(combined, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # path→gid (for groupwise sampling and colorbar labels)
    path2gid = build_path2gid(cache_file)

    # Infer class count & load model (same approach as the standard code)
    state = torch.load(model_path, map_location=device)
    linear_keys = [k for k, v in state.items() if hasattr(v, 'dim') and v.dim() == 2]
    last_key = sorted(linear_keys)[-1]
    n_cls_from_ckpt = state[last_key].shape[0]

    model = CNN(in_shape=in_shape, n_classes=n_cls_from_ckpt).to(device)
    # allow non-strict to avoid minor key mismatches
    model.load_state_dict(state, strict=False)
    model.eval()

    # Forward pass to extract features (use hidden embedding returned by forward)
    feats, paths = [], []
    with torch.no_grad():
        for x, y, p in loader:
            _, h = model(x.to(device))
            feats.append(h.cpu().numpy())
            paths.extend(p)

    H = np.vstack(feats)                                        # [N, D]
    G = np.array([path2gid.get(pp, -1) for pp in paths], int)   # group ids
    defect = np.array([ratio_dict.get(pp, np.nan) for pp in paths], np.float32)

    # —— Per-group downsampling (to keep t-SNE size manageable), same logic as standard code ——
    idxs = []
    rng = np.random.RandomState(seed)
    for g in np.unique(G):
        inds = np.where(G == g)[0]
        if len(inds) > max_per_group:
            inds = rng.choice(inds, max_per_group, replace=False)
        idxs.extend(inds.tolist())
    idxs = np.array(idxs)

    H_sub      = H[idxs]          # [n_points, D]
    G_sub      = G[idxs]
    defect_sub = defect[idxs]

    # —— Pre-reduction (PCA → ≤50 dims) for speed/denoising ——
    pca_dim = int(min(50, H_sub.shape[1]))
    H_pca = PCA(n_components=pca_dim, random_state=seed).fit_transform(H_sub)

    # —— t-SNE to 2D (parameters aligned with your previous script) ——
    n_points = H_pca.shape[0]
    perp = int(max(5, min(50, round(n_points / 100))))  # adaptive perplexity
    tsne_kwargs = dict(
        n_components=2,
        perplexity=perp,
        learning_rate=200.0,     # numeric for compatibility with older sklearn
        init='random',
        metric='euclidean',
        random_state=seed,
        verbose=1,
        # method='barnes_hut',   # uncomment if you need to force it
    )
    try:
        H_emb = TSNE(n_iter=1000, **tsne_kwargs).fit_transform(H_pca)
    except TypeError:
        H_emb = TSNE(**tsne_kwargs).fit_transform(H_pca)

    emb1, emb2 = H_emb[:, 0], H_emb[:, 1]

    # Plot: color by defect ratio (other interactions unchanged)
    fig, ax = plt.subplots(figsize=(6,6))
    sc = ax.scatter(emb1, emb2, c=defect_sub, cmap='viridis',
                    s=25, alpha=0.7, picker=5)
    ax.set_title(f't-SNE of Hidden Features (perplexity={perp})\ncolor = Defect Ratio')
    ax.set_xlabel('t-SNE-1'); ax.set_ylabel('t-SNE-2')
    plt.colorbar(sc, ax=ax, label='Defect Ratio')

    highlights = {}
    def on_pick(event):
        i0 = int(event.ind[0])   # index in the subsampled set
        i  = int(idxs[i0])       # map back to global index in `combined`

        # Toggle highlight
        if i in highlights:
            highlights[i].remove(); highlights.pop(i)
            fig.canvas.draw_idle()
            return
        hl, = ax.plot(emb1[i0], emb2[i0], marker='o', markersize=12,
                      markerfacecolor='none', markeredgecolor='red', linewidth=2)
        highlights[i] = hl
        fig.canvas.draw_idle()

        # Popup: original / mask / overlay
        img_t, y_label, path = combined[i]  # CachedImageDataset returns (tensor, label, path)
        gray = img_t.squeeze().numpy().astype(np.float32)
        mi, ma = gray.min(), gray.max()
        if ma > 1 or mi < 0:  # normalize to [0,1] if needed
            gray = (gray - mi) / (ma - mi + 1e-8)

        mask = defect_mask_float_gauss(gray)
        overlay = make_overlay(gray, mask, alpha=0.3)

        real_gid = int(path2gid.get(path, -1))
        win = plt.figure(figsize=(8,2))
        for k, (im, title) in enumerate([
            (gray,   'Original (gray)'),
            (mask,   'Binary (float)'),
            (overlay,'Overlay')
        ], 1):
            axp = win.add_subplot(1,3,k)
            if im.ndim == 2:
                axp.imshow(im, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
            else:
                axp.imshow(im, vmin=0, vmax=1, interpolation='nearest')
            axp.set_title(f"{title}\n{group_names.get(real_gid,'?')}/{os.path.basename(path or '')}",
                          fontsize=8)
            axp.axis('off')
        plt.tight_layout()

        def on_close(evt):
            art = highlights.pop(i, None)
            if art:
                art.remove()
                fig.canvas.draw_idle()
        win.canvas.mpl_connect('close_event', on_close)
        plt.show()

    fig.canvas.mpl_connect('pick_event', on_pick)
    plt.show()

if __name__ == '__main__':
    extract_and_plot_defect_ratio()
