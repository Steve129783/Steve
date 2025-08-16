#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import os, re, random
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader, random_split, ConcatDataset
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Same as baseline script: use Dataset and CNN from c1_frozen (with keep_channels)
from c1_frozen import CachedImageDataset, CNN

# =========================
# 1) Configuration
# =========================
file_name      = '1_2_3_4'
cha            = 1
cache_file     = rf'E:\Project_SNV\1N\1_Pack\{file_name}\data_cache.pt'
model_path     = rf'E:\Project_SNV\2N\c2_cache\{file_name}\{cha}\best_model_{cha}.pth'
ratio_csv      = rf'E:\Project_SNV\2N\c2_cache\{file_name}\defect_ratios_m.csv'
splits         = (0.7, 0.15, 0.15)
seed           = 42
batch_size     = 32
num_workers    = 0
in_shape       = (50, 50)
device         = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
max_per_group  = 1500
keep_channels  = [cha]  # Channels preserved in the frozen model

# Masking parameters
W, C           = 5, 10/255
ksize          = 2*W + 1

# =========================
# 2) Utility functions
# =========================
def _norm_path(p):
    return os.path.normpath(p) if isinstance(p, str) else p

def load_group_names(cache_path):
    """Read gid→name mapping from info.txt (optional)."""
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

def build_path2gid(cache_path):
    """Build path→gid mapping from cache; normalize paths."""
    d = torch.load(cache_path, map_location='cpu')
    paths = d.get('paths', None)
    gids  = d.get('group_ids', None)
    if paths is None or gids is None:
        raise RuntimeError("cache_file missing 'paths' or 'group_ids'")
    return {_norm_path(p): int(g) for p, g in zip(paths, gids)}

def load_paths_from_cache(cache_path):
    """Read paths; if not available, use None placeholders."""
    d = torch.load(cache_path, map_location='cpu')
    if 'paths' not in d:
        return [None] * len(d['group_ids'])
    return [ _norm_path(p) for p in d['paths'] ]

class WithPaths(torch.utils.data.Dataset):
    """Wrapper: add path to original CachedImageDataset, returning (img, label, path)."""
    def __init__(self, base_ds, paths):
        assert len(base_ds) == len(paths), "Number of paths does not match dataset size"
        self.base = base_ds
        self.paths = paths
    def __len__(self): return len(self.base)
    def __getitem__(self, idx):
        x, y = self.base[idx]          # original returns (img, label)
        return x, y, self.paths[idx]   # now returns (img, label, path)

def defect_mask_float_gauss(gray_f32, ksize=ksize, C=C):
    """Float-based local Gaussian threshold (pixels darker than local mean - C are 1)."""
    local_mean = cv2.GaussianBlur(gray_f32, (ksize, ksize), (ksize - 1) / 6)
    mask = (gray_f32 < (local_mean - C)).astype(np.float32)
    return mask

# Optional: red overlay
def make_overlay(gray, mask, alpha=0.3):
    gray = np.clip(gray, 0, 1).astype(np.float32)
    rgb  = np.dstack([gray, gray, gray])
    red  = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    idx  = mask > 0
    if np.any(idx):
        rgb[idx] = (1 - alpha) * rgb[idx] + alpha * red
    return np.clip(rgb, 0, 1)

# =========================
# 3) Main pipeline (same dataflow as baseline script)
# =========================
def extract_and_plot_defect_ratio():
    # Fix randomness
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    # Read group names & path→gid
    group_names = load_group_names(cache_file)
    n_classes   = len(group_names) if len(group_names) > 0 else None
    path2gid    = build_path2gid(cache_file)

    # Read defect ratio and normalize paths
    df = pd.read_csv(ratio_csv)
    if 'path' not in df.columns or 'defect_ratio' not in df.columns:
        raise RuntimeError("ratio_csv missing 'path' or 'defect_ratio' columns")
    df['path']  = df['path'].map(_norm_path)
    ratio_dict  = dict(zip(df['path'], df['defect_ratio']))

    # === Same as baseline: CachedImageDataset → random_split(seed) → ConcatDataset ===
    base = CachedImageDataset(cache_file, transform=None)    # no longer return_path
    paths = load_paths_from_cache(cache_file)
    ds = WithPaths(base, paths)                              # now returns (img, label, path)

    N  = len(ds)
    n1 = int(splits[0] * N)
    n2 = int((splits[0] + splits[1]) * N)
    train_ds, val_ds, test_ds = random_split(
        ds, [n1, n2 - n1, N - n2],
        generator=torch.Generator().manual_seed(seed)
    )
    combined = ConcatDataset([train_ds, val_ds, test_ds])
    loader   = DataLoader(combined, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # === Model: frozen CNN (c1_frozen.CNN with keep_channels), load matching weights (skip fc2) ===
    if n_classes is None:
        # Fallback: if no info.txt, use max group_id + 1
        n_classes = max(path2gid.values()) + 1
    model = CNN(in_shape=in_shape, n_classes=n_classes, keep_channels=keep_channels).to(device).eval()
    ckpt  = torch.load(model_path, map_location=device)
    own   = model.state_dict()
    # Only load weights with exact shape match and not fc2
    filtered = {k: v for k, v in ckpt.items() if (k in own and v.shape == own[k].shape and not k.startswith('fc2'))}
    own.update(filtered); model.load_state_dict(own)

    # Hook penultimate layer (bn2): capture hidden features
    feats, all_paths = [], []
    def hook_fn(m, i, o): feats.append(o.detach().cpu().numpy())
    handle = model.bn2.register_forward_hook(hook_fn)

    with torch.no_grad():
        for x, y, p in loader:            # loader returns (img, label, path)
            _ = model(x.to(device).float())
            all_paths.extend(p)
    handle.remove()

    H = np.vstack(feats)                                  # [N, D]
    P = [ _norm_path(pp) for pp in all_paths ]            # normalize
    G = np.array([ path2gid.get(pp, -1) for pp in P ], dtype=int)            # true group IDs
    defect = np.array([ ratio_dict.get(pp, np.nan) for pp in P ], dtype=np.float32)

    # —— Per-group downsampling (same logic as baseline) ——
    idxs = []
    rng = np.random.RandomState(seed)
    for g in np.unique(G):
        inds = np.where(G == g)[0]
        if len(inds) > max_per_group:
            inds = rng.choice(inds, max_per_group, replace=False)
        idxs.extend(inds.tolist())
    idxs = np.array(idxs)

    H_sub      = H[idxs]
    G_sub      = G[idxs]
    defect_sub = defect[idxs]

    # —— PCA preprocessing (≤50 dims) ——
    pca_dim = int(min(50, H_sub.shape[1]))
    H_pca   = PCA(n_components=pca_dim, random_state=seed).fit_transform(H_sub)

    # —— t-SNE (adaptive perplexity, n_iter=1000) ——
    n_points = H_pca.shape[0]
    perp = int(max(5, min(50, round(n_points / 100))))
    tsne_kwargs = dict(
        n_components=2,
        perplexity=perp,
        learning_rate=200.0,
        init='random',
        metric='euclidean',
        random_state=seed,
        verbose=1,
    )
    try:
        H_emb = TSNE(n_iter=1000, **tsne_kwargs).fit_transform(H_pca)
    except TypeError:
        H_emb = TSNE(**tsne_kwargs).fit_transform(H_pca)

    E1, E2 = H_emb[:, 0], H_emb[:, 1]

    # —— Plot: color by defect ratio; click to show Original / Binary / Overlay ——
    fig, ax = plt.subplots(figsize=(6, 6))
    sc = ax.scatter(E1, E2, c=defect_sub, cmap='viridis', s=25, alpha=0.7, picker=5)
    ax.set_title(f"t-SNE of Hidden Features (perplexity={perp})\ncolor = Defect Ratio")
    ax.set_xlabel('t-SNE-1'); ax.set_ylabel('t-SNE-2')
    plt.colorbar(sc, ax=ax, label='Defect Ratio')

    highlights = {}
    def on_pick(event):
        i0  = int(event.ind[0])           # index in subsampled set
        idx = int(idxs[i0])               # back to combined global index

        # Toggle highlight
        if i0 in highlights:
            highlights[i0].remove(); highlights.pop(i0)
            fig.canvas.draw_idle()
            return
        hl, = ax.plot(E1[i0], E2[i0], marker='o', markersize=12,
                      markerfacecolor='none', markeredgecolor='red', linewidth=2)
        highlights[i0] = hl
        fig.canvas.draw_idle()

        # Fetch patch and display: Original / Binary / Overlay
        img_t, y_label, path = combined[idx]   # triple: (img, label, path)
        gray = img_t.squeeze().numpy().astype(np.float32)
        mi, ma = gray.min(), gray.max()
        if ma > 1 or mi < 0:
            gray = (gray - mi) / (ma - mi + 1e-8)

        mask    = defect_mask_float_gauss(gray)
        overlay = make_overlay(gray, mask, alpha=0.3)

        # Group name (fallback to gid if no info.txt)
        real_gid = path2gid.get(_norm_path(path), int(y_label))
        gname = load_group_names(cache_file).get(int(real_gid), str(int(real_gid)))

        win = plt.figure(figsize=(8, 2))
        for k, (im, title) in enumerate([
            (gray,    'Original (gray)'),
            (mask,    'Binary (float)'),
            (overlay, 'Overlay'),
        ], 1):
            axp = win.add_subplot(1, 3, k)
            if im.ndim == 2:
                axp.imshow(im, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
            else:
                axp.imshow(im, vmin=0, vmax=1, interpolation='nearest')
            axp.set_title(f"{title}\n{gname}/{os.path.basename(path or '')}", fontsize=8)
            axp.axis('off')
        plt.tight_layout()

        def on_close(evt):
            art = highlights.pop(i0, None)
            if art: art.remove()
            fig.canvas.draw_idle()
        win.canvas.mpl_connect('close_event', on_close)
        plt.show()

    fig.canvas.mpl_connect('pick_event', on_pick)
    plt.show()

if __name__ == '__main__':
    extract_and_plot_defect_ratio()
