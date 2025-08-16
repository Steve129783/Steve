#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, random
import numpy as np
import torch
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.decomposition import PCA
from c2_CNN_model import CNN  # import the CNN from your training script

# =========================
# 1. Configuration
# =========================
file_name   = '1_2_3_4'
cache_file  = rf'E:/Project_SNV/1N/1_Pack/{file_name}/data_cache.pt'
model_path  = rf'E:/Project_SNV/1N/c2_cache/{file_name}/best_model.pth'  # NOTE: path points to the model saved by your training script
ratio_csv   = rf'E:/Project_SNV/1N/c2_cache/{file_name}/defect_ratios_m.csv'
seed        = 42
batch_size  = 32
num_workers = 0
in_shape    = (50, 50)
device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
max_per_grp = 1500
W, C = 5, 10/255
ksize = 2*W + 1

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
n_classes   = len(group_names)

# =========================
# 2. Mask & Overlay
# =========================
def defect_mask_float_gauss(gray_f32, ksize=ksize, C=C):
    local_mean = cv2.GaussianBlur(gray_f32, (ksize, ksize), (ksize - 1) / 6)
    mask = (gray_f32 < (local_mean - C)).astype(np.float32)
    return mask  # return mask directly

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
# 3. Lightweight Dataset (for this script only)
# =========================
class PathDataset(Dataset):
    def __init__(self, cache_path):
        data = torch.load(cache_path, map_location='cpu')
        self.images = data['images']      # [N,1,H,W] or array
        self.labels = torch.tensor(data['group_ids'], dtype=torch.long)
        self.paths  = data.get('paths', [None]*len(self.labels))

    def __len__(self): 
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.images[idx]
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        x = x.float()
        if x.ndim == 2:
            x = x.unsqueeze(0)
        elif x.ndim == 3 and x.shape[-1] == 1:
            x = x.permute(2,0,1)
        return x, self.labels[idx], self.paths[idx]

# =========================
# 4. Main: PCA + Interaction
# =========================
def extract_and_plot_defect_ratio():
    # Fix seeds
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    # Load defect_ratio
    df = pd.read_csv(ratio_csv)
    ratio_dict = dict(zip(df['path'], df['defect_ratio']))

    # Data & model
    ds = PathDataset(cache_file)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Infer number of classes from checkpoint (or directly use the known value)
    state = torch.load(model_path, map_location=device)
    # infer n_classes via the last linear layer weight shape (if n_classes was fixed at save time, you can just use that)
    linear_keys = [k for k,v in state.items() if v.ndim == 2]
    last_key = sorted(linear_keys)[-1]
    n_cls_from_ckpt = state[last_key].shape[0]

    model = CNN(in_shape=in_shape, n_classes=n_cls_from_ckpt).to(device)
    model.load_state_dict(state)
    model.eval()

    # Forward pass to extract features (use h returned by forward)
    feats, grps, paths = [], [], []
    with torch.no_grad():
        for x, y, p in loader:
            logits, h = model(x.to(device))
            feats.append(h.cpu().numpy())
            grps.extend(y.numpy())
            paths.extend(p)

    H = np.vstack(feats)
    G = np.array(grps)
    defect = np.array([ratio_dict.get(pp, np.nan) for pp in paths], dtype=np.float32)

    # PCA -> 2D
    pcs = PCA(n_components=2, random_state=seed).fit_transform(H)
    pc1, pc2 = pcs[:,0], pcs[:,1]

    # Limit per-group sample count (seed fixed)
    idxs = []
    for g in np.unique(G):
        inds = np.where(G==g)[0]
        if len(inds) > max_per_grp:
            inds = np.random.RandomState(seed).choice(inds, max_per_grp, replace=False)
        idxs.extend(inds.tolist())
    idxs = np.array(idxs)

    # Plot
    fig, ax = plt.subplots(figsize=(6,6))
    sc = ax.scatter(pc1[idxs], pc2[idxs], c=defect[idxs], cmap='viridis',
                    s=25, alpha=0.7, picker=5)
    ax.set_title('PC1 vs PC2 (color = Defect Ratio)')
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
    plt.colorbar(sc, ax=ax, label='Defect Ratio')

    highlights = {}
    def on_pick(event):
        i0 = event.ind[0]
        i  = idxs[i0]
        # toggle highlight
        if i in highlights:
            highlights[i].remove(); highlights.pop(i)
            fig.canvas.draw_idle()
            return
        hl, = ax.plot(pc1[i], pc2[i], marker='o', markersize=12,
                      markerfacecolor='none', markeredgecolor='red', linewidth=2)
        highlights[i] = hl
        fig.canvas.draw_idle()

        # Popup: original / mask / overlay
        img_t, gid, path = ds[i]
        gray = img_t.squeeze().numpy().astype(np.float32)
        mi, ma = gray.min(), gray.max()
        if ma > 1 or mi < 0:
            gray = (gray - mi) / (ma - mi + 1e-8)
        mask = defect_mask_float_gauss(gray)
        overlay = make_overlay(gray, mask, alpha=0.3)

        win = plt.figure(figsize=(8,2))
        for k,(im,title) in enumerate([
            (gray, 'Original (gray)'),
            (mask, 'Binary (float)'),
            (overlay, 'Overlay')
        ], 1):
            axp = win.add_subplot(1,3,k)
            if im.ndim == 2:
                axp.imshow(im, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
            else:
                axp.imshow(im, vmin=0, vmax=1, interpolation='nearest')
            axp.set_title(f"{title}\n{group_names.get(int(gid),'?')}/{os.path.basename(path or '')}", fontsize=8)
            axp.axis('off')
        plt.tight_layout()

        def on_close(evt):
            art = highlights.pop(i, None)
            if art: art.remove(); fig.canvas.draw_idle()
        win.canvas.mpl_connect('close_event', on_close)
        plt.show()

    fig.canvas.mpl_connect('pick_event', on_pick)
    plt.show()

if __name__ == '__main__':
    extract_and_plot_defect_ratio()
