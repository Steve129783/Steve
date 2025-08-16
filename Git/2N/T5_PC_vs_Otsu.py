#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import random
import numpy as np
import torch
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
from c1_frozen import CNN  # only import CNN with chan_mask support

# =========================
# 1. Configuration
# =========================
file_name      = '1_2_3_4'
cha            = 1
cache_file     = rf'E:\Project_SNV\1N\1_Pack\{file_name}\data_cache.pt'
model_path     = rf'E:\Project_SNV\2N\c1_cache\{file_name}\{cha}\best_model_{cha}.pth'
ratio_csv      = rf'E:\Project_SNV\2N\c1_cache\{file_name}\defect_ratios_o.csv'
seed           = 42
batch_size     = 32
num_workers    = 0
in_shape       = (50, 50)
device         = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
max_per_group  = 1500
keep_channels  = [cha]

# —— Read gid → name mapping —— #
def load_group_names(cache_path):
    info = {}
    p = re.compile(r'^\s*(\d+)\s*:\s*([^\s(]+)')
    info_path = Path(cache_path).parent / 'info.txt'
    with open(info_path, encoding='utf-8') as f:
        for line in f:
            m = p.match(line)
            if m:
                info[int(m.group(1))] = m.group(2)
    return info

group_names = load_group_names(cache_file)
n_classes   = len(group_names)

# -------- Float Otsu thresholding --------
def otsu_from_floats(img, nbins=256):
    hist, bin_edges = np.histogram(img.ravel(), bins=nbins)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    w0 = np.cumsum(hist)
    total = hist.sum()
    w1 = total - w0
    cum_sum = np.cumsum(hist * centers)
    total_sum = cum_sum[-1]
    m0 = cum_sum / np.where(w0==0, 1, w0)
    m1 = (total_sum - cum_sum) / np.where(w1==0, 1, w1)
    var_between = w0[:-1] * w1[:-1] * (m0[:-1] - m1[:-1])**2
    idx = np.nanargmax(var_between)
    return centers[idx]

def defect_mask_otsu_float(gray):
    # gray: float32 in [0,1]
    ret = otsu_from_floats(gray, nbins=256)
    mask = (gray <= ret)         # boolean mask
    bw = mask.astype(np.float32) # float binary image
    mask_u8 = mask.astype(np.uint8)
    return ret, bw, mask_u8

# =========================
# 2. Dataset with path
# =========================
class PathDataset(Dataset):
    def __init__(self, cache_path):
        data = torch.load(cache_path, map_location='cpu')
        self.images = data['images']                          # Tensor [N,1,H,W]
        self.labels = torch.tensor(data['group_ids'], dtype=torch.long)
        self.paths  = data.get('paths', [None] * len(self.labels))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx], self.paths[idx]

# =========================
# 3. Main routine: PC1 vs defect_ratio + Float Otsu
# =========================
def extract_and_plot_from_csv():
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Read defect_ratio CSV
    df = pd.read_csv(ratio_csv, dtype={'path': str, 'defect_ratio': float})
    ratio_dict = dict(zip(df['path'], df['defect_ratio']))

    # Dataset & model
    ds = PathDataset(cache_file)
    model = CNN(in_shape=in_shape,
                n_classes=n_classes,
                keep_channels=keep_channels).to(device).eval()

    # Partially load pretrained weights: skip fc2 layer
    ckpt = torch.load(model_path, map_location=device)
    model_dict = model.state_dict()
    pretrained = {k: v for k, v in ckpt.items() if (k in model_dict) and (not k.startswith('fc2.'))}
    model_dict.update(pretrained)
    model.load_state_dict(model_dict)

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # hook into penultimate layer (bn2)
    feats = []
    def hook_fn(m, inp, out):
        feats.append(out.detach().cpu().numpy())
    handle = model.bn2.register_forward_hook(hook_fn)

    all_labels, all_paths = [], []
    with torch.no_grad():
        for x, y, paths in loader:
            _ = model(x.to(device).float())   # trigger hook
            all_labels.extend(y.numpy())
            all_paths.extend(paths)
    handle.remove()

    H = np.vstack(feats)
    G = np.array(all_labels)
    defect = np.array([ratio_dict[p] for p in all_paths])

    # PCA → PC1
    pc1 = PCA(n_components=1, random_state=seed).fit_transform(H).flatten()

    # Subsampling limit per group
    idxs = []
    for g in np.unique(G):
        inds = np.where(G == g)[0]
        if len(inds) > max_per_group:
            inds = np.random.RandomState(seed).choice(inds, max_per_group, replace=False)
        idxs.extend(inds.tolist())
    idxs = np.array(idxs)

    # Pearson correlation
    r, _ = pearsonr(pc1[idxs], defect[idxs])
    print(f"Pearson r (PC1 vs Defect Ratio): {r:.3f}")

    # Scatter plot
    fig, ax = plt.subplots(figsize=(6,6))
    sc = ax.scatter(pc1[idxs], defect[idxs], c=G[idxs], cmap='tab10',
                    s=20, alpha=0.7, picker=5)
    ax.set_title(f"PC1 vs Defect Ratio  (r={r:.3f})")
    ax.set_xlabel("PC1"); ax.set_ylabel("Defect Ratio")
    cbar = fig.colorbar(sc, ax=ax, label="Group")
    ticks = sorted(group_names.keys())
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([group_names[t] for t in ticks])

    # Fit and plot regression line
    x1, y1 = pc1[idxs], defect[idxs]
    m, b = np.polyfit(x1, y1, 1)
    xs = np.linspace(x1.min(), x1.max(), 100)
    ax.plot(xs, m*xs + b, 'r-', label=f"y={m:.3f}x+{b:.3f}")
    ax.legend(loc='upper left')

    # Interactive highlight & popup
    highlights = {}
    def on_pick(event):
        sel = idxs[event.ind[0]]
        # toggle highlight
        if sel in highlights:
            highlights[sel].remove()
            highlights.pop(sel)
            fig.canvas.draw_idle()
        hl, = ax.plot(pc1[sel], defect[sel], 'o', ms=12, mfc='none', mec='red', mew=2)
        highlights[sel] = hl
        fig.canvas.draw_idle()

        # float32 image & Otsu
        img_t, gid, path = ds[sel]
        gray = img_t.squeeze().cpu().numpy().astype(np.float32)
        # already normalized to [0,1]
        ret, bw, mask = defect_mask_otsu_float(gray)

        titles = ['Original', 'Binary', 'Mask', 'Overlay']
        orig = gray; binf = bw; mdisp = mask.astype(np.float32)
        orig3 = np.stack([orig]*3, axis=-1)
        overlay = orig3.copy()
        overlay[mask==1] = [1,0,0]

        win = plt.figure(figsize=(8,2))
        for i,(im,t) in enumerate(zip([orig, binf, mdisp, overlay], titles),1):
            axp = win.add_subplot(1,4,i)
            cmap = 'gray' if im.ndim==2 else None
            axp.imshow(im, cmap=cmap, vmin=0, vmax=1)
            axp.set_title(f"{t}\n{os.path.basename(path)}", fontsize=8)
            axp.axis('off')
        plt.tight_layout()

        def on_close(evt):
            art = highlights.pop(sel, None)
            if art:
                art.remove()
                fig.canvas.draw_idle()
        win.canvas.mpl_connect('close_event', on_close)
        plt.show()  # block until closed

    fig.canvas.mpl_connect('pick_event', on_pick)
    plt.show()  # block main plot

if __name__ == "__main__":
    extract_and_plot_from_csv()
