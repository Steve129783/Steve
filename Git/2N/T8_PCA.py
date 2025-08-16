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
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.decomposition import PCA
from scipy.stats import spearmanr
from c1_frozen import CNN  # Import CNN with channel masking

# =========================
# 1. Configuration
# =========================
file_name      = '1_2_3_4'
cha            = 1
cache_file     = rf'E:/Project_SNV/1N/1_Pack/{file_name}/data_cache.pt'
model_path     = rf'E:\Project_SNV\2N\c2_cache\{file_name}\{cha}\best_model_{cha}.pth'
ratio_csv      = rf'E:/Project_SNV/2N/c2_cache/{file_name}/defect_ratios_m.csv'
seed           = 42
batch_size     = 32
num_workers    = 0
in_shape       = (50, 50)
device         = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
max_per_group  = 1500
keep_channels  = [cha]  # Channels kept in the frozen CNN first layer
W = 5
ksize = 2*W + 1
C     = 10/255

# --- Load gid → name mapping ---
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

# =========================
# 2. Float-based Gaussian adaptive threshold
# =========================
def defect_mask_float_gauss(gray_f32, ksize=ksize, C=C):
    """
    Perform Gaussian-weighted local thresholding on float32 grayscale images.
      - gray_f32: float32 grayscale image (range [0,1])
      - ksize: Gaussian kernel size (odd)
      - C: constant subtracted from the local mean
    Returns:
      - bw: float32 binary image (0.0 or 1.0)
      - mask: float32 mask (0.0 or 1.0)
    """
    local_mean = cv2.GaussianBlur(gray_f32, (ksize, ksize), (ksize-1)/6)
    mask = (gray_f32 < (local_mean - C)).astype(np.float32)
    bw = mask.copy()
    return bw, mask

# =========================
# 3. Custom Dataset
# =========================
class PathDataset(Dataset):
    def __init__(self, cache_path):
        data = torch.load(cache_path, map_location='cpu')
        self.images = data['images']  # Tensor [N,1,H,W]
        self.labels = torch.tensor(data['group_ids'], dtype=torch.long)
        self.paths  = data.get('paths', [None]*len(self.labels))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx], self.paths[idx]

# =========================
# 4. Main routine: PCA to 2D + float-threshold popup
# =========================
def extract_and_plot_defect_ratio():
    # Fix random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Read defect_ratio CSV
    df = pd.read_csv(ratio_csv)
    ratio_dict = dict(zip(df['path'], df['defect_ratio']))

    # Dataset & model
    ds = PathDataset(cache_file)
    loader = DataLoader(ds, batch_size=batch_size,
                        shuffle=False, num_workers=num_workers)
    model = CNN(in_shape=in_shape, n_classes=n_classes,
                keep_channels=keep_channels).to(device).eval()

    ckpt = torch.load(model_path, map_location=device)
    state = model.state_dict()
    pretrained = {k:v for k,v in ckpt.items() if k in state and not k.startswith('fc2')}
    state.update(pretrained); model.load_state_dict(state)

    # Hook the penultimate layer
    feats, all_grp, all_paths = [], [], []
    handle = model.bn2.register_forward_hook(
        lambda m,i,o: feats.append(o.detach().cpu().numpy())
    )
    with torch.no_grad():
        for x,y,paths in loader:
            _ = model(x.to(device).float())
            all_grp.extend(y.numpy()); all_paths.extend(paths)
    handle.remove()

    H = np.vstack(feats)
    G = np.array(all_grp)
    defect = np.array([ratio_dict.get(p, np.nan) for p in all_paths], dtype=np.float32)

    # PCA -> PC1, PC2
    pcs = PCA(n_components=2, random_state=seed).fit_transform(H)
    pc1, pc2 = pcs[:,0], pcs[:,1]

    # Sampling limit per group
    idxs = []
    for g in np.unique(G):
        inds = np.where(G==g)[0]
        if len(inds)>max_per_group:
            inds = np.random.RandomState(seed).choice(inds, max_per_group, replace=False)
        idxs.extend(inds.tolist())
    idxs = np.array(idxs)

    # Plot
    fig, ax = plt.subplots(figsize=(6,6))
    fig.suptitle("PC1 vs PC2 (color = Defect Ratio)")
    sc = ax.scatter(pc1[idxs], pc2[idxs], c=defect[idxs], cmap='viridis',
                    s=25, alpha=0.7, picker=5)
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
    plt.colorbar(sc, ax=ax, label='Defect Ratio')

    highlights = {}

    def on_pick(event):
        sel = event.ind[0]
        idx = idxs[sel]
        # Toggle highlight
        if idx in highlights:
            highlights[idx].remove()
            del highlights[idx]
            fig.canvas.draw_idle()
            return

        # Add highlight
        hl, = ax.plot(
            pc1[idx], pc2[idx],
            marker='o', markersize=12,
            markerfacecolor='none', markeredgecolor='red'
        )
        highlights[idx] = hl
        fig.canvas.draw_idle()

        # Popup: float-based segmentation visualization
        img_t, gid, path = ds[idx]
        gray = img_t.squeeze().numpy().astype(np.float32)
        mi, ma = gray.min(), gray.max()
        if ma>1 or mi<0:
            gray = (gray - mi) / (ma - mi)
        bw, mask = defect_mask_float_gauss(gray)

        overlay = np.stack([gray,gray,gray], axis=-1) \
                  + np.stack([mask, np.zeros_like(mask), np.zeros_like(mask)], axis=-1) * 0.3
        titles = ['Original','Binary(float)','Overlay']
        imgs = [gray, bw, overlay]

        win = plt.figure(figsize=(8,2))
        for i,(im,t) in enumerate(zip(imgs,titles),1):
            axp = win.add_subplot(1,3,i)
            cmap_ = 'gray' if im.ndim==2 else None
            axp.imshow(im, cmap=cmap_, vmin=0, vmax=1)
            axp.set_title(f"{t}\n{group_names[int(gid)]}/{os.path.basename(path)}", fontsize=8)
            axp.axis('off')
        plt.tight_layout()

        # Remove all highlights when popup closes
        def on_close(event):
            for h in highlights.values():
                try:
                    h.remove()
                except Exception:
                    pass
            highlights.clear()
            fig.canvas.draw_idle()
            win.canvas.mpl_disconnect(close_cid)

        close_cid = win.canvas.mpl_connect('close_event', on_close)
        plt.show()

    fig.canvas.mpl_connect('pick_event', on_pick)
    plt.show()

if __name__ == '__main__':
    extract_and_plot_defect_ratio()
