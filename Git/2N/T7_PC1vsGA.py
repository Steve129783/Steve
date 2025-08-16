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
from c1_frozen import CNN  # CNN with channel masking

# =========================
# 1. Configuration
# =========================
file_name     = '1_2_3_4'
cha           = 11
cache_file    = rf'E:/Project_SNV/1N/1_Pack/{file_name}/data_cache.pt'
model_path    = rf'E:\Project_SNV\2N\c2_cache\{file_name}\{cha}\best_model_{cha}.pth'
ratio_csv     = rf'E:/Project_SNV/2N/c2_cache/{file_name}/defect_ratios_m.csv'
seed          = 42
batch_size    = 32
num_workers   = 0
in_shape      = (50, 50)
device        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
max_per_group = 1500
keep_channels = [cha]
W = 5
ksize = 2*W + 1
C     = 10/255

# Read gid → name mapping
def load_group_names(cache_path):
    info = {}
    p = re.compile(r'^\s*(\d+)\s*:\s*([^\s(]+)')
    info_path = Path(cache_path).parent / 'info.txt'
    with open(info_path, 'r', encoding='utf-8') as f:
        for line in f:
            m = p.match(line)
            if m:
                info[int(m.group(1))] = m.group(2)
    return info

group_names = load_group_names(cache_file)
n_classes   = len(group_names)

# Float-based Gaussian adaptive threshold
def defect_mask_float_gauss(gray, ksize=ksize, C=C):
    local_mean = cv2.GaussianBlur(gray, (ksize, ksize), (ksize-1)/6)
    mask = (gray < (local_mean - C)).astype(np.float32)
    bw   = mask.copy()
    return bw, mask

# Dataset wrapper
class PathDataset(Dataset):
    def __init__(self, cache_path):
        data = torch.load(cache_path, map_location='cpu')
        self.images = data['images']      # [N,1,H,W]
        self.labels = torch.tensor(data['group_ids'], dtype=torch.long)
        self.paths  = data.get('paths', [None]*len(self.labels))

    def __len__(self): return len(self.labels)
    def __getitem__(self, idx): return self.images[idx], self.labels[idx], self.paths[idx]

# Main routine
def extract_and_plot_defect_ratio():
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    # Read defect_ratio CSV
    df = pd.read_csv(ratio_csv, dtype={'path': str, 'defect_ratio': float})
    ratio_dict = dict(zip(df['path'], df['defect_ratio']))

    ds = PathDataset(cache_file)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Model loading
    model = CNN(in_shape=in_shape, n_classes=n_classes, keep_channels=keep_channels).to(device).eval()
    ckpt = torch.load(model_path, map_location=device)
    state = model.state_dict()
    pretrained = {k:v for k,v in ckpt.items() if k in state and not k.startswith('fc2')}
    state.update(pretrained); model.load_state_dict(state)

    # Hook penultimate layer
    feats, groups, paths = [], [], []
    handle = model.bn2.register_forward_hook(lambda m,i,o: feats.append(o.detach().cpu().numpy()))
    with torch.no_grad():
        for x,y,p in loader:
            _ = model(x.to(device).float())
            groups.extend(y.numpy()); paths.extend(p)
    handle.remove()

    # Collect data
    H = np.vstack(feats)
    G = np.array(groups)
    defect = np.array([ratio_dict.get(pt, np.nan) for pt in paths], dtype=np.float32)

    # PCA 2D
    pcs = PCA(n_components=2, random_state=seed).fit_transform(H)
    pc1, pc2 = pcs[:,0], pcs[:,1]

    # Per-group sampling cap
    idxs = []
    for g in np.unique(G):
        inds = np.where(G==g)[0]
        if len(inds)>max_per_group:
            inds = np.random.RandomState(seed).choice(inds, max_per_group, replace=False)
        idxs.extend(inds.tolist())
    idxs = np.array(idxs)

    # Spearman correlation
    rho, pval = spearmanr(pc1[idxs], defect[idxs])
    print(f"Spearman's rho = {rho:.3f}, p-value = {pval:.6f}")

    # Histograms: PC1, PC2, Defect Ratio
    for data,title,xlab in [(pc1, 'Distribution of PC1', 'PC1'), 
                            (pc2, 'Distribution of PC2', 'PC2'), 
                            (defect, 'Distribution of Defect Ratio', 'Defect Ratio')]:
        
        plt.figure(figsize=(6,4))
        plt.hist(data[idxs], bins=30)
        plt.title(title); plt.xlabel(xlab); plt.ylabel('Frequency'); plt.grid(True)
        plt.show()

    # Scatter: PC1 vs Defect Ratio
    fig, ax = plt.subplots(figsize=(6,6))
    sc = ax.scatter(pc1[idxs], defect[idxs], c=G[idxs], cmap='tab10', s=25, alpha=0.7, picker=5)
    ax.set_xlabel('PC1'); ax.set_ylabel('Defect Ratio')
    ax.set_title(f"PC1 vs Defect Ratio (ρ={rho:.3f}, p={pval:.3f})")
    cbar = fig.colorbar(sc, ax=ax, label='Group')
    cbar.set_ticks(sorted(group_names.keys()))
    cbar.set_ticklabels([group_names[int(t)] for t in cbar.get_ticks()])

    # Interactive popup
    highlights = {}
    def on_pick(event):
        sel = event.ind[0]; idx = idxs[sel]
        # Toggle highlight
        if idx in highlights:
            highlights[idx].remove(); highlights.pop(idx)
        else:
            hl, = ax.plot(pc1[idx], defect[idx], marker='o', markersize=12,
                          markerfacecolor='none', markeredgecolor='red', linewidth=2)
            highlights[idx] = hl
        fig.canvas.draw_idle()

        # Popup display with float processing
        img_t, gid, pt = ds[idx]
        gray = img_t.squeeze().cpu().numpy().astype(np.float32)
        mi,ma = gray.min(),gray.max()
        if ma>1 or mi<0:
            gray = (gray-mi)/(ma-mi)
        bw,mask = defect_mask_float_gauss(gray)

        orig = gray
        binf = bw
        rgb  = np.stack([gray]*3,axis=-1)
        redm = np.stack([mask, np.zeros_like(mask), np.zeros_like(mask)],axis=-1)
        ov   = rgb*0.7 + redm*0.3

        imgs=[orig,binf,ov]
        tts=['Original','Binary(float)','Overlay(float)']
        win = plt.figure(figsize=(8,2))
        for i,(im,tt) in enumerate(zip(imgs,tts),1):
            axp = win.add_subplot(1,3,i)
            if im.ndim==2:
                axp.imshow(im,cmap='gray',vmin=0,vmax=1)
            else:
                axp.imshow(im,vmin=0,vmax=1)
            axp.set_title(f"{tt}\n{group_names[int(gid)]}/{os.path.basename(pt)}",fontsize=8)
            axp.axis('off')
        plt.tight_layout()

        # Ensure highlight removed when popup closed
        def on_close(evt, idx=idx):
            art = highlights.pop(idx, None)
            if art:
                art.remove()
                fig.canvas.draw_idle()
        win.canvas.mpl_connect('close_event', on_close)
        plt.show()

    fig.canvas.mpl_connect('pick_event', on_pick)
    plt.show()

if __name__=='__main__':
    extract_and_plot_defect_ratio()
