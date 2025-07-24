#!/usr/bin/env python3
import os
import re
import random
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from c2_CNN_model import CachedImageDataset, CNN
# =========================
# 1. 参数配置
# =========================
file_name     = 'h_c_l'
cache_file    = rf'E:\Project_SNV\1N\1_Pack\{file_name}\data_cache.pt'
model_path    = rf'E:\Project_SNV\1N\c1_cache\{file_name}\best_model.pth'
seed          = 42
batch_size    = 32
num_workers   = 0
in_shape      = (50, 50)
device        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
max_per_group = 1500

# —— 读取 gid→name 映射 ——#
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
# 4. 主流程：分别绘两图
# =========================
def extract_and_plot_all():
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    ds = CachedImageDataset(cache_file)
    model = CNN(in_shape, len(group_names)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model.eval()

    loader = DataLoader(ds, batch_size=batch_size,
                        shuffle=False, num_workers=num_workers)

    all_h, all_contrast, all_lap, all_g = [], [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            _, h = model(x)
            all_h.append(h.cpu().numpy())

            # Contrast
            flat  = x.view(x.size(0), -1)
            mins, maxs, means = flat.min(1).values, flat.max(1).values, flat.mean(1)
            contrast = ((means - mins) / (maxs - mins + 1e-6)).cpu().numpy()

            # Laplacian Variance
            imgs = (x.cpu().numpy() * 255).astype(np.uint8)
            lap_vals = []
            for im in imgs:
                gray = im.squeeze()
                lap = cv2.Laplacian(gray, cv2.CV_64F)
                lap_vals.append(lap.var())
            lap_vals = np.array(lap_vals)

            all_contrast.extend(contrast)
            all_lap.extend(lap_vals)
            all_g.extend(y.cpu().numpy())

    H        = np.vstack(all_h)
    contrast = np.array(all_contrast)
    lap_var  = np.array(all_lap)
    grp      = np.array(all_g)

    # PCA → PC1
    pc1 = PCA(n_components=1, random_state=seed).fit_transform(H).flatten()

    # 限流
    idxs = []
    for g in np.unique(grp):
        gi = np.where(grp==g)[0]
        if len(gi)>max_per_group:
            gi = np.random.RandomState(seed).choice(gi, max_per_group, replace=False)
        idxs.extend(gi.tolist())
    idxs = np.array(idxs)

    # 计算相关
    r1, _ = pearsonr(pc1, contrast)
    r2, _ = pearsonr(pc1, lap_var)

    # —— 第一张图：Contrast vs PC1 ——#
    fig1, ax1 = plt.subplots(figsize=(6,6))
    sc1 = ax1.scatter(pc1[idxs], contrast[idxs],
                      c=grp[idxs], cmap='tab10',
                      s=20, alpha=0.7, picker=5)
    ax1.set_title(f"PC1 vs Contrast  (r={r1:.3f})")
    ax1.set_xlabel("PC1"); ax1.set_ylabel("Contrast")
    cbar1 = fig1.colorbar(sc1, ax=ax1, label="Group")
    cbar1.set_ticks(sorted(group_names.keys()))
    cbar1.set_ticklabels([group_names[int(t)] for t in cbar1.get_ticks()])

    # —— 第二张图：Laplacian Var vs PC1 ——#
    fig2, ax2 = plt.subplots(figsize=(6,6))
    sc2 = ax2.scatter(pc1[idxs], lap_var[idxs],
                      c=grp[idxs], cmap='tab10',
                      s=20, alpha=0.7, picker=5)
    ax2.set_title(f"PC1 vs Laplacian Var  (r={r2:.3f})")
    ax2.set_xlabel("PC1"); ax2.set_ylabel("Laplacian Variance")
    cbar2 = fig2.colorbar(sc2, ax=ax2, label="Group")
    cbar2.set_ticks(sorted(group_names.keys()))
    cbar2.set_ticklabels([group_names[int(t)] for t in cbar2.get_ticks()])

    # 交互高亮 & 弹窗函数复用
    highlights = {}
    def on_pick(event):
        ax = event.artist.axes
        ind0 = event.ind[0]
        sel  = idxs[ind0]
        x0   = pc1[sel]
        y0   = contrast[sel] if ax is ax1 else lap_var[sel]

        hl, = ax.plot(x0, y0,
                      marker='o', markersize=12,
                      markerfacecolor='none',
                      markeredgecolor='red', linewidth=2)
        highlights[sel] = hl
        if ax is ax1:
            fig1.canvas.draw_idle()
        else:
            fig2.canvas.draw_idle()

        img_t, gid = ds[sel]
        img = img_t.squeeze().cpu().numpy()
        pf, pa = plt.subplots(figsize=(4,4))
        pa.imshow(img, cmap='gray', vmin=0, vmax=1)
        pa.axis('off')
        pa.set_title(f"{group_names[int(gid)]} — idx {sel}")

        def on_close(ev):
            art = highlights.pop(sel, None)
            if art:
                art.remove()
                if ax is ax1:
                    fig1.canvas.draw_idle()
                else:
                    fig2.canvas.draw_idle()

        pf.canvas.mpl_connect('close_event', on_close)
        plt.show()

    fig1.canvas.mpl_connect('pick_event', on_pick)
    fig2.canvas.mpl_connect('pick_event', on_pick)

    plt.show()

if __name__ == "__main__":
    extract_and_plot_all()
