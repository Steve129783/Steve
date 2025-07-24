#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from c2_CNN_model import CachedImageDataset, CNN

# =========================
# 1. 参数配置
# =========================
file_name     = 'h_c_l'
cache_file    = rf'E:\Project_SNV\1N\1_Pack\{file_name}\data_cache.pt'
model_path    = rf'E:\Project_SNV\1N\c1_cache\{file_name}\best_model.pth'
ratio_csv     = rf'E:\Project_SNV\1N\c1_cache\{file_name}\defect_ratios.csv'
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
    info_path = os.path.join(os.path.dirname(cache_path), 'info.txt')
    with open(info_path, encoding='utf-8') as f:
        for line in f:
            m = p.match(line)
            if m:
                info[int(m.group(1))] = m.group(2)
    return info

group_names = load_group_names(cache_file)

# -------- MSER Mask 生成 --------
def defect_mask_mser(gray, min_area=5, max_area=150):
    bw = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        51, 5
    )
    mser = cv2.MSER_create()
    mser.setMinArea(min_area)
    mser.setMaxArea(max_area)
    regions, _ = mser.detectRegions(bw)
    mask = np.zeros_like(gray, dtype=np.uint8)
    for pts in regions:
        cv2.fillPoly(mask, [pts.reshape(-1,1,2)], 1)
    return bw, mask

# =========================
# 2. 主流程：绘制 PC1/PC2 vs 已有 defect_ratio，并加入交互
# =========================
def extract_and_plot_defect_ratio():
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # 载入 defect_ratios.csv
    df_ratio = pd.read_csv(ratio_csv, dtype={'path': str, 'defect_ratio': float})
    ratio_dict = dict(zip(df_ratio['path'], df_ratio['defect_ratio']))

    # 数据集与模型：return_path=True
    ds = CachedImageDataset(cache_file, transform=None, return_path=True)
    model = CNN(in_shape, len(group_names)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model.eval()

    loader = DataLoader(ds, batch_size=batch_size,
                        shuffle=False, num_workers=num_workers)

    all_h   = []
    all_def = []
    all_grp = []
    all_paths = []

    with torch.no_grad():
        for x, y, paths in loader:
            x = x.float().to(device)
            _, h = model(x)
            all_h.append(h.cpu().numpy())

            for p in paths:
                all_def.append(ratio_dict[p])
            all_grp.extend(y.cpu().numpy())
            all_paths.extend(paths)

    H      = np.vstack(all_h)
    defect = np.array(all_def)
    grp    = np.array(all_grp)

    # PCA 提取前两主成分
    pcs = PCA(n_components=2, random_state=seed).fit_transform(H)
    pc1, pc2 = pcs[:,0], pcs[:,1]

    # 限流：每组最多 max_per_group 个点
    idxs = []
    for g in np.unique(grp):
        gi = np.where(grp == g)[0]
        if len(gi) > max_per_group:
            gi = np.random.RandomState(seed).choice(gi, max_per_group, replace=False)
        idxs.extend(gi.tolist())
    idxs = np.array(idxs)

    # 计算相关系数
    r1, _ = pearsonr(pc1, defect)
    r2, _ = pearsonr(pc2, defect)

    # —— 图1：PC1 vs defect_ratio ——#
    fig1, ax1 = plt.subplots(figsize=(6,6))
    sc1 = ax1.scatter(pc1[idxs], defect[idxs],
                      c=grp[idxs], cmap='tab10',
                      s=20, alpha=0.7, picker=5)
    ax1.set_title(f"PC1 vs Defect Ratio  (r={r1:.3f})")
    ax1.set_xlabel("PC1"); ax1.set_ylabel("Defect Ratio")
    cbar1 = fig1.colorbar(sc1, ax=ax1, label="Group")
    cbar1.set_ticks(sorted(group_names.keys()))
    cbar1.set_ticklabels([group_names[int(t)] for t in cbar1.get_ticks()])

    # —— 图2：PC2 vs defect_ratio ——#
    fig2, ax2 = plt.subplots(figsize=(6,6))
    sc2 = ax2.scatter(pc2[idxs], defect[idxs],
                      c=grp[idxs], cmap='tab10',
                      s=20, alpha=0.7, picker=5)
    ax2.set_title(f"PC2 vs Defect Ratio  (r={r2:.3f})")
    ax2.set_xlabel("PC2"); ax2.set_ylabel("Defect Ratio")
    cbar2 = fig2.colorbar(sc2, ax=ax2, label="Group")
    cbar2.set_ticks(sorted(group_names.keys()))
    cbar2.set_ticklabels([group_names[int(t)] for t in cbar2.get_ticks()])

    # 交互高亮 & 弹窗
    highlights = {}
    def on_pick(event):
        ax = event.artist.axes
        sel_idx = idxs[event.ind[0]]

        # 高亮所选点
        if ax is ax1:
            x0, y0 = pc1[sel_idx], defect[sel_idx]
            fig, target_ax = fig1, ax1
        else:
            x0, y0 = pc2[sel_idx], defect[sel_idx]
            fig, target_ax = fig2, ax2

        hl, = target_ax.plot(x0, y0,
                             marker='o', markersize=12,
                             markerfacecolor='none',
                             markeredgecolor='red', linewidth=2)
        highlights[sel_idx] = hl
        fig.canvas.draw_idle()

        # 弹窗：四图展示
        gray, _, path = ds[sel_idx]
        if isinstance(gray, torch.Tensor):
            arr = gray.squeeze().cpu().numpy()
        else:
            arr = np.squeeze(gray)
        if arr.max() <= 1.0:
            arr = (arr * 255).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)

        bw, mask = defect_mask_mser(arr)
        img_bgr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        red     = np.zeros_like(img_bgr); red[:,:,2] = mask*255
        overlay = cv2.addWeighted(img_bgr, 0.7, red, 0.3, 0)
        overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

        titles = ['Original', 'Binary', 'Mask', 'Overlay']
        images = [arr, bw, mask*255, overlay]
        plt.figure(figsize=(8,2))
        for i,(im,t) in enumerate(zip(images,titles),1):
            axp = plt.subplot(1,4,i)
            cmap = 'gray' if im.ndim==2 else None
            plt.imshow(im, cmap=cmap)
            axp.set_title(f"{t}\n{os.path.basename(path)}", fontsize=8)
            plt.axis('off')
        plt.tight_layout()

        def on_close(evt):
            art = highlights.pop(sel_idx, None)
            if art:
                art.remove()
                fig.canvas.draw_idle()
        plt.gcf().canvas.mpl_connect('close_event', on_close)
        plt.show()

    fig1.canvas.mpl_connect('pick_event', on_pick)
    fig2.canvas.mpl_connect('pick_event', on_pick)

    plt.show()


if __name__ == "__main__":
    extract_and_plot_defect_ratio()
