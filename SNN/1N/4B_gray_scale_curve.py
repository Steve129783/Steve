#!/usr/bin/env python3
import os
import re
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# =========================
# 1. 参数配置
# =========================
file_name   = '1_2_3_4'
cache_file  = rf'E:\Project_SNV\1N\1_Pack\{file_name}\data_cache.pt'
model_path  = rf'E:\Project_SNV\1N\c1_cache\{file_name}\best_model.pth'
seed        = 42
batch_size  = 32
num_workers = 0   # 确保可复现
in_shape    = (50, 50)
device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
max_per_group = 1500  # 每个组别最多显示的点数

# —— 从 cache_file 同目录读取 gid→name ——#
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
# 2. Dataset
# =========================
class CachedImageDataset(Dataset):
    def __init__(self, cache_path, transform=None):
        data = torch.load(cache_path, map_location='cpu')
        self.images    = data['images']           # [N,1,H,W]
        self.group_ids = torch.tensor(data['group_ids'], dtype=torch.long)
        self.transform = transform

    def __len__(self):
        return len(self.group_ids)

    def __getitem__(self, idx):
        x = self.images[idx]
        if self.transform:
            x = self.transform(x)
        y = self.group_ids[idx]
        return x, y

# =========================
# 3. 模型封装（与训练脚本一致）
# =========================
class CNNWithHidden(nn.Module):
    def __init__(self, in_shape, n_classes):
        super().__init__()
        c_list = [1,16,32,64]
        layers = []
        for i in range(2):
            layers += [
                nn.Conv2d(c_list[i], c_list[i+1], 3, padding='same'),
                nn.BatchNorm2d(c_list[i+1]), nn.ELU(),
                nn.Dropout2d(0.2), nn.MaxPool2d(2)
            ]
        layers.append(nn.Flatten())
        with torch.no_grad():
            dummy   = torch.zeros(1,1,*in_shape)
            flat_dim= nn.Sequential(*layers)(dummy).shape[1]
        layers += [
            nn.Linear(flat_dim, 128),
            nn.BatchNorm1d(128), nn.ELU(), nn.Dropout(0.5),
            nn.Linear(128, n_classes)
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        h = self.model[:-1](x)         # 倒数第二层特征
        logits = self.model[-1](h)     # 最后一层分类
        return logits, h

# =========================
# 4. 主流程：全数据 PC1 vs Brightness 并线性拟合 + 限流 + 交互高亮
# =========================
def extract_and_plot_all():
    # 固定随机种子
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # 数据加载
    ds = CachedImageDataset(cache_file)

    # 模型加载
    model = CNNWithHidden(in_shape, len(group_names)).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()

    # 提取所有样本的隐藏向量和归一化亮度
    loader = DataLoader(ds, batch_size=batch_size,
                        shuffle=False, num_workers=num_workers)
    all_h, all_bri, all_g = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            _, h = model(x)
            all_h.append(h.cpu().numpy())
            x0 = x
            # Flat and calculate min/max/mean
            flat  = x0.view(x0.size(0), -1)
            mins  = flat.min(dim=1).values
            maxs  = flat.max(dim=1).values
            means = flat.mean(dim=1)
            # Add eps 
            bri   = (means - mins) / (maxs - mins + 1e-6)
            all_bri.extend(bri.cpu().numpy())
            all_g.extend(y.numpy())

    H = np.vstack(all_h)
    grp = np.array(all_g)
    # global normalization
    # Use all_bri（Whole bri list）to normalization
    bri = np.array(all_bri)
    bri_norm = bri

    # PCA → PC1
    pc1 = PCA(n_components=1, random_state=seed).fit_transform(H).flatten()
    pc1_min, pc1_max = pc1.min(), pc1.max()
    pc1 = (pc1 - pc1_min) / (pc1_max - pc1_min)

    # 每个组别限流
    idxs = []
    for g in np.unique(grp):
        gi = np.where(grp == g)[0]
        if len(gi) > max_per_group:
            gi = np.random.RandomState(seed).choice(gi, max_per_group, replace=False)
        idxs.extend(gi.tolist())
    idxs = np.array(idxs)

    # 绘制散点 + 拟合直线
    fig, ax = plt.subplots(figsize=(6,6))
    sc = ax.scatter(pc1[idxs], bri_norm[idxs],
                    c=grp[idxs], cmap='tab10', s=20, picker=5, alpha=0.8)
    a, b = np.polyfit(pc1[idxs], bri_norm[idxs], 1)
    xs = np.linspace(pc1[idxs].min(), pc1[idxs].max(), 200)
    ax.plot(xs, a*xs + b, '-', color='black', lw=2, label=f'y={a:.3f}x+{b:.3f}')
    ax.legend()

    cbar = plt.colorbar(sc, ax=ax, label="Group")
    ticks = sorted(group_names.keys())
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([group_names[t] for t in ticks])

    ax.set_title("PC1 vs Normalized Contrast (Linear Fit) — All Data")
    ax.set_xlabel("PC1")
    ax.set_ylabel("Normalized Contrast")

    # 交互高亮
    highlights = {}
    def on_pick(evt):
        ind0 = evt.ind[0]
        ind = idxs[ind0]
        x0, y0 = pc1[ind], bri_norm[ind]
        hl, = ax.plot(x0, y0,
                      marker='o', markersize=12,
                      markerfacecolor='none',
                      markeredgecolor='red', lw=2)
        fig.canvas.draw_idle()

        img, gid = ds[ind]
        img = img.squeeze().numpy()
        pf, pa = plt.subplots(figsize=(4,4))
        pa.imshow(img, cmap='gray', vmin=0, vmax=1)
        pa.axis('off')
        pa.set_title(f"{group_names[int(gid)]} — idx {ind}")
        highlights[pf] = hl

        def on_close(ev):
            art = highlights.pop(pf, None)
            if art:
                art.remove()
                fig.canvas.draw_idle()
        pf.canvas.mpl_connect('close_event', on_close)
        plt.show()

    fig.canvas.mpl_connect('pick_event', on_pick)
    plt.show()

if __name__ == "__main__":
    extract_and_plot_all()
