#!/usr/bin/env python3
import os
import re
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
from torchvision import transforms
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# =========================
# 1. 参数配置
# =========================
file_name     = '1_2_3_4'
cache_file    = rf'E:\Project_SNV\1N\1_Pack\{file_name}\data_cache.pt'
model_path    = rf'E:\Project_SNV\2N\c2_cache\{file_name}\best_model.pth'
splits        = (0.7, 0.15, 0.15)
seed          = 42
batch_size    = 32
num_workers   = 0   # 保证可复现
in_shape      = (50, 50)
device        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
max_per_group = 1500  # 每个组别最多保留的点数

# ——— 在这里指定要保留的第一层通道编号 ———
desired_channels = [15]

# —— 读取 gid→name 映射 ——#
def load_group_names(cache_path):
    info_path = os.path.join(os.path.dirname(cache_path), 'info.txt')
    pattern   = re.compile(r'^\s*(\d+)\s*:\s*([^\s(]+)')
    mapping   = {}
    with open(info_path, 'r', encoding='utf-8') as f:
        for line in f:
            m = pattern.match(line)
            if m:
                mapping[int(m.group(1))] = m.group(2)
    return mapping

group_names = load_group_names(cache_file)

# =========================
# 2. Dataset
# =========================
class CachedImageDataset(Dataset):
    def __init__(self, cache_path, transform=None):
        data = torch.load(cache_path, map_location='cpu')
        self.images    = data['images']
        self.group_ids = torch.tensor(data['group_ids'], dtype=torch.long)
        self.paths     = data.get('paths', [None]*len(self.images))
        self.transform = transform

    def __len__(self):
        return len(self.group_ids)

    def __getitem__(self, idx):
        x = self.images[idx]
        if self.transform:
            x = self.transform(x)
        y = self.group_ids[idx]
        p = self.paths[idx]
        return x, y, p

# =========================
# 3. CNNWithHidden 结构
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
        h      = self.model[:-1](x)
        logits = self.model[-1](h)
        return logits, h

# =========================
# 4. 只屏蔽并冻结第一层
# =========================
def apply_mask_and_freeze_conv1(model, desired):
    conv1, bn1 = model.model[0], model.model[1]
    out_ch     = conv1.out_channels
    mask = torch.zeros((out_ch,1,1,1), device=conv1.weight.device)
    mask[desired] = 1.0
    with torch.no_grad():
        conv1.weight.mul_(mask)
        if conv1.bias is not None:
            conv1.bias.mul_(mask.view(-1))
    # 冻结 conv1 + bn1
    for p in conv1.parameters(): p.requires_grad = False
    for p in bn1.parameters():   p.requires_grad = False
    bn1.eval()

# =========================
# 5. 主流程：合并三集合并绘图（限流+交互）
# =========================
def extract_and_plot_all():
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    ds = CachedImageDataset(
        cache_file,
        transform=None
    )
    N  = len(ds)
    n1 = int(splits[0]*N)
    n2 = int((splits[0]+splits[1])*N)
    train_ds, val_ds, test_ds = random_split(
        ds, [n1, n2-n1, N-n2],
        generator=torch.Generator().manual_seed(seed)
    )

    model = CNNWithHidden(in_shape, len(group_names)).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state, strict=False)
    apply_mask_and_freeze_conv1(model, desired_channels)
    model.eval()

    combined = ConcatDataset([train_ds, val_ds, test_ds])
    loader = DataLoader(combined, batch_size=batch_size,
                        shuffle=False, num_workers=num_workers)
    all_h, all_g, all_paths = [], [], []
    with torch.no_grad():
        for x, y, p in loader:
            _, h = model(x.to(device))
            all_h.append(h.cpu().numpy())
            all_g.extend(y.numpy())
            all_paths.extend(p)
    H = np.vstack(all_h)
    G = np.array(all_g)

    H_pca = PCA(n_components=2, random_state=seed).fit_transform(H)
    
    # 每组限流
    idxs = []
    for grp in np.unique(G):
        gi = np.where(G==grp)[0]
        if len(gi) > max_per_group:
            gi = np.random.RandomState(seed).choice(gi,
                  max_per_group, replace=False)
        idxs.extend(gi.tolist())
    idxs = np.array(idxs)

    fig, ax = plt.subplots(figsize=(8,7))
    sc = ax.scatter(
        H_pca[idxs,0], H_pca[idxs,1],
        c=G[idxs], cmap='tab10', s=30,
        picker=True, alpha=0.8
    )
    ax.set_title("PCA of Hidden Features (All Splits)")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")

    cbar = fig.colorbar(sc, ax=ax, label="Group")
    ticks = sorted(group_names.keys())
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([group_names[k] for k in ticks])

    highlights = {}
    def on_pick(event):
        ind0 = event.ind[0]
        ind  = idxs[ind0]
        x0, y0 = H_pca[ind]
        hl, = ax.plot(x0, y0,
                      marker='o', markersize=12,
                      markerfacecolor='none',
                      markeredgecolor='red', linewidth=2)
        fig.canvas.draw_idle()

        img_t, gid, path = combined[ind]
        img = (img_t * 0.5 + 0.5).clamp(0,1).squeeze().numpy()
        pf, pa = plt.subplots(figsize=(4,4))
        pa.imshow(img, cmap='gray', vmin=0, vmax=1)
        pa.axis('off')
        pa.set_title(f"{os.path.basename(path)}\nGroup {group_names[int(gid)]}")
        highlights[pf] = hl

        def on_close(evt):
            art = highlights.pop(pf, None)
            if art:
                art.remove()
                fig.canvas.draw_idle()
        pf.canvas.mpl_connect('close_event', on_close)
        plt.show()

    fig.canvas.mpl_connect('pick_event', on_pick)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    extract_and_plot_all()
