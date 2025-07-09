import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# =========================
# 1. 参数配置
# =========================
cache_file   = r'E:\Project_CNN\2_Pack\data_cache.pt'
model_path   = r'E:\Project_CNN\c_cache\best_model.pth'
splits       = (0.7, 0.15, 0.15)
seed         = 42
batch_size   = 32
num_workers  = 4
in_shape     = (50, 50)
device       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

group_names = {0: "correct", 1: "high", 2: "low"}

# =========================
# 2. Dataset
# =========================
class CachedImageDataset(Dataset):
    def __init__(self, cache_path, transform=None):
        cache = torch.load(cache_path, map_location='cpu')
        self.images    = cache['images']
        self.group_ids = torch.tensor(cache['group_ids'], dtype=torch.long)
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
# 3. CNNWithHidden
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
            dummy = torch.zeros(1,1,*in_shape)
            flat_dim = nn.Sequential(*layers)(dummy).shape[1]
        layers += [
            nn.Linear(flat_dim, 128),
            nn.BatchNorm1d(128), nn.ELU(), nn.Dropout(0.5),
            nn.Linear(128, n_classes)
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        h = self.model[:-1](x)
        logits = self.model[-1](h)
        return logits, h

# =========================
# 4. 提取 & 可视化 PCA
# =========================
def extract_and_plot():
    # 保证可复现
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # —— 数据加载 & 划分 —— #
    transform = transforms.Normalize((0.5,), (0.5,))
    ds = CachedImageDataset(cache_file, transform=transform)
    N = len(ds)
    n1 = int(splits[0] * N)
    n2 = int((splits[0] + splits[1]) * N)
    _, _, test_ds = random_split(
        ds, [n1, n2 - n1, N - n2],
        generator=torch.Generator().manual_seed(seed)
    )
    loader = DataLoader(test_ds, batch_size=batch_size,
                        shuffle=False, num_workers=num_workers)

    # —— 模型加载 —— #
    num_classes = len(set(ds.group_ids.tolist()))
    model = CNNWithHidden(in_shape, num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # —— 提取隐藏向量 —— #
    all_h, all_g = [], []
    with torch.no_grad():
        for x, y in loader:
            _, h = model(x.to(device))
            all_h.append(h.cpu().numpy())
            all_g.extend(y.numpy())
    all_h = np.vstack(all_h)
    all_g = np.array(all_g)
    M = all_h.shape[0]

    # —— PCA 降到 2 维 —— #
    h_pca = PCA(n_components=2, random_state=seed).fit_transform(all_h)

    # —— 绘制散点并保存原色 —— #
    plt.ion()  # 开启交互模式
    fig, ax = plt.subplots(figsize=(6,6))
    sc = ax.scatter(
        h_pca[:,0], h_pca[:,1],
        c=all_g, cmap='tab10', s=20, picker=5
    )
    # 确保 facecolors 是 (M,4)
    raw = sc.get_facecolors()
    if raw.shape[0] == 1:
        facecols = np.repeat(raw, M, axis=0)
    else:
        facecols = raw.copy()
    sc.set_facecolors(facecols)
    orig_colors = facecols.copy()

    ax.set_title("PCA of CNN Hidden Features")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    cbar = plt.colorbar(sc, ax=ax, label="Group")
    ticks = list(group_names.keys())
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([group_names[t] for t in ticks])

    # —— 点击事件回调 —— #
    def on_pick(event):
        ind = event.ind[0]

        # 1) 把点标红
        cols = sc.get_facecolors()
        cols[ind] = np.array([1, 0, 0, 1])
        sc.set_facecolors(cols)
        fig.canvas.draw_idle()

        # 2) 阻塞式弹窗显示原图
        img_tensor, gid = test_ds[ind]
        img = (img_tensor * 0.5 + 0.5).clamp(0,1).squeeze().numpy()
        plt.figure(figsize=(4,4))
        plt.imshow(img, cmap='gray', vmin=0, vmax=1)
        plt.title(f"{group_names[int(gid)]} — idx {ind}")
        plt.axis('off')
        plt.show(block=True)  # 等待窗口关闭后再继续

        # 3) 恢复主图中点的原始颜色
        cols[ind] = orig_colors[ind]
        sc.set_facecolors(cols)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('pick_event', on_pick)
    plt.show(block=True)  # 主图也阻塞显示

if __name__ == "__main__":
    extract_and_plot()
