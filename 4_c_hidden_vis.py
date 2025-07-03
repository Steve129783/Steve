import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# =========================
# 1. 参数配置
# =========================
cache_file  = r'E:\Project_CNN\2_Pack\data_cache.pt'
splits      = (0.7, 0.15, 0.15)
seed        = 42
batch_size  = 32
num_workers = 4
in_shape    = (50, 50)
device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 类别名称映射
group_names = {0: "correct", 1: "high", 2: "low"}

# =========================
# 2. Dataset（不变）
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
# 3. 修改后的 CNN：返回 logits 和隐藏向量 h
# =========================
class CNNWithHidden(nn.Module):
    def __init__(self, in_shape, n_classes):
        super().__init__()
        c_list = [1,16,32,64]
        layers = []
        # 两个 conv + pool
        for i in range(2):
            layers += [
                nn.Conv2d(c_list[i], c_list[i+1], kernel_size=3, padding='same'),
                nn.BatchNorm2d(c_list[i+1]),
                nn.ELU(),
                nn.Dropout2d(0.2),
                nn.MaxPool2d(2)
            ]
        layers.append(nn.Flatten())
        # 计算 flatten 后维度
        with torch.no_grad():
            dummy = torch.zeros(1,1,*in_shape)
            flat_dim = nn.Sequential(*layers)(dummy).shape[1]
        # 隐藏层 128
        layers += [
            nn.Linear(flat_dim, 128),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.Dropout(0.5)
        ]
        # 输出层
        layers += [nn.Linear(128, n_classes)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        out = x
        # 前 6 层到 Flatten 之后（conv/pool/flatten）
        for i in range(6):
            out = self.model[i](out)
        h = out  # 128 维隐藏向量
        logits = self.model[6](h)
        return logits, h

# =========================
# 4. 批量提取隐藏向量并 PCA 降 2D（交互式）
# =========================
def extract_and_plot_interactive():
    # 数据加载与划分
    transform = transforms.Normalize((0.5,), (0.5,))
    ds = CachedImageDataset(cache_file, transform=transform)
    N = len(ds)
    n1 = int(splits[0]*N); n2 = int((splits[0]+splits[1])*N)
    _, _, test_ds = random_split(ds, [n1, n2-n1, N-n2],
                                generator=torch.Generator().manual_seed(seed))
    loader = DataLoader(test_ds, batch_size=batch_size,
                        shuffle=False, num_workers=num_workers)

    # 模型加载
    num_classes = len(set(ds.group_ids.tolist()))
    model = CNNWithHidden(in_shape, num_classes).to(device)
    model.load_state_dict(torch.load(os.path.join(r'E:\Project_CNN\c_cache','best_model.pth'),
                                     map_location=device))
    model.eval()

    # 提取隐藏向量
    all_h, all_g = [], []
    with torch.no_grad():
        for x, y in loader:
            _, h = model(x.to(device))
            h = h.view(h.size(0), -1)
            all_h.append(h.cpu().numpy())
            all_g.extend(y.numpy())
    all_h = np.vstack(all_h)  # [N,128]

    # PCA 降到 2D
    h2d = PCA(n_components=2).fit_transform(all_h)

    # 绘制交互式散点图
    fig, ax = plt.subplots(figsize=(6,6))
    scatter = ax.scatter(
        h2d[:,0], h2d[:,1],
        c=all_g, cmap='tab10', s=20, picker=True
    )
    ax.set_title("Interactive PCA of CNN Hidden Features")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")

    # 用名称替换 colorbar 刻度
    cbar = plt.colorbar(scatter, label="Group")
    ticks = list(group_names.keys())
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([group_names[t] for t in ticks])

    plt.tight_layout()

    # 点击事件回调
    def on_pick(event):
        ind = event.ind[0]
        img_tensor, gid = test_ds[ind]
        # 反归一化
        img = img_tensor * 0.5 + 0.5
        img = img.clamp(0,1).squeeze().numpy()
        # 弹出原图
        fig2, ax2 = plt.subplots(figsize=(4,4))
        ax2.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax2.set_title(f"{group_names[int(gid)]} — Index {ind}")
        ax2.axis('off')
        plt.show()

    fig.canvas.mpl_connect('pick_event', on_pick)
    plt.show()

if __name__ == "__main__":
    extract_and_plot_interactive()
