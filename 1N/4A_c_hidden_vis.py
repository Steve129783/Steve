#!/usr/bin/env python3
import os, re, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
from torchvision import transforms
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from c2_CNN_model import CachedImageDataset, CNN
# =========================
# 1. 参数配置
# =========================
file_name     = 'h_c_l'
cache_file    = rf'E:\Project_SNV\1N\1_Pack\{file_name}\data_cache.pt'
model_path    = rf'E:\Project_SNV\1N\c1_cache\{file_name}\best_model.pth'
splits        = (0.7, 0.15, 0.15)
seed          = 42
batch_size    = 32
num_workers   = 0   # 保证确定性
in_shape      = (50, 50)
device        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
max_per_group = 1500  # 每个组别最多显示的点数

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
# 4. 主流程：合并三集合并绘图（限流1500/组）+ 交互高亮 + 打印前5个PC信息占比
# =========================
def extract_and_plot_all():
    # 固定随机种子
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # 1) 数据读取与划分
    ds = CachedImageDataset(cache_file, transform=None, return_path=True)
    N  = len(ds)
    n1 = int(splits[0]*N)
    n2 = int((splits[0]+splits[1])*N)
    train_ds, val_ds, test_ds = random_split(
        ds, [n1, n2-n1, N-n2],
        generator=torch.Generator().manual_seed(seed)
    )

    # 2) 加载模型
    model = CNN(in_shape, len(group_names)).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()

    # 3) 合并三个子集
    combined = ConcatDataset([train_ds, val_ds, test_ds])

    # 4) DataLoader 提取隐藏向量、标签与路径
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

    # 5) PCA 降到 2 维，并打印前5个PC的 explained variance ratio
    pca = PCA(n_components=5, random_state=seed).fit(H)
    ratios = pca.explained_variance_ratio_
    formatted = [round(r, 4) for r in ratios]
    print("Explained variance ratio of first 5 PCs:", formatted)
    H_pca = pca.transform(H)[:,:2]  # 只取前两维用于绘图

    # 6) 按组别限流
    idxs = []
    for grp in np.unique(G):
        grp_idx = np.where(G == grp)[0]
        if len(grp_idx) > max_per_group:
            grp_idx = np.random.RandomState(seed).choice(
                grp_idx, max_per_group, replace=False
            )
        idxs.extend(grp_idx.tolist())
    idxs = np.array(idxs)

    # 7) 绘图（PC1 vs PC2）
    fig, ax = plt.subplots(figsize=(8,7))
    sc = ax.scatter(
        H_pca[idxs,0], H_pca[idxs,1],
        c=G[idxs], cmap='tab10', s=30, picker=True, alpha=0.8
    )
    ax.set_title("PCA of Hidden Features (All Splits, max1500/grp)")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")

    # 8) 颜色图例：组别
    cbar = fig.colorbar(sc, ax=ax, label="Group")
    ticks = sorted(group_names.keys())
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([group_names[k] for k in ticks])

    # 9) 交互高亮 & 弹窗
    highlights = {}
    def on_pick(event):
        ind0 = event.ind[0]
        ind  = idxs[ind0]  # 原索引
        x0, y0 = H_pca[ind]
        hl, = ax.plot(x0, y0,
                      marker='o', markersize=12,
                      markerfacecolor='none',
                      markeredgecolor='red', linewidth=2)
        fig.canvas.draw_idle()

        # 弹窗展示图像
        img_tensor, gid, path = combined[ind]
        img = img_tensor.squeeze().numpy()
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
