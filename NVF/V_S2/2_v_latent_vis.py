#!/usr/bin/env python3
import os, sys, random
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA
from matplotlib.backend_bases import CloseEvent

# ───────────────────────── 项目路径 & 自定义模块 ─────────────────────────
ROOT = os.path.abspath(os.path.join(__file__, "..", ".."))
sys.path.insert(0, ROOT)
from v1_VAE_model import VAEWithClassifier, seed_everything, CachedImageDataset

# ───────────────────────── 路径 & 常量 ─────────────────────────
CACHE_FILE = r"E:\Project_CNN\2_Pack\data_cache.pt"
CHECKPOINT = r"E:\Project_CNN\v_cache\best_model.pth"
LATENT_DIM = 128
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 可复现
seed_everything(SEED)

# Dataset
ds = CachedImageDataset(CACHE_FILE)
loader = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=False, pin_memory=True)

# 加载模型
num_groups = len(torch.unique(ds.groups))
model = VAEWithClassifier(1, LATENT_DIM, num_groups).to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
model.eval()

# 提取 mu
all_mu, all_gid = [], []
with torch.no_grad():
    for img, gid in loader:
        mu, _ = model.encode_mu(img.to(DEVICE))
        all_mu.append(mu.cpu().numpy())
        all_gid.append(gid.numpy())
all_mu = np.concatenate(all_mu, axis=0)
all_gid = np.concatenate(all_gid, axis=0)

# PCA
pca = PCA(n_components=2, random_state=SEED)
mu_pca = pca.fit_transform(all_mu)

# 组名映射
def get_group_names(gids):
    DEFAULT = ["correct","high","low"]
    uniq = sorted(int(i) for i in np.unique(gids))
    return {g: DEFAULT[i] if i < len(DEFAULT) else f"group {g}" for i,g in enumerate(uniq)}
group_names = get_group_names(all_gid)
unique_ids = sorted(group_names.keys())

# 绘制
fig, ax = plt.subplots(figsize=(8,6))
points = []  # store PathCollection
for gid in unique_ids:
    mask = all_gid == gid
    sc = ax.scatter(mu_pca[mask,0], mu_pca[mask,1], label=group_names[gid], s=20, picker=True)
    sc._dataset_indices = np.where(mask)[0]
    points.append(sc)
ax.set(title="PCA of VAE Latent Means", xlabel="PC1", ylabel="PC2")
ax.legend(title="Group", loc='best')
plt.tight_layout()

# track selected highlights
highlight_map = {}  # popup_fig -> highlight artist

# callback for pick event
def on_pick(event):
    artist = event.artist
    ind = event.ind[0]
    # highlight the selected point
    x, y = artist.get_offsets()[ind]
    hl = ax.scatter([x], [y], s=100, facecolors='none', edgecolors='red', linewidths=2)
    fig.canvas.draw()
    # show popup without closing previous ones
    ds_idx = int(artist._dataset_indices[ind])
    img, gid = ds[ds_idx]
    img_np = img.squeeze().numpy()
    popup_fig, ax2 = plt.subplots(figsize=(3, 3))
    ax2.imshow(img_np, cmap='gray', vmin=0.0, vmax=1.0)
    ax2.axis('off')
    ax2.set_title(group_names[int(gid)])
    # store mapping
    highlight_map[popup_fig] = hl
    # when this popup closes, remove its highlight
    def on_close(event):
        hl_artist = highlight_map.pop(popup_fig, None)
        if hl_artist is not None:
            hl_artist.remove()
            fig.canvas.draw()
    popup_fig.canvas.mpl_connect('close_event', on_close)
    plt.show()

fig.canvas.mpl_connect('pick_event', on_pick)
plt.show()
plt.show()
