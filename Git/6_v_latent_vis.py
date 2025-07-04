"""
visualize_latent_pca.py
在 improved 训练配置下，对整个数据集的 μ 做 PCA 并可点击查看原图
"""
import os, sys, random
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA

# ───────────────────────── 项目路径 & 自定义模块 ─────────────────────────
ROOT = os.path.abspath(os.path.join(__file__, "..", ".."))
sys.path.insert(0, ROOT)                    # 让 Python 找到自定义代码
from v5_VAE_model import VAEWithClassifier  # 如果文件名不同，请对应修改
from v5_VAE_model import seed_everything, CachedImageDataset  # ← 直接复用

# ───────────────────────── 路径 & 常量 ─────────────────────────
CACHE_FILE   = r"E:\Project_CNN\2_Pack\data_cache.pt"
CHECKPOINT   = r"E:\Project_CNN\v_cache\best_model.pth"
LATENT_DIM   = 128
SEED         = 42
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ───────────────────────── 可复现性 ─────────────────────────
seed_everything(SEED)

# ───────────────────────── Dataset / Dataloader ─────────────────────────
ds      = CachedImageDataset(CACHE_FILE)          # 与训练时完全一致
loader  = torch.utils.data.DataLoader(ds,
             batch_size=256, shuffle=False, pin_memory=True)

# ───────────────────────── 载入模型 ─────────────────────────
num_groups = len(torch.unique(ds.groups))
model = VAEWithClassifier(1, LATENT_DIM, num_groups).to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
model.eval()

# ───────────────────────── 提取 μ 并 PCA ─────────────────────────
all_mu, all_gid = [], []
with torch.no_grad():
    for img, gid in loader:
        mu, _logvar = model.encode(img.to(DEVICE))
        all_mu.append(mu.cpu().numpy())
        all_gid.append(gid.numpy())

all_mu  = np.concatenate(all_mu, axis=0)
all_gid = np.concatenate(all_gid, axis=0)

pca = PCA(n_components=2, random_state=SEED)
mu_pca = pca.fit_transform(all_mu)

# ───────────────────────── 组名映射（示例） ─────────────────────────
# 如果一共有 3 组且顺序固定，可手动写；否则动态产生更安全
DEFAULT_NAMES = ["group-0", "group-1", "group-2"]
unique_ids    = sorted(int(i) for i in np.unique(all_gid))
group_names   = {gid: DEFAULT_NAMES[i] if i < len(DEFAULT_NAMES) else f"group {gid}"
                 for i, gid in enumerate(unique_ids)}

# ───────────────────────── 绘制 & 点击回溯 ─────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
for gid in unique_ids:
    mask = (all_gid == gid)
    ax.scatter(mu_pca[mask, 0], mu_pca[mask, 1],
               label=group_names[gid], s=20, picker=True)

ax.set(title="PCA of VAE Latent Means", xlabel="PC 1", ylabel="PC 2")
ax.legend(title="Group", loc="best")
plt.tight_layout()

# —— 点击后显示对应原图 —— #
def on_pick(event):
    idx = event.ind[0]                 # 拿到被选中的第一个样本索引
    img, gid = ds[idx]                 # Dataset 里拿原像素 [0-1]
    img_np   = img.squeeze().numpy()   # [1,50,50] → [50,50]

    fig2, ax2 = plt.subplots(figsize=(3, 3))
    ax2.imshow(img_np, cmap="gray"); ax2.axis("off")
    ax2.set_title(group_names[int(gid)])
    plt.show()

fig.canvas.mpl_connect("pick_event", on_pick)
plt.show()
