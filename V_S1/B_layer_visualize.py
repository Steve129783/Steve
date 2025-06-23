import torch, os
import matplotlib.pyplot as plt
from pathlib import Path
from torchvision.utils import make_grid
from o3_VAE_model import CachedImageDataset, VAE_ResSkip_50

# —— 带 colorbar 的 show_heatmap 定义，只保留这一版 —— 
def show_heatmap(feature_tensor, title, nrow=8, figsize=(10,4)):
    # feature_tensor: [1, C, H, W]
    feat = feature_tensor.squeeze(0).unsqueeze(1)   # [C,1,H,W]
    grid = make_grid(feat, nrow=nrow, normalize=True, scale_each=True).cpu()
    # 取第0通道或 squeeze
    heat = grid[0].numpy() if grid.shape[0] > 1 else grid.squeeze(0).numpy()

    # 用 subplots 获取 fig 和 ax，这样才能给 fig 加 colorbar
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(heat, cmap="hot", vmin=heat.min(), vmax=heat.max())
    ax.axis("off")
    ax.set_title(title)

    # 在右侧加一个竖状 colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8)

# 1) 设置设备和路径
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_CACHE = Path(r"E:\Project_VAE\V1\Ori_Pack\1_2_3_4\BGMBDF\data_cache.pt")
MODEL_PATH = Path(r"E:\Project_VAE\V1\oVAE_cache\1_2_3_4\best_model.pth")

# 2) 加载模型
latent_dim = 24
model = VAE_ResSkip_50(img_channels=1, latent_dim=latent_dim).to(device)
ckpt  = torch.load(MODEL_PATH, map_location=device)
state = ckpt.get("model", ckpt)
model.load_state_dict(state, strict=False)
model.eval()

# 3) 加载 cache，看 paths 并按名称查样本
cache = torch.load(DATA_CACHE)
paths = cache["paths"]
target_name = "crop_Componenets-2.transformed172_100_250.png"

try:
    sample_idx = next(
        i for i, path in enumerate(paths)
        if os.path.basename(path) == target_name
    )
except StopIteration:
    raise ValueError(f"在 cache['paths'] 中找不到名称为 '{target_name}' 的文件")

ds = CachedImageDataset(cache_file=str(DATA_CACHE))
img, meta = ds[sample_idx]
print(f"Found sample #{sample_idx}, meta: {meta}")

# 4) 送模型
img = img.unsqueeze(0).to(device)  # shape (1,1,50,50)

# 5) 前向提取两层输出
with torch.no_grad():
    x1 = model.enc_conv1(img)  # [1,32,25,25]
    x2 = model.enc_conv2(x1)   # [1,64,12,12]

# 6) 展示热力图（会自动带上 colorbar）
show_heatmap(x1, f"enc_conv1 (32×25×25) heatmap for '{target_name}'", nrow=8, figsize=(8,4))
show_heatmap(x2, f"enc_conv2 (64×12×12) heatmap for '{target_name}'", nrow=8, figsize=(8,4))
plt.show()
