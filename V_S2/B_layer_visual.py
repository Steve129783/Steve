import os
import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import Dataset

# ─── 1) Dataset：读取 images + meta ───
class CacheWithMetaDataset(Dataset):
    def __init__(self, cache_file):
        cache = torch.load(cache_file, map_location="cpu")
        self.images = cache["images"]   # Tensor[N,1,50,50]
        self.meta   = cache["meta"]     # list of dict
        assert len(self.images) == len(self.meta)
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        return self.images[idx], self.meta[idx]

# ─── 2) CVAE Encoder 前两层 ───
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.gn1   = nn.GroupNorm(max(1, channels//8), channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.gn2   = nn.GroupNorm(max(1, channels//8), channels)
        self.relu  = nn.ReLU(True)
        self.drop  = nn.Dropout2d(0.1)
    def forward(self, x):
        out = self.relu(self.gn1(self.drop(self.conv1(x))))
        out = self.gn2(self.conv2(out))
        return self.relu(out + x)

class CVAE_ResSkip_50(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),
            nn.GroupNorm(4,32), nn.ReLU(True), ResBlock(32)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.GroupNorm(8,64), nn.ReLU(True), ResBlock(64)
        )
        # decode 层及其它不需要

    def encode(self, x, y=None):
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        return x1, x2

# ─── 3) 配置 & 加载 ───
device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CACHE_PT    = r"E:\Project_VAE\V2\Pack_json_png\Ori_drop\data_cache.pt"
MODEL_PATH  = r"E:\Project_VAE\V2\cvae_cache\best_model.pth"

# 按 “base_row_col” 这种格式索引
TARGET_NAME = "crop_Componenets_1.transformed060_5_4"

# 载入 dataset
ds = CacheWithMetaDataset(CACHE_PT)

# 在 meta 中找到匹配的那个索引
sample_idx = next(
    i for i, m in enumerate(ds.meta)
    if f"{m['base']}_{m['patch_row']}_{m['patch_col']}" == TARGET_NAME
)
print("定位到 sample_idx =", sample_idx, "meta =", ds.meta[sample_idx])

# 取出对应图像
img, _ = ds[sample_idx]
img = img.unsqueeze(0).to(device)  # [1,1,50,50]

# 加载模型，只用到 enc1/enc2
model = CVAE_ResSkip_50().to(device)
ckpt  = torch.load(MODEL_PATH, map_location=device)
rd    = ckpt.get("model", ckpt)
fixed = {
    k.replace("skip_conv_mid","skip_mid")
     .replace("skip_conv_low","skip_low"): v
    for k,v in rd.items()
}
model.load_state_dict(fixed, strict=False)
model.eval()

# ─── 4) 前向提取 enc1/enc2 ───
with torch.no_grad():
    x1, x2 = model.encode(img)

# ─── 5) 可视化热力图 ───
def plot_feature_maps(feat, title, ncol=8):
    feat = feat.squeeze(0)      # [C,H,W]
    C, H, W = feat.shape
    nrow    = math.ceil(C / ncol)
    fig, axs = plt.subplots(nrow, ncol,
                            figsize=(ncol*2, nrow*2),
                            squeeze=False,
                            constrained_layout=True)
    vmin, vmax = float(feat.min()), float(feat.max())
    for idx in range(C):
        r, c = divmod(idx, ncol)
        ax = axs[r][c]
        ax.imshow(feat[idx].cpu(), cmap="hot", vmin=vmin, vmax=vmax)
        ax.set_title(f"ch{idx}", fontsize=6)
        ax.axis("off")
    # 隐藏多余子图
    for idx in range(C, nrow*ncol):
        r, c = divmod(idx, ncol)
        axs[r][c].axis("off")
    plt.suptitle(title)
    plt.show()

plot_feature_maps(x1, f"enc1 — 32 通道 ({TARGET_NAME})")
plot_feature_maps(x2, f"enc2 — 64 通道 ({TARGET_NAME})")
