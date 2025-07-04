# visualize_encoder_features.py

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

# —— 开启交互模式 —— #
plt.ion()

# ——— 修改这两行即可 ———
LAYER    = '1'
IMG_PATH = r"E:\Project_CNN\0_image\low\007_x0_y100.png"
# ————————————————————

# 将项目根目录加入搜索路径
ROOT = os.path.abspath(os.path.join(__file__, "..", ".."))
sys.path.insert(0, ROOT)

from v5_VAE_model import VAEWithClassifier

CACHE_PATH = r"E:\Project_CNN\2_Pack\data_cache.pt"
CKPT_PATH  = r"E:\Project_CNN\v_cache\best_model.pth"
LATENT_DIM = 128
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IN_SHAPE   = (50, 50)

def show_image(img_np, title="Image", cmap="gray"):
    fig = plt.figure(figsize=(3,3))
    plt.imshow(img_np, cmap=cmap, vmin=0, vmax=1)
    plt.title(title)
    plt.axis("off")
    plt.show(block=False)  # 非阻塞模式

def visualize_feature_maps(feat, layer_name):
    C = feat.shape[1]
    cols = int(np.ceil(np.sqrt(C)))
    rows = int(np.ceil(C/cols))

    # ——— 方法 A：每个子图标题 ———
    fig = plt.figure(figsize=(cols*1.2, rows*1.2))
    for c in range(C):
        ax = fig.add_subplot(rows, cols, c+1)
        ax.imshow(feat[0,c].cpu(), cmap="gray", aspect="auto")
        ax.axis("off")
        ax.set_title(f"Ch {c}", fontsize=8)  # ← 这里加上通道编号
    fig.suptitle(f"{layer_name} — Grayscale", fontsize=14)
    plt.tight_layout()
    plt.show(block=False)

    # ——— 方法 B：图内角落标签 + 热力图版 ———
    fig2, axes = plt.subplots(rows, cols, figsize=(cols*1.2, rows*1.2))
    axes = axes.flatten()
    for c in range(C):
        ax = axes[c]
        im = ax.imshow(feat[0,c].cpu(), cmap="jet", aspect="auto")
        ax.axis("off")
        # 在右上角写编号，不占 suptitle
        ax.text(
            0.95, 0.05, f"{c}", 
            transform=ax.transAxes, 
            color='white', fontsize=6,
            ha='right', va='bottom',
            bbox=dict(boxstyle="round,pad=0.2", fc="black", ec="none", alpha=0.5)
        )
    # 多余子图隐藏
    for j in range(C, len(axes)):
        axes[j].axis("off")
    fig2.suptitle(f"{layer_name} — Heatmap", fontsize=14)
    cax = fig2.add_axes([0.92,0.15,0.02,0.7])
    mappable = plt.cm.ScalarMappable(cmap="jet")
    mappable.set_array(feat.cpu().numpy())
    fig2.colorbar(mappable, cax=cax)
    plt.tight_layout(rect=[0,0,0.9,1])
    plt.show(block=False)


def main():
    assert LAYER in ('1','2','3','all')
    assert os.path.isfile(IMG_PATH)

    # 1) 原图
    img_pil = Image.open(IMG_PATH).convert('L')
    img_pil = img_pil.resize(IN_SHAPE)
    show_image(np.array(img_pil)/255.0, title="Input Image")

    # 2) Tensor
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    img_tensor = tfm(img_pil).unsqueeze(0).to(DEVICE)

    # 3) 模型
    num_groups = len(set(torch.load(CACHE_PATH, map_location='cpu')["group_ids"]))
    model = VAEWithClassifier(1, LATENT_DIM, num_groups).to(DEVICE)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
    model.eval()

    # 4) 编码
    with torch.no_grad():
        f1 = model.enc1(img_tensor)
        f2 = model.enc2(f1)
        f3 = model.enc3(f2)

    # 5) 可视化
    if LAYER in ('1','all'):
        visualize_feature_maps(f1, 'Encoder Layer 1')
    if LAYER in ('2','all'):
        visualize_feature_maps(f2, 'Encoder Layer 2')
    if LAYER in ('3','all'):
        visualize_feature_maps(f3, 'Encoder Layer 3')

    # 等待用户操作后退出
    input("按回车后退出并关闭所有窗口…")

if __name__ == "__main__":
    main()
