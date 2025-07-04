# reconstruct_and_visualize.py
# 用于加载训练好的 VAEWithClassifier 权重，从缓存数据集中随机选五张图并进行重构、可视化比较

import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset

# 请确保以下模块能被导入，或者将本文件和 model.py 放在同一目录下
from v5_VAE_model import VAEWithClassifier, CachedImageDataset, seed_everything


def visualize_reconstructions(
    cache_file: str,
    checkpoint_path: str,
    num_samples: int = 5,
    seed: int = 42,
    device: str = None
):
    # 固定随机种子
    seed_everything(seed)

    # 设备
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载数据集
    dataset = CachedImageDataset(cache_file)
    total = len(dataset)

    # 随机选取索引
    indices = random.sample(range(total), num_samples)
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=num_samples, shuffle=False)

    # 加载模型
    num_groups = len(torch.unique(dataset.groups))
    model = VAEWithClassifier(img_channels=1, latent_dim=128, num_groups=num_groups)
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    model.to(device).eval()

    # 获取一批样本
    imgs, _ = next(iter(loader))  # imgs: [num_samples,1,50,50]
    imgs = imgs.to(device)

    # 重构
    with torch.no_grad():
        recon, mu, logvar, logits = model(imgs)

    recon = recon.cpu().numpy()
    origs = imgs.cpu().numpy()

    # 可视化
    fig, axes = plt.subplots(num_samples, 2, figsize=(4, 2*num_samples))
    for i in range(num_samples):
        # 原图
        axes[i, 0].imshow(origs[i,0], cmap='gray', vmin=0, vmax=1)
        axes[i, 0].set_title('Original')
        axes[i, 0].axis('off')
        # 重构
        axes[i, 1].imshow(recon[i,0], cmap='gray', vmin=0, vmax=1)
        axes[i, 1].set_title('Reconstruction')
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # 配置路径
    CACHE_FILE = r"E:/Project_CNN/2_Pack/data_cache.pt"
    CHECKPOINT = r"E:/Project_CNN/v_cache/best_model.pth"
    # 可视化重构
    visualize_reconstructions(
        cache_file=CACHE_FILE,
        checkpoint_path=CHECKPOINT,
        num_samples=5,
        seed=42
    )
