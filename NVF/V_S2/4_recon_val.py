# reconstruct_and_visualize.py
# 用于加载训练好的 VAEWithClassifier 权重，从缓存数据集中随机选五张图并进行重构、可视化比较，并计算 MSE/PSNR/SSIM

import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# 请确保以下模块能被导入，或者将本文件和 model.py 放在同一目录下
from v1_VAE_model import VAEWithClassifier, CachedImageDataset, seed_everything


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

    # 转回 CPU 及 numpy
    origs = imgs.cpu().numpy()      # shape (N,1,50,50)
    recon = recon.cpu().numpy()     # shape (N,1,50,50)

    # 计算指标
    mse_vals = []
    psnr_vals = []
    ssim_vals = []
    for i in range(num_samples):
        orig = origs[i, 0]
        pred = recon[i, 0]
        mse = np.mean((orig - pred) ** 2)
        psnr = peak_signal_noise_ratio(orig, pred, data_range=1.0)
        ssim = structural_similarity(orig, pred, data_range=1.0)
        mse_vals.append(mse)
        psnr_vals.append(psnr)
        ssim_vals.append(ssim)

    avg_full_mse = float(np.mean(mse_vals))
    avg_psnr     = float(np.mean(psnr_vals))
    avg_ssim     = float(np.mean(ssim_vals))

    # 打印平均指标
    print(f"整张 50×50 平均每像素 MSE: {avg_full_mse:.6f}")
    print(f"整张 50×50 平均 PSNR: {avg_psnr:.2f} dB")
    print(f"整张 50×50 平均 SSIM: {avg_ssim:.4f}")

    # 可视化：2 行 × num_samples 列，高 DPI + 最近邻插值
    fig, axes = plt.subplots(
        2, num_samples,
        figsize=(2.5 * num_samples, 6),  # 每列宽 2.5 英寸，高度 6 英寸
        dpi=150                         # 提高清晰度
    )

    for i in range(num_samples):
        # 第一行：原图
        axes[0, i].imshow(
            origs[i, 0],
            cmap='gray',
            vmin=0, vmax=1,
            interpolation='nearest'
        )
        if i == 0:
            axes[0, i].set_title('Original', fontsize=12)
        axes[0, i].axis('off')

        # 第二行：重构图
        axes[1, i].imshow(
            recon[i, 0],
            cmap='gray',
            vmin=0, vmax=1,
            interpolation='nearest'
        )
        if i == 0:
            axes[1, i].set_title('Reconstruction', fontsize=12)
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # 配置路径
    CACHE_FILE   = r"E:/Project_CNN/2_Pack/data_cache.pt"
    CHECKPOINT   = r"E:/Project_CNN/v_cache/best_model.pth"
    # 可视化重构并打印指标
    visualize_reconstructions(
        cache_file=CACHE_FILE,
        checkpoint_path=CHECKPOINT,
        num_samples=5,
        seed=42
    )
