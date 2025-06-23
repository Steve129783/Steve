import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from torch.utils.data import DataLoader

# 请根据实际路径和模块名修改导入
from o3_VAE_model import VAE_ResSkip_50, CachedImageDataset


def load_model(ckpt_path, device, latent_dim=24):
    """
    加载已经训练好的 VAE_ResSkip_50。latent_dim 要与训练时保持一致。
    ckpt_path 指向 best_model.pth（一个 dict），我们只取 'model' 部分。
    """
    # 1) 实例化网络
    model = VAE_ResSkip_50(img_channels=1, latent_dim=latent_dim).to(device)
    # 2) 载入 checkpoint
    ckpt = torch.load(ckpt_path, map_location=device)
    # 3) 只从 ckpt['model'] 加载权重
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model

def evaluate_full50(model, loader, device):
    """
    对整个 50×50 图像计算 MSE、PSNR、SSIM，并打印平均值。
    """
    total_mse_sum = 0.0
    total_pixels  = 0
    psnr_list     = []
    ssim_list     = []

    with torch.no_grad():
        for imgs, _ in loader:
            imgs  = imgs.to(device)               # (B,1,50,50)
            recon, _, _ = model(imgs)             # (B,1,50,50)

            # 计算 MSE 总和
            diff2 = (recon - imgs).pow(2)          # (B,1,50,50)
            batch_mse_sum = diff2.sum().item()
            total_mse_sum += batch_mse_sum
            total_pixels  += imgs.numel()          # B × 50 × 50

            # 转成 numpy，逐张计算 PSNR/SSIM
            orig_np  = imgs.squeeze(1).cpu().numpy()   # (B,50,50)
            recon_np = recon.squeeze(1).cpu().numpy()  # (B,50,50)
            for i in range(orig_np.shape[0]):
                o50 = orig_np[i]
                r50 = recon_np[i]
                psnr_list.append(compare_psnr(o50, r50, data_range=1.0))
                ssim_list.append(compare_ssim(o50, r50, data_range=1.0))

    # 计算平均每像素 MSE、平均 PSNR、平均 SSIM
    avg_full_mse = total_mse_sum / total_pixels
    avg_psnr     = np.mean(psnr_list)
    avg_ssim     = np.mean(ssim_list)

    print(f"整张 50×50 平均每像素 MSE: {avg_full_mse:.6f}")
    print(f"整张 50×50 平均 PSNR: {avg_psnr:.2f} dB")
    print(f"整张 50×50 平均 SSIM: {avg_ssim:.4f}")


def visualize_full50_only(model, loader, device, num_images=5):
    """
    从 loader 中取一个 batch，只绘制整个 50×50 区域的原图和重建。
    """
    imgs, _ = next(iter(loader))  # 取第一 batch
    imgs = imgs.to(device)        # (B,1,50,50)
    with torch.no_grad():
        recon, _, _ = model(imgs) # (B,1,50,50)

    batch_size = imgs.shape[0]
    n_show = min(num_images, batch_size)

    plt.figure(figsize=(n_show * 2, 4))
    for i in range(n_show):
        # 原图 50×50
        input_50 = imgs[i, 0, :, :]    # (50,50)
        plt.subplot(2, n_show, i + 1)
        plt.imshow(input_50.cpu(), cmap='gray')
        plt.title("input (50×50)")
        plt.axis("off")

        # 重建 50×50
        recon_50 = recon[i, 0, :, :]   # (50,50)
        plt.subplot(2, n_show, n_show + i + 1)
        plt.imshow(recon_50.cpu(), cmap='gray')
        plt.title("recons (50×50)")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # —— 根据实际路径修改 —— 
    cache_pt    = r"E:\Project_VAE\V2\Ori_img\Val\data_cache.pt"
    ckpt_path   = r"E:\Project_VAE\V1\oVAE_cache\1_2_3_4\best_model.pth"
    batch_size  = 64
    # —— 

    # 1) 构造 Dataset、DataLoader（验证时不做数据增强）
    val_dataset = CachedImageDataset(cache_pt)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, num_workers=4, pin_memory=True
    )

    # 2) 加载模型：latent_dim 要与训练时保持一致
    model = load_model(
    r"E:\Project_VAE\V1\oVAE_cache\1_2_3_4\best_model.pth",
    device,
    latent_dim=24
)

    # 3) 定量评估（整张 50×50）
    evaluate_full50(model, val_loader, device)

    # 4) 可视化示例（整张 50×50）
    visualize_full50_only(model, val_loader, device)
