import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity    as compare_ssim
from torch.utils.data import DataLoader

# 根据你项目路径修改以下两行
from c5_CVAE_model import CVAE_ResSkip_50, CachedImageDataset  

def load_cvae(ckpt_path, device, latent_dim=24):
    """加载 CVAE_ResSkip_50 模型。"""
    model = CVAE_ResSkip_50(img_channels=1, latent_dim=latent_dim, label_dim=1).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    # 处理 key 的命名差异
    raw = ckpt.get('model', ckpt.get('model_state_dict', ckpt))
    fixed = {}
    for k, v in raw.items():
        k2 = k.replace('skip_conv_mid','skip_mid')\
              .replace('skip_conv_low','skip_low')
        fixed[k2] = v
    model.load_state_dict(fixed)
    model.eval()
    return model

def evaluate_cvae(model, loader, device):
    """对 CVAE 进行重构误差、PSNR、SSIM 评估。"""
    total_mse_sum = 0.0
    total_pixels  = 0
    psnr_list     = []
    ssim_list     = []

    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device)            # (B,1,50,50)
            B = imgs.size(0)
            # 条件 y 全 1，表示有缺陷
            y = torch.ones(B, 1, device=device)
            recon, mu, logvar = model(imgs, y)  # (B,1,50,50)

            # 累积 MSE
            diff2 = (recon - imgs).pow(2)
            total_mse_sum += diff2.sum().item()
            total_pixels  += imgs.numel()

            # 转 numpy 计算 PSNR/SSIM
            orig_np  = imgs.squeeze(1).cpu().numpy()
            recon_np = recon.squeeze(1).cpu().numpy()
            for i in range(B):
                psnr_list.append(compare_psnr(orig_np[i], recon_np[i], data_range=1.0))
                ssim_list.append(compare_ssim(orig_np[i], recon_np[i], data_range=1.0))

    avg_mse  = total_mse_sum / total_pixels
    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)

    print(f"全 ROI 平均每像素 MSE: {avg_mse:.6f}")
    print(f"全 ROI 平均 PSNR: {avg_psnr:.2f} dB")
    print(f"全 ROI 平均 SSIM: {avg_ssim:.4f}")

def visualize_reconstructions(model, loader, device, n_show=5):
    """展示前 n_show 张 ROI 的 原图 vs 重构 图像。"""
    imgs, _ = next(iter(loader))
    imgs = imgs.to(device)
    B = imgs.size(0)
    y = torch.ones(B, 1, device=device)
    with torch.no_grad():
        recon, _, _ = model(imgs, y)

    n = min(n_show, B)
    plt.figure(figsize=(n*2, 4))
    for i in range(n):
        plt.subplot(2, n, i+1)
        plt.imshow(imgs[i,0].cpu(), cmap='gray')
        plt.title("orig")
        plt.axis('off')

        plt.subplot(2, n, n+i+1)
        plt.imshow(recon[i,0].cpu(), cmap='gray')
        plt.title("recon")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # ——— 配置路径，请根据实际修改 ———
    cache_pt  = r"E:\Project_VAE\V2\Pack_json_png\Ori_drop\data_cache.pt"
    ckpt_path = r"E:\Project_VAE\V2\cvae_cache\best_model.pth"
    batch_size = 64
    # ———————————————————————————
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) DataLoader
    val_ds = CachedImageDataset(cache_pt, label_list=[1]*len(torch.load(cache_pt)["images"]))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    # 2) 加载 CVAE 模型
    model = load_cvae(ckpt_path, device, latent_dim=24)

    # 3) 定量评估
    evaluate_cvae(model, val_loader, device)

    # 4) 可视化前几张重构
    visualize_reconstructions(model, val_loader, device, n_show=5)
