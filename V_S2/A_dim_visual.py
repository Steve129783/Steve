import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity    as compare_ssim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# ─── 1) 复用你训练时的 Dataset ───
class CachedImageDataset(Dataset):
    def __init__(self, cache_file, label_list):
        data = torch.load(cache_file, map_location="cpu")
        self.images = data["images"]   # Tensor[N,1,50,50], float32
        assert len(label_list) == len(self.images)
        self.labels = torch.tensor(label_list, dtype=torch.float32).unsqueeze(1)
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

# ─── 2) 载入你的 CVAE 模型 ───
def load_cvae(ckpt_path, device, latent_dim=24):
    from c5_CVAE_model import CVAE_ResSkip_50
    model = CVAE_ResSkip_50(img_channels=1, latent_dim=latent_dim, label_dim=1).to(device)
    raw = torch.load(ckpt_path, map_location=device)
    rd  = raw.get('model', raw.get('model_state_dict', raw))
    fixed = {}
    for k, v in rd.items():
        k2 = k.replace('skip_conv_mid','skip_mid').replace('skip_conv_low','skip_low')
        fixed[k2] = v
    model.load_state_dict(fixed)
    model.eval()
    return model

# ─── 3) 重构评估 ───
def evaluate_recon(model, loader, device):
    total_mse = 0.0
    total_pix = 0
    psnrs = []
    ssims = []
    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device)
            B = imgs.size(0)
            y = torch.ones(B,1,device=device)
            recon, _, _ = model(imgs, y)

            diff2 = (recon - imgs).pow(2)
            total_mse += diff2.sum().item()
            total_pix += imgs.numel()

            o = imgs.squeeze(1).cpu().numpy()
            r = recon.squeeze(1).cpu().numpy()
            for i in range(B):
                psnrs.append(compare_psnr(o[i], r[i], data_range=1.0))
                ssims.append(compare_ssim(o[i], r[i], data_range=1.0))

    print("=== Reconstruction Metrics ===")
    print(f"Per-pixel MSE: {total_mse/total_pix:.6f}")
    print(f"Avg PSNR:      {np.mean(psnrs):.2f} dB")
    print(f"Avg SSIM:      {np.mean(ssims):.4f}")

# ─── 4) 潜空间遍历（无 torchvision 依赖） ───
def traverse_dims(model, first_img, device, latent_dim=24, steps=11, span=0.5, out_dir="traversals"):
    os.makedirs(out_dir, exist_ok=True)
    img0 = first_img.unsqueeze(0).to(device)
    y0   = torch.ones(1,1,device=device)
    with torch.no_grad():
        mu0, _, (f2,f3) = model.encode(img0, y0)
    mu0 = mu0.squeeze(0)

    for d in range(latent_dim):
        # 1) 重复 skip
        f2s = f2.repeat(steps,1,1,1)
        f3s = f3.repeat(steps,1,1,1)
        # 2) 构造 z 序列
        alphas = torch.linspace(mu0[d]-span, mu0[d]+span, steps, device=device)
        z0 = mu0.unsqueeze(0).repeat(steps,1)
        z0[:,d] = alphas
        # 3) 解码
        with torch.no_grad():
            recons = model.decode(z0, y0.repeat(steps,1), (f2s,f3s))  # [steps,1,50,50]

        # 4) NumPy 拼接上半重构
        rec_np = recons.squeeze(1).cpu().numpy()  # (steps,50,50)
        # 加 2px 间隔
        H, W = rec_np.shape[1], rec_np.shape[2]
        grid = np.ones((H, steps*W + (steps-1)*2)) * np.nan  # nan 用白显
        for i in range(steps):
            start = i*(W+2)
            grid[:, start:start+W] = rec_np[i]

        # 5) 差异热图
        base = rec_np[steps//2]
        diffs = rec_np - base  # (steps,50,50)
        diff_strip = np.concatenate(list(diffs), axis=1)
        vmax = np.nanpercentile(np.abs(diff_strip), 99)

        # 6) 可视化
        fig,(ax1,ax2) = plt.subplots(2,1,figsize=(steps*1.2,4),
                                     gridspec_kw={"height_ratios":[1,1]},
                                     constrained_layout=True)
        ax1.imshow(grid, cmap='gray', vmin=0, vmax=1)
        ax1.axis('off')
        ax1.set_title(f"Dim {d}   ±{span}")
        im = ax2.imshow(diff_strip, cmap='seismic', vmin=-vmax, vmax=vmax)
        ax2.axis('off')
        ax2.set_title("Δ to center")
        plt.colorbar(im, ax=ax2, orientation='horizontal', fraction=0.05, pad=0.02)

        outp = os.path.join(out_dir, f"traversal_dim{d}.png")
        plt.savefig(outp, dpi=120)
        plt.close(fig)
        print(f"Saved {outp}")

if __name__ == "__main__":
    # 用户配置
    cache_pt   = r"E:\Project_VAE\V2\Pack_json_png\data_cache.pt"
    ckpt       = r"E:\Project_VAE\V2\cvae_cache\best_model.pth"
    batch_size = 64

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # DataLoader
    cache = torch.load(cache_pt, map_location="cpu")
    labels = [1]*len(cache["images"])
    ds = CachedImageDataset(cache_pt, labels)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Load model & eval
    model = load_cvae(ckpt, device, latent_dim=24)
    evaluate_recon(model, loader, device)

    # Traverse first sample
    first_img, _ = ds[45]
    traverse_dims(model, first_img, device,
                  latent_dim=24, steps=11, span=0.5,
                  out_dir=r"E:\Project_VAE\V2\A_B\A")
