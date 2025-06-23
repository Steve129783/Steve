import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from o3_VAE_model import CachedImageDataset, VAE_ResSkip_50

def latent_interpolate(
    model, dataset, device,
    idxA, idxB,
    steps,
    out_dir
):
    os.makedirs(out_dir, exist_ok=True)

    # 1) 准备图 A 和 B
    imgA, _ = dataset[idxA]; imgB, _ = dataset[idxB]
    imgA = imgA.unsqueeze(0).to(device)
    imgB = imgB.unsqueeze(0).to(device)

    # 2) 编码拿到 mu/logvar 和 skips
    model.eval()
    with torch.no_grad():
        muA, logvarA, skipsA = model.encode(imgA)
        zA = model.reparameterize(muA, logvarA)
        muB, logvarB, skipsB = model.encode(imgB)
        zB = model.reparameterize(muB, logvarB)

    # 3) Latent 插值
    alphas = torch.linspace(0, 1, steps, device=device)
    z_interp = torch.lerp(zA, zB, alphas.unsqueeze(1))  # [steps, latent_dim]

    # 4) Skip 插值
    f3_A, f4_A = skipsA
    f3_B, f4_B = skipsB
    f3_A, f4_A = f3_A.squeeze(0), f4_A.squeeze(0)
    f3_B, f4_B = f3_B.squeeze(0), f4_B.squeeze(0)
    f3_interp = torch.lerp(f3_A, f3_B, alphas.view(-1,1,1,1))
    f4_interp = torch.lerp(f4_A, f4_B, alphas.view(-1,1,1,1))

    # 5) 解码
    with torch.no_grad():
        recons = model.decode(z_interp, (f3_interp, f4_interp))

    # 6) 保存帧和拼图
    for i in range(steps):
        out_png = os.path.join(out_dir, f"frame_{i:02d}.png")
        plt.imsave(out_png, recons[i,0].cpu(), cmap="gray", vmin=0, vmax=1)

    grid = make_grid(recons.cpu(), nrow=steps, padding=2)
    plt.figure(figsize=(steps * 0.6, 3))
    plt.imshow(grid[0], cmap="gray", vmin=0, vmax=1)
    plt.axis("off")
    plt.tight_layout()
    grid_path = os.path.join(out_dir, "interpolation_grid.png")
    plt.savefig(grid_path, dpi=120, bbox_inches="tight")
    plt.close()

    print(f"Saved {steps} frames + grid in {out_dir}")

if __name__ == "__main__":
    # ==== 配置区，请修改这三行为你本地的实际路径 ====  
    cache_path = rf"E:\Project_VAE\V1\Ori_Pack\1_2_3_4\BGMBDF\data_cache.pt"  
    model_path = rf"E:\Project_VAE\V1\oVAE_cache\1_2_3_4\best_model.pth" 
    out_dir    = rf"E:\Project_VAE\V1\Interpolation\C"    
    # ==== 结束配置 ====  

    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    latent_dim = 24       # 与训练时一致
    idxA, idxB = 22395, 5052 # 要插值的两张图索引
    steps      = 20       # 插值步数

    # 加载数据和模型
    dataset = CachedImageDataset(cache_file=cache_path)
    model   = VAE_ResSkip_50(img_channels=1, latent_dim=latent_dim).to(device)

    # 载入 checkpoint（假设你保存时是 {'model': state_dict, ...} 结构）
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt['model'], strict=False)
    model.eval()

    # 执行插值并保存结果
    latent_interpolate(
        model, dataset, device,
        idxA, idxB,
        steps=steps,
        out_dir=out_dir
    )



# rf"E:\Project_VAE\Ori_Pack\1_2_3_4\data_cache.pt"   
# rf"E:\Project_VAE\oVAE_cache\1_2_3_4\best_model.pth" 
# rf"E:\Project_VAE\Interpolation\8"   