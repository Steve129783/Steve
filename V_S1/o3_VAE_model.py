import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.amp import GradScaler, autocast  
import math
from torchvision.models import vgg16, VGG16_Weights

torch.backends.cudnn.benchmark = True

# ───────────── 1) 提前定义 device ─────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

weights = VGG16_Weights.DEFAULT
vgg = vgg16(weights=weights).features[:8].to(device).eval()
for p in vgg.parameters():
    p.requires_grad = False

# ================= TC 判别器定义 =================
class TCDiscriminator(nn.Module):
    def __init__(self, latent_dim, hidden_dim=100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, 1),
        )
    def forward(self, z):
        # z: (batch, latent_dim)
        return self.net(z).view(-1)  # 输出 (batch,)

# =========================
# 2) 定义 CachedImageDataset：直接从 data_cache.pt 里加载 50×50 图
# =========================
class CachedImageDataset(Dataset):
    def __init__(self, cache_file):
        """
        cache_file: 用 build_cache.py 生成的 .pt 文件路径（里面存了 (N,1,50,50) 的 images 和 paths 列表）
        """
        data = torch.load(cache_file, map_location="cpu")
        self.images = data["images"]       # Tensor of shape (N,1,50,50), dtype=torch.float32, 已归一化到 [0,1]
        self.paths  = data.get("paths", None)
        assert isinstance(self.images, torch.FloatTensor), "images 必须是 FloatTensor"

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        """
        返回：
          img : Tensor, 形状 (1,50,50), dtype=torch.float32, 值在 [0,1]
          idx_or_path: int 或 str（路径，用于调试）
        """
        img = self.images[idx]  # (1,50,50)
        if self.paths is not None:
            return img, self.paths[idx]
        else:
            return img, idx


# =========================
# 3) 定义 ResBlock 和 VAE_ResSkip_50 模型
# =========================
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.dropout = nn.Dropout2d(0.2)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.dropout = nn.Dropout2d(0.1)  # 新增
        self.gn1 = nn.GroupNorm(max(1, channels//8), channels)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(max(1, channels//8), channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.dropout(out)  # 关键位置
        out = self.gn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.gn2(out)
        return self.relu(out + residual)


class VAE_ResSkip_50(nn.Module):
    def __init__(self, img_channels=1, latent_dim=16):  # 建议增大潜在维度
        super().__init__()
        self.latent_dim = latent_dim
        self.register_buffer('current_epoch', torch.tensor(0))
        
        # Skip connections
        self.skip_conv_mid  = nn.Conv2d(128, 256, kernel_size=1)  
        self.skip_conv_low  = nn.Conv2d(64,  128, kernel_size=1)
        
        # 可学习融合系数 (使用sigmoid约束到[0,1]范围)
        self.alpha_low = nn.Parameter(torch.tensor(0.3))
        self.alpha_mid = nn.Parameter(torch.tensor(0.5))
        nn.init.constant_(self.alpha_low, 0.3)
        nn.init.constant_(self.alpha_mid, 0.5)
        # ---------- Encoder (输入是 1×50×50) ----------
        # 1: 1×50×50 → 32×25×25
        self.enc_conv1 = nn.Sequential(
            nn.Conv2d(img_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(num_groups=4, num_channels=32),
            nn.ReLU(True),
            ResBlock(32)
        )
        # 2: 32×25×25 → 64×12×12 →
        self.enc_conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=64),
            nn.ReLU(True),
            ResBlock(64)
        )
        # 3: 64×12×12 → 128×6×6
        self.enc_conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(num_groups=16, num_channels=128),
            nn.ReLU(True),
            ResBlock(128)
        )
        # 4: 128×6×6 → 256×3×3
        self.enc_conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=256),
            nn.ReLU(True),
            ResBlock(256)
        )
        # 5: 256×3×3 → 512×1×1
        self.enc_conv5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=512),
            nn.ReLU(True),
            ResBlock(512)
        )
        # 从 (512×1×1) flatten 到 latent
        self.fc_mu     = nn.Linear(512 * 1 * 1, latent_dim)
        self.fc_logvar = nn.Linear(512 * 1 * 1, latent_dim)
        nn.init.constant_(self.fc_logvar.bias, -1.0)

        # ---------- Decoder (从 latent → 重构 1×50×50) ----------
        # latent_dim → 512×1×1
        self.fc_dec = nn.Linear(latent_dim, 512 * 1 * 1)

        # 5': (512) @ 3×3 → 256 @ 3×3
        self.dec_conv5 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=256),
            nn.ReLU(True), 
            ResBlock(256)
        )
        # 4': (256) @ 6×6 → 128 @ 6×6
        self.dec_conv4 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=16, num_channels=128),
            nn.ReLU(True),
            ResBlock(128)
        )
        # 3': (128) @ 12×12 → 64 @ 12×12
        self.dec_conv3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=16, num_channels=64),
            nn.ReLU(True),
            ResBlock(64)
        )
        # 2': (64) @ 25×25 → 32 @ 25×25
        self.dec_conv2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=32),
            nn.ReLU(True),
            ResBlock(32)
        )
        # 1': 32 @ 50×50 → 16 @ 50×50 → 1 @ 50×50
        self.dec_conv1 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=4, num_channels=16),
            nn.ReLU(True),
            ResBlock(16),
            nn.Conv2d(16, img_channels, kernel_size=3, padding=1),
            nn.Sigmoid()  # 输出范围 [0,1]
        )

    def encode(self, x):
        # 1) 前几层
        x1 = self.enc_conv1(x)   # 32×25×25
        x2 = self.enc_conv2(x1)  # 64×12×12  ← f2
        x3 = self.enc_conv3(x2)  # 128×6×6  ← f3
        x4 = self.enc_conv4(x3)  # 256×3×3
        x5 = self.enc_conv5(x4)  # 512×1×1
    
        # 2) 扁平化并投影到潜空间
        h = x5.view(x5.size(0), -1)           # [B, 512]
        mu     = self.fc_mu(h)               # [B, latent_dim]
        logvar = self.fc_logvar(h)           # [B, latent_dim]
        
        # 3) 返回 mu, logvar 以及 (x2, x3) 作为 skips
        return mu, logvar, (x2, x3)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        # 添加正交约束
        if self.training:
            z_norm = F.normalize(z, dim=1)
            cov = torch.mm(z_norm.T, z_norm)
            eye = torch.eye(self.latent_dim, device=device)
            self.orth_loss = (cov - eye).pow(2).mean()  # 均方误差
            z = z - 1e-5 * self.orth_loss * z_norm.detach()
            
        return z

    def decode(self, z, skips):
        # 拆包 skips
        f2, f3 = skips  # f2: 64×12×12, f3: 128×6×6

        # 1) 从 z 开始：latent_dim → 512×1×1 → 上采样到 3×3
        h0 = self.fc_dec(z).view(-1, 512, 1, 1)      # [B,512,1,1]
        h1 = F.interpolate(h0, size=3, mode='nearest')  # [B,512,3,3]

        # 2) dec_conv5：和原来一样，把 h1→d5 (256×3×3)
        d5 = self.dec_conv5(h1)  # [B,256,3,3]

        # 3) 融合第三层 skip (f3) → dec_conv4
        #   3.1 上采样 d5 到 6×6
        h2 = F.interpolate(d5, size=6, mode='nearest')  # [B,256,6,6]
        #   3.2 对 f3 做 128→256 的映射
        f3_ = self.skip_conv_mid(f3)                   # [B,256,6,6]
        #   3.3 计算加权系数
        a_mid = torch.sigmoid(self.alpha_mid) * 0.9 + 0.05
        #   3.4 融合并卷积
        d4 = self.dec_conv4(a_mid * h2 + (1 - a_mid) * f3_)  # [B,128,6,6]

        # 4) 融合第二层 skip (f2) → dec_conv3
        #   4.1 上采样 d4 到 12×12
        h3 = F.interpolate(d4, size=12, mode='nearest')     # [B,128,12,12]
        #   4.2 对 f2 做 64→128 的映射，并上采样到 12×12
        f2_up = F.interpolate(f2, size=12, mode='nearest')  # [B,64,12,12]
        f2_   = self.skip_conv_low(f2_up)                   # [B,128,12,12]
        #   4.3 计算加权系数
        a_low = torch.sigmoid(self.alpha_low) * 0.9 + 0.05
        #   4.4 融合并卷积
        d3 = self.dec_conv3(a_low * h3 + (1 - a_low) * f2_)  # [B,64,12,12]

        # 5) 继续还原到 25×25 → dec_conv2
        h4 = F.interpolate(d3, size=25, mode='nearest')     # [B,64,25,25]
        d2 = self.dec_conv2(h4)                             # [B,32,25,25]

        # 6) 最后还原到 50×50 → dec_conv1
        h5 = F.interpolate(d2, size=50, mode='nearest')     # [B,32,50,50]
        out = self.dec_conv1(h5)                            # [B,1,50,50]

        return out

    def forward(self, x):
        mu, logvar, skips = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, skips)
        return recon, mu, logvar

# =========================
# 4) 定义 MSE Loss（对整个 50×50 计算）
# =========================
mse_loss = nn.MSELoss()



# =========================
# 6) 谱归一化卷积层
# =========================
class SpectralNormConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.utils.spectral_norm(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        )
    
    def forward(self, x):
        return self.conv(x)


# =========================
# 7) 改进的判别器（使用谱归一化）
# =========================
class Discriminator50(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # 直接用普通 Conv2d，去掉谱归一化
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # 最后输出 1×3×3
            nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        out = self.net(x)         # (batch,1,3,3)
        b = out.size(0)
        out = out.view(b, -1)     # (batch, 9)
        return out.mean(1)        # (batch,)

# =========================
# 8) 训练脚本主入口
# =========================
if __name__ == "__main__":
    # --- 超参数配置 ---
    cache_path = r"E:\Project_VAE\V1\Aug_Pack\1_2_3_4\data_cache.pt"
    save_dir = r"E:\Project_VAE\V1\oVAE_cache\1_2_3_4"
    os.makedirs(save_dir, exist_ok=True)
    
    # training parameters
    batch_size = 64
    num_epochs = 20
    latent_dim = 24
    beta_start, beta_end = 0.001, 0.1  
    warmup_epochs_beta = 12  
    
    # Loss weights
    recon_weight = 1
    perc_weight = 0.05
    tc_weight = 0.3
    orth_weight = 1e-5

    # --- 初始化 ---
    model = VAE_ResSkip_50(img_channels=1, latent_dim=latent_dim).to(device)
    tc_discriminator = TCDiscriminator(latent_dim).to(device)
    # 1) 划分子集
    full_dataset = CachedImageDataset(cache_file=cache_path)
    N = len(full_dataset)
    train_size = int(0.7 * N)
    val_size   = int(0.15 * N)
    test_size  = N - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
    full_dataset,
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)
)
    # --- 数据加载 ---
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    # 优化器
    optimizer_vae = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    optimizer_tc = optim.Adam(tc_discriminator.parameters(), lr=1e-4)
    
    # 学习率调度
    scheduler_vae = optim.lr_scheduler.CyclicLR(
    optimizer_vae,
    base_lr=1e-5,
    max_lr=1e-4,
    step_size_up=len(train_loader)*5,
    cycle_momentum=False
)
    
    scaler = GradScaler()

    # --- 训练循环 ---
    best_val_loss = float('inf')
    for epoch in range(1, num_epochs + 1):
        model.train()
        tc_discriminator.train()
        model.current_epoch.fill_(epoch)
        
        # β值计算
        beta = beta_start + (beta_end-beta_start)*min(1.0, epoch/warmup_epochs_beta)
        
        for imgs, _ in train_loader:
            imgs = imgs.to(device)
            
            # --- TC判别器更新 ---
            with torch.no_grad():
                mu, logvar, _ = model.encode(imgs)
                z = model.reparameterize(mu, logvar)
            
            optimizer_tc.zero_grad()
            z_perm = z[torch.randperm(z.size(0))]
            
            logits_j = tc_discriminator(z.detach())
            logits_m = tc_discriminator(z_perm.detach())
            
            loss_tc_d = 0.5 * (
                F.binary_cross_entropy_with_logits(logits_j, torch.ones_like(logits_j)) +
                F.binary_cross_entropy_with_logits(logits_m, torch.zeros_like(logits_m))
            )
            loss_tc_d.backward()
            optimizer_tc.step()

            # --- VAE主模型更新 ---
            optimizer_vae.zero_grad()
            
            with autocast(device_type=device.type):
                # 前向传播
                recon, mu, logvar = model(imgs)
                
                # 各项损失计算
                recon_loss = mse_loss(recon, imgs) * recon_weight
                perc_loss = F.mse_loss(vgg(recon.repeat(1,3,1,1)),
                       vgg(imgs.repeat(1,3,1,1))) * perc_weight
                
                # KL散度
                kl_per_dim = -0.5*(1 + logvar - mu.pow(2) - logvar.exp())
                min_kl = 0.25 * latent_dim  # 最小KL目标值=6
                kl_loss = torch.max(kl_per_dim.sum(1).mean(), torch.tensor(min_kl * beta, device=device))
                
                # TC惩罚
                z = model.reparameterize(mu, logvar)
                tc_loss = (tc_discriminator(z)**2).mean() * tc_weight
                
                # 总损失
                total_loss = recon_loss + beta*kl_loss + perc_loss + tc_loss
                
                # 正交约束
                if hasattr(model, 'orth_loss'):
                    total_loss += orth_weight * model.orth_loss

            # 反向传播
            scaler.scale(total_loss).backward()
            scaler.step(optimizer_vae)
            scaler.update()
            scheduler_vae.step()

        # --- 验证阶段 ---
        model.eval()
        val_loss = 0.0
        z_list, kl_list = [], []
        
        with torch.no_grad():
            for imgs, _ in val_loader:
                imgs = imgs.to(device)
                recon, mu, logvar = model(imgs)
                
                val_loss += mse_loss(recon, imgs).item()
                z = model.reparameterize(mu, logvar)
                z_list.append(z)
                
                kl = -0.5*(1 + logvar - mu.pow(2) - logvar.exp()).sum(1)
                kl_list.append(kl)
        
        # 计算指标
        val_loss /= len(val_loader)
        z_all = torch.cat(z_list)
        kl_all = torch.cat(kl_list)
        
        z_mean = z_all.mean(0)
        z_std = z_all.std(0)
        active_dims = (z_std > 0.05).sum().item()
        z_cov = torch.cov(z_all.T)
        cond_num = torch.linalg.cond(z_cov)
        
        # 打印日志
        print(f"\nEpoch {epoch}/{num_epochs}:")
        print(f"β={beta:.4f} | LR={optimizer_vae.param_groups[0]['lr']:.2e}")
        print(f"Train Loss: {total_loss.item():.4f} (Recon:{recon_loss.item():.4f}, KL:{kl_loss.item():.4f})")
        print(f"Val MSE: {val_loss:.6f}")
        print(f"Latent Space - μ:{z_mean.mean():.3f}±{z_mean.std():.3f}")
        print(f"σ:{z_std.mean():.3f}±{z_std.std():.3f} | Active:{active_dims}/{latent_dim}")
        print(f"Cov Cond: {cond_num:.1f} | KL/dim: {kl_all.mean()/latent_dim:.4f}")
        # 潜在空间健康度检测
        kl_per_dim = -0.5*(1 + logvar - mu.pow(2) - logvar.exp())
        print(f"KL per dim: {kl_per_dim.mean(0).cpu().detach().numpy()}")  # 各维度KL分布

        # 重建质量分析
        psnr = 10 * torch.log10(1 / mse_loss(recon, imgs)) 
        print(f"PSNR: {psnr.mean().item():.2f} dB")

        # 模型保存
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer_vae.state_dict(),
                'best_val_loss': best_val_loss,
            }, os.path.join(save_dir, 'best_model.pth'))

    # 最终保存
    torch.save(model.state_dict(), os.path.join(save_dir, 'final_model.pth'))
    print("Training completed!")