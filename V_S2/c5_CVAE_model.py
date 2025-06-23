import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
from torchvision.models import vgg16, VGG16_Weights

# ───── Dataset ─────
class CachedImageDataset(Dataset):
    def __init__(self, cache_file, label_list):
        data = torch.load(cache_file, map_location="cpu")
        self.images = data["images"]   # Tensor[N,1,50,50]
        assert len(label_list) == len(self.images), "标签长度必须等于样本数量"
        self.labels = torch.tensor(label_list, dtype=torch.float32).unsqueeze(1)
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

# ───── ResBlock ─────
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

# ───── CVAE with skip-connections ─────
class CVAE_ResSkip_50(nn.Module):
    def __init__(self, img_channels=1, latent_dim=24, label_dim=1):
        super().__init__()
        self.latent_dim = latent_dim
        self.label_dim  = label_dim

        # Encoder blocks
        self.enc1 = nn.Sequential(
            nn.Conv2d(img_channels, 32, 4, 2, 1),
            nn.GroupNorm(4,32), nn.ReLU(True), ResBlock(32)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.GroupNorm(8,64), nn.ReLU(True), ResBlock(64)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(64,128,4,2,1),
            nn.GroupNorm(16,128), nn.ReLU(True), ResBlock(128)
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(128,256,4,2,1),
            nn.GroupNorm(32,256), nn.ReLU(True), ResBlock(256)
        )
        self.enc5 = nn.Sequential(
            nn.Conv2d(256,512,4,2,1),
            nn.GroupNorm(32,512), nn.ReLU(True), ResBlock(512)
        )
        # μ & logvar heads
        self.fc_mu     = nn.Linear(512 + label_dim, latent_dim)
        self.fc_logvar = nn.Linear(512 + label_dim, latent_dim)

        # 1×1 skips to align channels
        self.skip_mid = nn.Conv2d(128, 256, 1)
        self.skip_low = nn.Conv2d(64,  128, 1)

        # Decoder blocks
        self.fc_dec = nn.Linear(latent_dim + label_dim, 512)
        self.dec5   = nn.Sequential(
            nn.Conv2d(512,256,3,1,1), nn.GroupNorm(32,256), nn.ReLU(True), ResBlock(256)
        )
        self.dec4   = nn.Sequential(
            nn.Conv2d(256,128,3,1,1), nn.GroupNorm(16,128), nn.ReLU(True), ResBlock(128)
        )
        self.dec3   = nn.Sequential(
            nn.Conv2d(128,64,3,1,1),  nn.GroupNorm(16,64),  nn.ReLU(True), ResBlock(64)
        )
        self.dec2   = nn.Sequential(
            nn.Conv2d(64,32,3,1,1),   nn.GroupNorm(8,32),   nn.ReLU(True), ResBlock(32)
        )
        self.dec1   = nn.Sequential(
            nn.Conv2d(32,16,3,1,1), nn.GroupNorm(4,16), nn.ReLU(True), ResBlock(16),
            nn.Conv2d(16,img_channels,3,1,1), nn.Sigmoid()
        )

    def encode(self, x, y):
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x5 = self.enc5(x4)
        h  = x5.view(x5.size(0), 512)
        h_cond = torch.cat([h, y], dim=1)
        mu     = self.fc_mu(h_cond)
        logvar = self.fc_logvar(h_cond)
        return mu, logvar, (x2, x3)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z, y, skips):
        f2, f3 = skips
        zc     = torch.cat([z, y], dim=1)
        h0     = self.fc_dec(zc).unsqueeze(-1).unsqueeze(-1)
        d5     = self.dec5(F.interpolate(h0, size=3, mode='nearest'))

        # fuse skip at 3rd layer
        f3m = self.skip_mid(f3)
        d4  = self.dec4(F.interpolate(d5, size=6, mode='nearest') + f3m)

        # fuse skip at 2nd layer
        f2m = self.skip_low(F.interpolate(f2, size=12, mode='nearest'))
        d3  = self.dec3(F.interpolate(d4, size=12, mode='nearest') + f2m)

        d2  = self.dec2(F.interpolate(d3, size=25, mode='nearest'))
        return self.dec1(F.interpolate(d2, size=50, mode='nearest'))

    def forward(self, x, y):
        mu, logvar, skips = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, y, skips), mu, logvar

# ───── Training Script ─────
def main():
    # —— Config —— #
    CACHE_FILE     = r"E:\Project_VAE\V2\Pack_json_png\data_cache.pt"
    CHECKPOINT_DIR = "checkpoints"
    BATCH_SIZE     = 64
    NUM_EPOCHS     = 12
    LR, WD         = 1e-4, 1e-5
    beta_start, beta_end = 0.001, 0.06
    perc_weight    = 0.05
    min_kl_factor  = 0.25  # min_kl = min_kl_factor * latent_dim
    DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # —— Load cache & auto-labels —— #
    cache = torch.load(CACHE_FILE, map_location="cpu")
    masks = cache["masks"]  # Tensor[N,1,50,50]
    labels= (masks.view(len(masks), -1).sum(1) > 0).long().tolist()

    # —— VGG perceptual net —— #
    vgg = vgg16(weights=VGG16_Weights.DEFAULT).features[:8].to(DEVICE).eval()
    for p in vgg.parameters():
        p.requires_grad = False

    # —— Datasets & loaders —— #
    ds        = CachedImageDataset(CACHE_FILE, labels)
    n_train   = int(0.7 * len(ds))
    train_ds, val_ds = random_split(
        ds, [n_train, len(ds) - n_train],
        generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)

    # —— Model, optimizer, AMP —— #
    latent_dim = 24
    model      = CVAE_ResSkip_50(1, latent_dim=latent_dim, label_dim=1).to(DEVICE)
    optimizer  = optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    scaler     = GradScaler()
    mse_loss   = nn.MSELoss()

    # —— β schedule —— #
    def get_beta(ep):
        return beta_start + (beta_end - beta_start) * (ep - 1) / (NUM_EPOCHS - 1)

    best_val = float('inf')
    for epoch in range(1, NUM_EPOCHS + 1):
        beta = get_beta(epoch)
        min_kl = min_kl_factor * latent_dim

        # ——— train ———
        model.train()
        for imgs, labs in train_loader:
            imgs, labs = imgs.to(DEVICE), labs.to(DEVICE)
            optimizer.zero_grad()
            with autocast():
                recon, mu, logvar = model(imgs, labs)

                # recon + perceptual
                rec_loss = mse_loss(recon, imgs)
                pr = recon.repeat(1,3,1,1)
                po = imgs.repeat(1,3,1,1)
                perc_loss = F.mse_loss(vgg(pr), vgg(po)) * perc_weight

                # per-dim KL
                kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())  # [B,D]
                kl_batch  = kl_per_dim.sum(1).mean()  # scalar

                # apply lower bound
                kl_loss = torch.max(kl_batch, torch.tensor(min_kl * beta, device=DEVICE))

                loss = rec_loss + beta * kl_loss + perc_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # ——— validate ———
        model.eval()
        tot_mse, kl_dims_all = 0.0, []
        with torch.no_grad():
            for imgs, labs in val_loader:
                imgs, labs = imgs.to(DEVICE), labs.to(DEVICE)
                recon, mu, logvar = model(imgs, labs)
                tot_mse += mse_loss(recon, imgs).item() * imgs.size(0)
                kl_pd = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())  # [B,D]
                kl_dims_all.append(kl_pd)

        val_mse   = tot_mse / len(val_ds)
        kl_tensor = torch.cat(kl_dims_all, dim=0)         # [N_val, D]
        kl_per    = kl_tensor.mean(0)                    # [D]
        kl_total  = kl_per.sum().item()

        print(f"Epoch {epoch:02d}/{NUM_EPOCHS} | β={beta:.4f}"
              f" | Val MSE={val_mse:.6f} | Val KL={kl_total:.4f}")
        print(f"    KL per dim: {kl_per.cpu().numpy()}")

        # save best
        if val_mse < best_val:
            best_val = val_mse
            torch.save(model.state_dict(),
                       os.path.join(CHECKPOINT_DIR, "best_cvae.pth"))

    print("Training complete.")

if __name__ == "__main__":
    # Windows 下 DataLoader 多进程支持
    import torch.multiprocessing as mp
    mp.freeze_support()
    main()
