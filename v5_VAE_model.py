import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
from torchvision.models import vgg16, VGG16_Weights

# ───────────────────────── Reproducibility ─────────────────────────
def seed_everything(seed: int = 42):
    """一键固定 Python / NumPy / Torch / CUDA 随机性"""
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32       = False


# ───────────────────────── Dataset ─────────────────────────
class CachedImageDataset(Dataset):
    """只返回 (image, group_id)。image: [1,50,50], group_id: long"""
    def __init__(self, cache_file):
        data = torch.load(cache_file, map_location="cpu")
        self.images = data["images"]
        self.groups = torch.tensor(data["group_ids"], dtype=torch.long)
        assert len(self.images) == len(self.groups)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.groups[idx]


# ───────────────────────── ResBlock ─────────────────────────
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


# ─────────────────────── VAE + Classifier ───────────────────
class VAEWithClassifier(nn.Module):
    def __init__(self, img_channels=1, latent_dim=128, num_groups=16):
        super().__init__()
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(img_channels, 64, 4, 2, 1),  # 50→25
            nn.GroupNorm(8, 64), nn.ReLU(True), ResBlock(64)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1),           # 25→12
            nn.GroupNorm(16, 128), nn.ReLU(True), ResBlock(128)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, 4, 2, 1),          # 12→6
            nn.GroupNorm(32, 256), nn.ReLU(True), ResBlock(256)
        )
        feat_dim = 256*6*6
        self.fc_mu     = nn.Linear(feat_dim, latent_dim)
        self.fc_logvar = nn.Linear(feat_dim, latent_dim)
        self.fc_dec    = nn.Linear(latent_dim, feat_dim)

        # Decoder
        self.dec3 = nn.Sequential(
            nn.Conv2d(256,128,3,1,1), nn.GroupNorm(16,128), nn.ReLU(True), ResBlock(128)
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(128,64,3,1,1), nn.GroupNorm(8,64), nn.ReLU(True), ResBlock(64)
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(64,32,3,1,1), nn.GroupNorm(4,32), nn.ReLU(True), ResBlock(32),
            nn.Conv2d(32,img_channels,3,1,1), nn.Sigmoid()
        )

        # —— 新增：平滑卷积（去除插值伪影） —— #
        self.smooth3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.smooth2 = nn.Conv2d( 64,  64, kernel_size=3, padding=1)

        # Classifier on μ
        self.classifier = nn.Linear(latent_dim, num_groups)

    def encode(self, x):
        x3 = self.enc3(self.enc2(self.enc1(x)))
        flat = x3.view(x3.size(0), -1)
        return self.fc_mu(flat), self.fc_logvar(flat)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        return mu + torch.randn_like(std)*std

    def decode(self, z):
        # z: [B, latent_dim]
        h = self.fc_dec(z).view(-1, 256, 6, 6)

        # 第一次上采样到 12×12
        d = self.dec3(h)  # -> [B,128,6,6]
        d = F.interpolate(d, size=12, mode='bilinear', align_corners=False, antialias=True)
        d = F.relu(self.smooth3(d))  # 平滑

        # 第二次上采样到 25×25
        d = self.dec2(d)  # -> [B,64,12,12]
        d = F.interpolate(d, size=25, mode='bilinear', align_corners=False, antialias=True)
        d = F.relu(self.smooth2(d))  # 平滑

        # 最后输出到 50×50
        d = self.dec1(d)  # -> [B,1,25,25]
        d = F.interpolate(d, size=50, mode='bilinear', align_corners=False, antialias=True)
        return d

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        logits = self.classifier(mu)
        return recon, mu, logvar, logits


# ───────────────────────── Discriminator ─────────────────────────
class Discriminator(nn.Module):
    def __init__(self, img_channels=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(img_channels,32,4,2,1), nn.LeakyReLU(0.2,True),
            nn.Conv2d(32,64,4,2,1),           nn.BatchNorm2d(64), nn.LeakyReLU(0.2,True),
            nn.Conv2d(64,128,4,2,1),          nn.BatchNorm2d(128), nn.LeakyReLU(0.2,True),
            nn.Flatten(),
            nn.Linear(128*6*6,1),             nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


# ─────────────────────── Training Loop ──────────────────────
def main():
    seed_everything(42)

    # paths & params
    CACHE_FILE     = r"E:\Project_CNN\2_Pack\data_cache.pt"
    CHECKPOINT_DIR = r"E:\Project_CNN\v_cache"; os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    BATCH_SIZE, NUM_EPOCHS = 64, 50
    LR, WEIGHT_DECAY       = 1e-4, 1e-5
    beta_start, beta_end   = 0.002, 0.003
    perc_weight, cls_weight, adv_weight = 0.3, 1.0, 0.001
    patience               = 7
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # data
    ds = CachedImageDataset(CACHE_FILE)
    n = len(ds)
    n_train = int(0.7*n); n_val = int(0.15*n); n_test = n-n_train-n_val
    train_ds, val_ds, test_ds = random_split(
        ds, [n_train,n_val,n_test], generator=torch.Generator().manual_seed(42)
    )
    g = torch.Generator().manual_seed(42)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              pin_memory=True, generator=g)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    # frozen perceptual net
    vgg = vgg16(weights=VGG16_Weights.DEFAULT).features[:8].to(DEVICE).eval()
    for p in vgg.parameters(): p.requires_grad = False

    # models & optimizers
    num_groups = len(torch.unique(ds.groups))
    model = VAEWithClassifier(1, 128, num_groups).to(DEVICE)
    disc  = Discriminator(1).to(DEVICE)
    opt_g = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    opt_d = optim.Adam(disc.parameters(),  lr=LR* 0.05, betas=(0.5,0.999))
    scaler = GradScaler()
    mse_loss = nn.MSELoss()
    ce_loss  = nn.CrossEntropyLoss()
    beta_fn  = lambda ep: beta_start + (beta_end-beta_start)*(ep-1)/(NUM_EPOCHS-1)

    best_val_mse, no_improve = float('inf'), 0

    for epoch in range(1, NUM_EPOCHS+1):
        beta = beta_fn(epoch)
        model.train(); disc.train()
        running_d_real, running_d_fake, running_adv = 0.0,0.0,0.0
        batches = 0

        for img, gid in train_loader:
            batches += 1
            img, gid = img.to(DEVICE), gid.to(DEVICE)

            # 1) update Discriminator
            with torch.no_grad():
                fake, *_ = model(img)
            pred_real = disc(img)
            pred_fake = disc(fake)
            loss_d = -torch.mean(torch.log(pred_real+1e-8) + torch.log(1-pred_fake+1e-8))
            opt_d.zero_grad(); loss_d.backward(); opt_d.step()
            running_d_real += pred_real.mean().item()
            running_d_fake += pred_fake.mean().item()

            # 2) update Generator/VAE
            opt_g.zero_grad()
            with autocast():
                recon, mu, logvar, logits = model(img)
                rec  = mse_loss(recon, img)
                perc = F.mse_loss(vgg(recon.repeat(1,3,1,1)),
                                  vgg(img.repeat(1,3,1,1))) * perc_weight
                kl   = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                cls  = ce_loss(logits, gid)
                adv  = -torch.mean(torch.log(disc(recon)+1e-8))
                loss = rec + perc + beta*kl + cls_weight*cls + adv_weight*adv

            scaler.scale(loss).backward()
            scaler.step(opt_g)
            scaler.update()
            running_adv += adv.item()

        # log GAN metrics
        avg_real = running_d_real/batches
        avg_fake = running_d_fake/batches
        avg_adv  = running_adv/batches
        print(f"[Epoch {epoch}/{NUM_EPOCHS}] "
              f"D_real={avg_real:.3f}  D_fake={avg_fake:.3f}  Adv_loss={avg_adv:.3f}")

        # validation
        model.eval(); disc.eval()
        val_mse, val_kl, correct_val, total_val = 0.0,0.0,0,0
        with torch.no_grad():
            for img, gid in val_loader:
                img, gid = img.to(DEVICE), gid.to(DEVICE)
                recon, mu, logvar, logits = model(img)
                val_mse += mse_loss(recon, img).item()*img.size(0)
                val_kl  += (-0.5*torch.mean(1+logvar-mu.pow(2)-logvar.exp(),dim=1)).sum().item()
                correct_val += (logits.argmax(1)==gid).sum().item()
                total_val   += gid.size(0)

        val_mse /= total_val
        val_acc  = correct_val/total_val
        val_kl   = val_kl/total_val
        print(f"         Val Acc={val_acc:.4f}  Val MSE={val_mse:.6f}  Val KL={val_kl:.4f}")

        # early stop
        if val_mse < best_val_mse:
            best_val_mse, no_improve = val_mse, 0
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "best_model.pth"))
        else:
            no_improve += 1
            if no_improve>=patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # test
    model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR,"best_model.pth")))
    model.eval()
    test_mse, correct_te, total_te = 0.0,0,0
    with torch.no_grad():
        for img, gid in test_loader:
            img, gid = img.to(DEVICE), gid.to(DEVICE)
            recon, mu, logvar, logits = model(img)
            test_mse   += mse_loss(recon,img).item()*img.size(0)
            correct_te += (logits.argmax(1)==gid).sum().item()
            total_te   += gid.size(0)

    print(f"Test Acc={correct_te/total_te:.4f}  Test MSE={test_mse/total_te:.6f}")


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.freeze_support()
    main()
