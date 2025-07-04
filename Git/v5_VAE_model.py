import os
import random, numpy as np          # 追加：用于全局固定随机种子
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
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"   # 保证 cuBLAS 可复现

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    # 关闭 TF32，避免与 FP32 路径混用导致不可复现
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32       = False


# ───────────────────────── Dataset ─────────────────────────
class CachedImageDataset(Dataset):
    """
    只返回 (image, group_id)。
    image: Tensor [1, 50, 50]
    group_id: Long
    """
    def __init__(self, cache_file):
        data = torch.load(cache_file, map_location="cpu")
        self.images = data["images"]                           # [N,1,50,50]
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
        self.gn1   = nn.GroupNorm(max(1, channels // 8), channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.gn2   = nn.GroupNorm(max(1, channels // 8), channels)
        self.relu  = nn.ReLU(True)
        self.drop  = nn.Dropout2d(0.1)

    def forward(self, x):
        out = self.relu(self.gn1(self.drop(self.conv1(x))))
        out = self.gn2(self.conv2(out))
        return self.relu(out + x)


# ─────────────────────── VAE + Classifier ───────────────────
class VAEWithClassifier(nn.Module):
    """普通 VAE（无条件），输出重构图 + 组分类 logits"""
    def __init__(self, img_channels=1, latent_dim=128, num_groups=16):
        super().__init__()
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(img_channels, 64, 4, 2, 1),  # 50→25
            nn.GroupNorm(8, 64), nn.ReLU(True), ResBlock(64))
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1),           # 25→12
            nn.GroupNorm(16, 128), nn.ReLU(True), ResBlock(128))
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, 4, 2, 1),          # 12→6
            nn.GroupNorm(32, 256), nn.ReLU(True), ResBlock(256))

        feat_dim = 256 * 6 * 6
        self.fc_mu     = nn.Linear(feat_dim, latent_dim)
        self.fc_logvar = nn.Linear(feat_dim, latent_dim)
        self.fc_dec    = nn.Linear(latent_dim, feat_dim)

        # Decoder
        self.dec3 = nn.Sequential(nn.Conv2d(256, 128, 3, 1, 1),
                                  nn.GroupNorm(16, 128), nn.ReLU(True), ResBlock(128))
        self.dec2 = nn.Sequential(nn.Conv2d(128, 64, 3, 1, 1),
                                  nn.GroupNorm(8, 64), nn.ReLU(True), ResBlock(64))
        self.dec1 = nn.Sequential(nn.Conv2d(64, 32, 3, 1, 1),
                                  nn.GroupNorm(4, 32), nn.ReLU(True), ResBlock(32),
                                  nn.Conv2d(32, img_channels, 3, 1, 1), nn.Sigmoid())

        # Group classifier
        self.classifier = nn.Linear(latent_dim, num_groups)

    # ── VAE ops ──
    def encode(self, x):
        x3 = self.enc3(self.enc2(self.enc1(x)))
        flat = x3.view(x3.size(0), -1)
        mu     = self.fc_mu(flat)
        logvar = self.fc_logvar(flat)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z):
        h = self.fc_dec(z).view(-1, 256, 6, 6)
        d = F.interpolate(self.dec3(h), size=12, mode='nearest')
        d = F.interpolate(self.dec2(d), size=25, mode='nearest')
        return F.interpolate(self.dec1(d), size=50, mode='nearest')

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        logits = self.classifier(mu)
        return recon, mu, logvar, logits


# ─────────────────────── Training Loop ──────────────────────
def main():
    # 固定一切随机性
    seed_everything(42)

    # ---- 路径 & 超参 ----
    CACHE_FILE     = r"E:\Project_CNN\2_Pack\data_cache.pt"
    CHECKPOINT_DIR = r"E:\Project_CNN\v_cache"; os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    BATCH_SIZE, NUM_EPOCHS = 64, 50
    LR, WEIGHT_DECAY = 2e-4, 1e-5
    beta_start, beta_end = 0.001, 0.002           # 线性 β
    perc_weight, cls_weight = 1.8, 1.0
    patience = 5
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Dataset & Dataloader ----
    ds = CachedImageDataset(CACHE_FILE)
    num_groups = len(torch.unique(ds.groups))

    n_total = len(ds)
    n_train = int(0.7 * n_total)
    n_val   = int(0.15 * n_total)
    n_test  = n_total - n_train - n_val
    train_ds, val_ds, test_ds = random_split(
        ds, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(42))

    g = torch.Generator().manual_seed(42)  # 控制 shuffle 顺序
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        pin_memory=True, generator=g
    )
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    # ---- 感知网络 (冻结) ----
    vgg = vgg16(weights=VGG16_Weights.DEFAULT).features[:8].to(DEVICE).eval()
    for p in vgg.parameters():
        p.requires_grad = False

    # ---- Model & Optimizer ----
    model     = VAEWithClassifier(1, 128, num_groups).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler    = GradScaler()

    mse_loss = nn.MSELoss()
    ce_loss  = nn.CrossEntropyLoss()
    beta_fn  = lambda ep: beta_start + (beta_end - beta_start) * (ep - 1) / (NUM_EPOCHS - 1)

    best_val_mse, no_improve = float('inf'), 0

    # ------------------------- Training -------------------------
    for epoch in range(1, NUM_EPOCHS + 1):
        beta = beta_fn(epoch)

        # ---- Train ----
        model.train()
        correct_tr, total_tr = 0, 0
        for img, gid in train_loader:
            img, gid = img.to(DEVICE), gid.to(DEVICE)

            optimizer.zero_grad()
            with autocast():
                recon, mu, logvar, logits = model(img)
                rec  = mse_loss(recon, img)
                perc = F.mse_loss(
                    vgg(recon.repeat(1, 3, 1, 1)),
                    vgg(img.repeat(1, 3, 1, 1))
                ) * perc_weight
                kl   = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                cls  = ce_loss(logits, gid)
                loss = rec + perc + beta * kl + cls_weight * cls

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            correct_tr += (logits.argmax(1) == gid).sum().item()
            total_tr   += gid.size(0)

        train_acc = correct_tr / total_tr

        # ---- Validate ----
        model.eval()
        correct_val, total_val = 0, 0
        val_mse, val_kl = 0.0, 0.0
        with torch.no_grad():
            for img, gid in val_loader:
                img, gid = img.to(DEVICE), gid.to(DEVICE)
                recon, mu, logvar, logits = model(img)
                val_mse += mse_loss(recon, img).item() * img.size(0)
                val_kl  += (-0.5 * torch.mean(
                    1 + logvar - mu.pow(2) - logvar.exp(), dim=1
                )).sum().item()
                correct_val += (logits.argmax(1) == gid).sum().item()
                total_val   += gid.size(0)

        val_mse /= total_val
        val_acc  = correct_val / total_val
        val_kl   = val_kl / total_val

        print(f"Epoch {epoch}/{NUM_EPOCHS} | Train Acc={train_acc:.4f} "
              f"| Val Acc={val_acc:.4f} | Val MSE={val_mse:.6f} | Val KL={val_kl:.4f}")

        # ---- Early Stopping ----
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            no_improve = 0
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "best_model.pth"))
        else:
            no_improve += 1
            print(f"No improvement for {no_improve} epoch(s)")
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # -------------------------- Test ---------------------------
    model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "best_model.pth")))
    model.eval()
    correct_te, total_te, test_mse = 0, 0, 0.0
    with torch.no_grad():
        for img, gid in test_loader:
            img, gid = img.to(DEVICE), gid.to(DEVICE)
            recon, mu, logvar, logits = model(img)
            test_mse += mse_loss(recon, img).item() * img.size(0)
            correct_te += (logits.argmax(1) == gid).sum().item()
            total_te   += gid.size(0)

    print(f"Test Acc={correct_te / total_te:.4f} | Test MSE={test_mse / total_te:.6f}")


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.freeze_support()
    main()
