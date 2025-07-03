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
    def __init__(self, cache_file, label_list, group_list):
        data = torch.load(cache_file, map_location="cpu")
        self.images = data["images"]
        assert len(label_list) == len(self.images), "标签长度必须等于样本数量"
        assert len(group_list) == len(self.images), "组 ID 长度必须等于样本数量"
        self.labels = torch.tensor(label_list, dtype=torch.float32).unsqueeze(1)
        self.groups = torch.tensor(group_list, dtype=torch.long)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx], self.groups[idx]

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

# ───── CVAE + Classifier ─────
class CVAE_ResSimple_50(nn.Module):
    def __init__(self, img_channels=1, latent_dim=32, label_dim=1, num_groups=16):
        super().__init__()

        self.enc1 = nn.Sequential(
            nn.Conv2d(img_channels, 64, 4, 2, 1), nn.GroupNorm(8,64), nn.ReLU(True), ResBlock(64) 
        )# conv：1→64 C，kernel=4，stride=2，padding=1, 50→25
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1), nn.GroupNorm(16,128), nn.ReLU(True), ResBlock(128)
        )# conv：64→128 C，kernel=4，stride=2，padding=1, 25→12
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, 4, 2, 1), nn.GroupNorm(32,256), nn.ReLU(True), ResBlock(256)
        )# conv：128→256 C，kernel=4，stride=2，padding=1, 12→6
        feat_dim = 256 * 6 * 6
        self.fc_mu     = nn.Linear(feat_dim + label_dim, latent_dim)
        self.fc_logvar = nn.Linear(feat_dim + label_dim, latent_dim)
        self.fc_dec    = nn.Linear(latent_dim + label_dim, feat_dim)
        self.classifier = nn.Linear(latent_dim, num_groups)
        self.dec3 = nn.Sequential(nn.Conv2d(256,128,3,1,1), nn.GroupNorm(16,128), nn.ReLU(True), ResBlock(128))
        self.dec2 = nn.Sequential(nn.Conv2d(128,64,3,1,1),  nn.GroupNorm(8,64),  nn.ReLU(True), ResBlock(64))
        self.dec1 = nn.Sequential(
            nn.Conv2d(64,32,3,1,1), nn.GroupNorm(4,32), nn.ReLU(True), ResBlock(32),
            nn.Conv2d(32,img_channels,3,1,1), nn.Sigmoid()
        )

    def encode(self, x, y):
        x3 = self.enc3(self.enc2(self.enc1(x)))
        flat = x3.view(x3.size(0), -1)
        h_cond = torch.cat([flat, y], dim=1)
        mu     = self.fc_mu(h_cond)
        logvar = self.fc_logvar(h_cond)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z, y):
        zc = torch.cat([z, y], dim=1)
        h  = self.fc_dec(zc).view(-1, 256, 6, 6)
        d  = F.interpolate(self.dec3(h), size=12, mode='nearest')
        d  = F.interpolate(self.dec2(d), size=25, mode='nearest')
        return F.interpolate(self.dec1(d), size=50, mode='nearest')

    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, y)
        cls_logits = self.classifier(mu)
        return recon, mu, logvar, cls_logits

# ───── Training with Early Stopping ─────
def main():
    CACHE_FILE = r"E:\Project_CNN\2_Pack\data_cache.pt"
    CHECKPOINT_DIR = r"E:\Project_CNN\v_cache"; os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    BATCH_SIZE, NUM_EPOCHS = 64, 50
    LR, WD = 1e-4, 1e-5
    beta_start, beta_end = 0.07, 0.085
    perc_weight, cls_weight = 0.3, 1.0
    patience = 5
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cache = torch.load(CACHE_FILE, map_location="cpu")
    masks = cache["masks"]
    labels = (masks.view(len(masks),-1).sum(1)>0).long().tolist()
    groups = cache.get("group_ids")
    num_groups = len(set(groups))
    ds = CachedImageDataset(CACHE_FILE, labels, groups)

    vgg = vgg16(weights=VGG16_Weights.DEFAULT).features[:8].to(DEVICE).eval()
    for p in vgg.parameters(): p.requires_grad=False

    n_total = len(ds)
    n_train = int(0.7 * n_total)
    n_val   = int(0.15 * n_total)
    n_test  = n_total - n_train - n_val
    train_ds, val_ds, test_ds = random_split(ds, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    model = CVAE_ResSimple_50(1, 128, 1, num_groups).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    scaler    = GradScaler()
    mse_loss  = nn.MSELoss()
    ce_loss   = nn.CrossEntropyLoss()
    get_beta  = lambda ep: beta_start + (beta_end-beta_start)*(ep-1)/(NUM_EPOCHS-1)

    best_val_mse = float('inf')
    epochs_no_improve = 0
    for epoch in range(1, NUM_EPOCHS+1):
        beta = get_beta(epoch)

        # Train
        model.train()
        correct_train, total_train = 0, 0
        for img, lbl, gid in train_loader:
            img, lbl, gid = img.to(DEVICE), lbl.to(DEVICE), gid.to(DEVICE)
            optimizer.zero_grad()
            with autocast():
                recon, mu, logvar, logits = model(img, lbl)
                rec  = mse_loss(recon, img)
                perc = F.mse_loss(vgg(recon.repeat(1,3,1,1)), vgg(img.repeat(1,3,1,1))) * perc_weight
                kl_pd = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
                kl_loss  = kl_pd.sum(1).mean()
                cls = ce_loss(logits, gid)
                loss = rec + perc + beta*kl_loss + cls_weight*cls
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            preds = logits.argmax(dim=1)
            correct_train += (preds == gid).sum().item()
            total_train   += gid.size(0)
        train_acc = correct_train/total_train

        # Validate
        model.eval()
        correct_val, total_val = 0, 0
        val_mse = 0.0
        kl_list = []
        with torch.no_grad():
            for img, lbl, gid in val_loader:
                img, lbl, gid = img.to(DEVICE), lbl.to(DEVICE), gid.to(DEVICE)
                recon, mu, logvar, logits = model(img, lbl)
                val_mse += mse_loss(recon, img).item() * img.size(0)
                kl_pd = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
                kl_list.append(kl_pd.cpu())
                preds = logits.argmax(dim=1)
                correct_val += (preds == gid).sum().item()
                total_val   += gid.size(0)
        val_acc = correct_val/total_val
        val_mse = val_mse/total_val
        kl_tensor  = torch.cat(kl_list, dim=0)
        kl_per_dim = kl_tensor.mean(0)
        kl_total   = kl_per_dim.sum().item()

        print(f"Epoch {epoch}/{NUM_EPOCHS} | Train Acc={train_acc:.4f} | Val Acc={val_acc:.4f} | Val MSE={val_mse:.6f} | Val KL={kl_total:.4f}")
        # 仅显示KL最大的5个维度
        topk = kl_per_dim.topk(5)
        idxs = topk.indices.cpu().numpy()
        vals = topk.values.cpu().numpy()
        print(f"    Top 5 KL dims (dim, kl): {list(zip(idxs.tolist(), vals.tolist()))}")

        # Early stopping
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "best_model.pth"))
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epoch(s)")
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

    # Test evaluation
    model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "best_model.pth")))
    correct_test, total_test, test_mse = 0, 0, 0.0
    with torch.no_grad():
        for img, lbl, gid in test_loader:
            img, lbl, gid = img.to(DEVICE), lbl.to(DEVICE), gid.to(DEVICE)
            recon, mu, logvar, logits = model(img, lbl)
            test_mse += mse_loss(recon, img).item() * img.size(0)
            preds = logits.argmax(dim=1)
            correct_test += (preds == gid).sum().item()
            total_test   += gid.size(0)
    test_acc = correct_test/total_test
    test_mse = test_mse/total_test
    print(f"Test Acc={test_acc:.4f} | Test MSE={test_mse:.6f}")

if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.freeze_support()
    main()
