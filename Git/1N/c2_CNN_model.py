#!/usr/bin/env python3
import os, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy
from torch.utils.data import Dataset, DataLoader, Subset
import matplotlib.pyplot as plt

# =========================
# 1. Global Deterministic Settings
# =========================
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

# =========================
# 2. Configuration
# =========================
g_name = '1_2_3_4'
cache_file  = rf'E:\Project_SNV\1N\1_Pack\{g_name}\data_cache.pt'
data_save   = rf'E:\Project_SNV\1N\c2_cache\{g_name}'
splits      = (0.7, 0.15, 0.15)
batch_size  = 32
num_workers = 0
in_shape    = (50, 50)
device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
use_brightness_augment = True  # enable brightness perturbation augmentation

# =========================
# 3. Dataset
# =========================
class CachedImageDataset(Dataset):
    def __init__(self, cache_path, transform=None, return_path=False):
        cache = torch.load(cache_path, map_location='cpu')
        self.images      = cache['images']
        self.group_ids   = torch.tensor(cache['group_ids'], dtype=torch.long)
        self.paths       = cache.get('paths', [None] * len(self.images))
        self.transform   = transform
        self.return_path = return_path

    def _to_tensor_chw1(self, x):
        # Convert to tensor and enforce [1,H,W] shape without changing values
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        x = x.float()

        while x.ndim > 3 and x.shape[0] == 1:
            x = x.squeeze(0)

        if x.ndim == 2:  # [H,W] -> [1,H,W]
            x = x.unsqueeze(0)
        elif x.ndim == 3:
            if x.shape[0] == 1:  # [1,H,W] -> OK
                pass
            elif x.shape[-1] == 1:  # [H,W,1] -> [1,H,W]
                x = x.permute(2, 0, 1)
            else:
                raise ValueError(f"Unexpected shape {tuple(x.shape)}; expected [H,W], [1,H,W], or [H,W,1].")
        else:
            raise ValueError(f"Unexpected ndim {x.ndim} for shape {tuple(x.shape)}")

        return x

    def __len__(self):
        return len(self.group_ids)

    def __getitem__(self, idx):
        x = self._to_tensor_chw1(self.images[idx])   # enforce [1,H,W]
        if self.transform:
            x = self.transform(x)
            x = self._to_tensor_chw1(x)              # recheck shape after transform
        y = self.group_ids[idx]
        p = self.paths[idx]
        if self.return_path:
            return x, y, p
        else:
            return x, y

# =========================
# 4. Random Brightness Perturbation Augmentation
# =========================
class RandomIlluminationGradient:
    def __init__(self, alpha=0.5, global_shift=0.3):
        self.alpha = alpha
        self.global_shift = global_shift

    def __call__(self, img):
        import torch.nn.functional as F
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img)
        img = img.float()

        if img.ndim == 2:
            img = img.unsqueeze(0)
        elif img.ndim == 3 and img.shape[0] != 1:
            raise ValueError(f"Unexpected image shape {img.shape}")

        _, h, w = img.shape
        dyn = (img.max() - img.min()).clamp(min=1e-6)

        direction = random.choice(['horizontal', 'vertical', 'diag', 'random'])
        if direction == 'horizontal':
            gradient = torch.linspace(0, 1, steps=w).unsqueeze(0).repeat(h, 1)
        elif direction == 'vertical':
            gradient = torch.linspace(0, 1, steps=h).unsqueeze(1).repeat(1, w)
        elif direction == 'diag':
            gx = torch.linspace(0, 1, steps=w).unsqueeze(0).repeat(h, 1)
            gy = torch.linspace(0, 1, steps=h).unsqueeze(1).repeat(1, w)
            gradient = (gx + gy) / 2
        else:
            noise = torch.rand(max(1, h//8), max(1, w//8))
            gradient = F.interpolate(noise.unsqueeze(0).unsqueeze(0),
                                      size=(h, w), mode='bilinear', align_corners=False).squeeze(0)

        gradient = (gradient - gradient.mean())
        gradient = gradient / (gradient.abs().max().clamp(min=1e-6))
        gradient = gradient * (self.alpha * dyn)
        gradient = gradient.unsqueeze(0)

        shift = (2 * torch.rand(1).item() - 1) * (self.global_shift * dyn)
        return img + gradient + shift

# =========================
# 5. CNN Definition
# =========================
class CNN(nn.Module):
    def __init__(self, in_shape, n_classes,
                 n_conv=2, k_sizes=[3,3], n_fc=1, fc_units=[128],
                 conv_drop=[0.2,0.2], fc_drop=[0.5]):
        super().__init__()
        c_list = [1,16,32,64]
        layers = []
        for i in range(n_conv):
            layers += [
                nn.Conv2d(c_list[i], c_list[i+1], kernel_size=k_sizes[i], padding='same'),
                nn.BatchNorm2d(c_list[i+1]),
                nn.ELU(),
                nn.Dropout2d(conv_drop[i]),
                nn.MaxPool2d(2)
            ]
        layers.append(nn.Flatten())
        with torch.no_grad():
            dummy = torch.zeros(1,1,*in_shape)
            flat_dim = nn.Sequential(*layers)(dummy).shape[1]
        fc_dims = [flat_dim] + fc_units
        for i in range(n_fc):
            layers += [
                nn.Linear(fc_dims[i], fc_dims[i+1]),
                nn.BatchNorm1d(fc_dims[i+1]),
                nn.ELU(),
                nn.Dropout(fc_drop[i])
            ]
        layers += [nn.Linear(fc_dims[-1], n_classes)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.float()
        h = self.model[:-1](x)
        logits = self.model[-1](h)
        return logits, h

# =========================
# 6. EarlyStopping
# =========================
class EarlyStopping:
    def __init__(self, patience=7, delta=0.0):
        self.patience   = patience
        self.delta      = delta
        self.best_loss  = np.inf
        self.counter    = 0
        self.best_state = None

    def step(self, loss, model):
        if loss < self.best_loss - self.delta:
            self.best_loss  = loss
            self.counter    = 0
            self.best_state = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
        return self.counter >= self.patience

    def restore(self, model):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)

# =========================
# 7. Training and Evaluation
# =========================
def train_and_evaluate(cache_path):
    ds_base = CachedImageDataset(cache_path, transform=None)
    N = len(ds_base)
    n1 = int(splits[0]*N)
    n2 = int((splits[0]+splits[1])*N)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(N, generator=g).tolist()
    idx_train, idx_val, idx_test = perm[:n1], perm[n1:n2], perm[n2:]

    train_aug = RandomIlluminationGradient(alpha=0.2, global_shift=0.2) if use_brightness_augment else None
    ds_train = Subset(CachedImageDataset(cache_path, transform=train_aug), idx_train)
    ds_val   = Subset(CachedImageDataset(cache_path, transform=None), idx_val)
    ds_test  = Subset(CachedImageDataset(cache_path, transform=None), idx_test)

    train_dl = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_dl   = DataLoader(ds_val,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_dl  = DataLoader(ds_test,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    num_groups = len(set(torch.load(cache_path)['group_ids']))
    model = CNN(in_shape=in_shape, n_classes=num_groups).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=7)
    early = EarlyStopping(patience=15)
    criterion = nn.CrossEntropyLoss()

    epochs, val_losses, val_accs = [], [], []

    for epoch in range(1, 200):
        model.train()
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, _ = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss, correct = 0.0, 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                logits, _ = model(x)
                val_loss += criterion(logits, y).item() * x.size(0)
                correct  += (logits.argmax(1)==y).sum().item()
        val_loss /= len(val_dl.dataset)
        val_acc   = correct / len(val_dl.dataset)
        print(f'Epoch {epoch:02d}  Val Loss={val_loss:.4f}  Val Acc={val_acc:.4f}')

        epochs.append(epoch)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        scheduler.step(val_loss)
        if early.step(val_loss, model):
            print('Early stopping')
            break

    early.restore(model)
    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in test_dl:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            correct += (logits.argmax(1)==y).sum().item()
    test_acc = correct / len(test_dl.dataset)
    print(f'Test Accuracy: {test_acc:.4f}')

    os.makedirs(data_save, exist_ok=True)
    plt.figure()
    plt.plot(epochs, val_losses, marker='o')
    plt.xlabel('Epoch'); plt.ylabel('Validation Loss')
    plt.title('Validation Loss vs. Epoch')
    plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(data_save, 'val_loss_curve.png')); plt.close()

    plt.figure()
    plt.plot(epochs, val_accs, marker='o')
    plt.xlabel('Epoch'); plt.ylabel('Validation Accuracy')
    plt.title('Validation Accuracy vs. Epoch')
    plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(data_save, 'val_accuracy_curve.png')); plt.close()

    return model

# =========================
# 8. Main
# =========================
if __name__ == '__main__':
    os.makedirs(data_save, exist_ok=True)
    best_model = train_and_evaluate(cache_file)
    save_path = os.path.join(data_save, 'best_model.pth')
    torch.save(best_model.state_dict(), save_path)
    print(f'Saved best model to {save_path}')
