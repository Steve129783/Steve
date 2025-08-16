#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import random
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn.functional as F
import matplotlib.pyplot as plt

# =========================
# 1. Global deterministic settings
# =========================
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# =========================
# 2. Configuration
# =========================
file_name       = '1_2_3_4'
cache_file      = rf'E:\Project_SNV\1N\1_Pack\{file_name}\data_cache.pt'
pretrained_full = rf'E:\Project_SNV\1N\c2_cache\{file_name}\best_model.pth'
splits          = (0.7, 0.15, 0.15)
batch_size      = 32
num_workers     = 0
in_shape        = (50, 50)
device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
max_epochs      = 200
cha             = 11
data_save       = rf'E:\Project_SNV\2N\c2_cache\{file_name}\{cha}'
keep_channels   = [cha]         # list(range(16)) or [cha]

# Data augmentation switch
use_brightness_augment = True  # True to enable brightness perturbation, False to disable

# =========================
# 3. Brightness perturbation augmentation
# =========================
class RandomIlluminationGradient:
    """Add low-frequency illumination perturbation with random direction and global shift."""
    def __init__(self, alpha=0.15, global_shift=0.3):
        self.alpha = alpha
        self.global_shift = global_shift

    def __call__(self, img):
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img)
        img = img.float()

        if img.ndim == 2:
            img = img.unsqueeze(0)  # [1,H,W]
        elif img.ndim != 3 or img.shape[0] != 1:
            raise ValueError(f"Unexpected image shape {img.shape}, expected single-channel.")

        _, h, w = img.shape

        # Random gradient direction
        direction = random.choice(['horizontal', 'vertical', 'diag', 'random'])
        if direction == 'horizontal':
            gradient = torch.linspace(0, 1, steps=w).unsqueeze(0).repeat(h, 1)
        elif direction == 'vertical':
            gradient = torch.linspace(0, 1, steps=h).unsqueeze(1).repeat(1, w)
        elif direction == 'diag':
            gx = torch.linspace(0, 1, steps=w).unsqueeze(0).repeat(h, 1)
            gy = torch.linspace(0, 1, steps=h).unsqueeze(1).repeat(1, w)
            gradient = (gx + gy) / 2
        else:  # Low-frequency random noise
            noise = torch.rand(max(1, h // 8), max(1, w // 8))
            gradient = F.interpolate(
                noise.unsqueeze(0).unsqueeze(0),
                size=(h, w), mode='bilinear', align_corners=False
            ).squeeze(0)

        # Scale to [-1,1]
        gradient = gradient * (2 * torch.rand(1).item() - 1)
        gradient = gradient.unsqueeze(0) * self.alpha

        # Add global brightness shift
        shift = (2 * torch.rand(1).item() - 1) * self.global_shift

        return torch.clamp(img + gradient + shift, 0, 1)

# =========================
# 4. Cached dataset definition
# =========================
class CachedImageDataset(Dataset):
    def __init__(self, cache_path, transform=None):
        data = torch.load(cache_path, map_location='cpu')
        self.images = data['images']
        self.labels = torch.tensor(data['group_ids'], dtype=torch.long)
        self.transform = transform

    def _to_chw1(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        x = x.float()

        # Remove redundant dimensions
        while x.ndim > 2 and x.shape[0] == 1:
            x = x.squeeze(0)

        if x.ndim == 2:  # [H,W] -> [1,H,W]
            x = x.unsqueeze(0)
        elif x.ndim == 3 and x.shape[0] != 1:
            raise ValueError(f"Unexpected shape {x.shape}, expected single-channel")
        return x

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self._to_chw1(self.images[idx])
        if self.transform:
            x = self.transform(x)
            x = self._to_chw1(x)  # ensure [1,H,W] again
        return x, self.labels[idx]

# =========================
# 5. CNN model definition
# =========================
class CNN(nn.Module):
    def __init__(self, in_shape, n_classes, keep_channels):
        super().__init__()
        # conv0 block
        self.conv0 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn0   = nn.BatchNorm2d(16)
        self.act0  = nn.ELU()
        self.drop0 = nn.Dropout2d(0.2)
        self.pool0 = nn.MaxPool2d(2)

        # Block1
        self.conv1 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.act1  = nn.ELU()
        self.drop1 = nn.Dropout2d(0.2)
        self.pool1 = nn.MaxPool2d(2)

        # FC layers
        flat_dim = (in_shape[0] // 4) * (in_shape[1] // 4) * 32
        self.fc1   = nn.Linear(flat_dim, 128)
        self.bn2   = nn.BatchNorm1d(128)
        self.act2  = nn.ELU()
        self.drop2 = nn.Dropout(0.5)

        # Classification head
        self.fc2   = nn.Linear(128, n_classes)

        # Channel mask: keep only given channels
        mask = torch.zeros(16, dtype=torch.float32)
        mask[keep_channels] = 1.0
        self.register_buffer('chan_mask', mask.view(1, -1, 1, 1))

    def forward(self, x):
        x = self.conv0(x)
        x = x * self.chan_mask
        x = self.bn0(x)
        x = self.act0(x)
        x = self.drop0(x)
        x = self.pool0(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.drop1(x)
        x = self.pool1(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.drop2(x)
        x = self.fc2(x)
        return x

# =========================
# 6. EarlyStopping utility
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
# 7. Training and evaluation pipeline
# =========================
def train_and_evaluate():
    # Generate split indices
    base_dataset = CachedImageDataset(cache_file, transform=None)
    N = len(base_dataset)
    n1 = int(splits[0] * N)
    n2 = int((splits[0] + splits[1]) * N)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(N, generator=g).tolist()
    idx_train, idx_val, idx_test = perm[:n1], perm[n1:n2], perm[n2:]

    # Datasets with different transforms
    train_aug = RandomIlluminationGradient(alpha=0.2, global_shift=0.2) if use_brightness_augment else None
    ds_train = Subset(CachedImageDataset(cache_file, transform=train_aug), idx_train)
    ds_val   = Subset(CachedImageDataset(cache_file, transform=None), idx_val)
    ds_test  = Subset(CachedImageDataset(cache_file, transform=None), idx_test)

    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(ds_val,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(ds_test,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Model initialization
    num_classes = len(set(base_dataset.labels.tolist()))
    model = CNN(in_shape, num_classes, keep_channels).to(device)

    # Load pretrained conv0
    full_ckpt = torch.load(pretrained_full, map_location=device)
    model.conv0.weight.data.copy_(full_ckpt['model.0.weight'])
    if 'model.0.bias' in full_ckpt:
        model.conv0.bias.data.copy_(full_ckpt['model.0.bias'])

    for p in model.conv0.parameters(): 
        p.requires_grad = False
    model.bn0.running_mean.zero_() 
    model.bn0.running_var.fill_(1.0)
    nn.init.ones_(model.bn0.weight)
    nn.init.zeros_(model.bn0.bias)
    model.bn0.weight.requires_grad=False
    model.bn0.bias.requires_grad=False

    # Initialize remaining layers
    for m in [model.conv1, model.fc1, model.fc2]:
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        nn.init.zeros_(m.bias)
    for m in [model.bn1, model.bn2]:
        nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
        m.running_mean.zero_(); m.running_var.fill_(1.0)

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=7
    )
    early_stopper = EarlyStopping(patience=15)
    criterion = nn.CrossEntropyLoss()

    val_losses = []
    val_accuracies = []

    # Training loop
    for epoch in range(1, max_epochs+1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_loss, correct = 0.0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                val_loss += criterion(logits, y).item() * x.size(0)
                correct  += (logits.argmax(1) == y).sum().item()
        val_loss /= len(val_loader.dataset)
        val_acc   = correct / len(val_loader.dataset)
        print(f"Epoch {epoch:02d} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        scheduler.step(val_loss)
        if early_stopper.step(val_loss, model):
            print("Early stopping triggered")
            break

    # Restore best model and test
    early_stopper.restore(model)
    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            correct += (logits.argmax(1) == y).sum().item()
    test_acc = correct / len(test_loader.dataset)
    print(f"Test Accuracy: {test_acc:.4f}")

    # Plot and save curves
    epochs = list(range(1, len(val_losses) + 1))
    os.makedirs(data_save, exist_ok=True)

    plt.figure(figsize=(6,4))
    plt.plot(epochs, val_losses, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss vs. Epoch')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(data_save, 'val_loss_curve.png'), dpi=300)
    plt.close()

    plt.figure(figsize=(6,4))
    plt.plot(epochs, val_accuracies, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.title('Validation Accuracy vs. Epoch')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(data_save, 'val_accuracy_curve.png'), dpi=300)
    plt.close()

    return model

# =========================
# 8. Main entry
# =========================
if __name__ == '__main__':
    os.makedirs(data_save, exist_ok=True)
    best_model = train_and_evaluate()
    out_path = os.path.join(data_save, f'best_model_{cha}.pth')
    torch.save(best_model.state_dict(), out_path)
    print(f'Model saved to {out_path}')
