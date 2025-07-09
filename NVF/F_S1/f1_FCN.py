import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm

# =========================
# 1. 显式配置（无需命令行输入）
# =========================
SEED = 42
CACHE_FILE = r'E:\Project_CNN\2_Pack\data_cache.pt'
OUTPUT_DIR = r'E:\Project_CNN\f_cache'
BATCH_SIZE = 32
NUM_WORKERS = 0
EPOCHS = 50
PATIENCE = 7
SPLIT_TRAIN = 0.70
SPLIT_VAL = 0.15
LEARNING_RATE = 1e-3
IN_SIZE = (50, 50)  # H, W

# =========================
# 2. 确定性设置
# =========================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

# =========================
# 3. 自定义 Dataset
# =========================
class CachedImageDataset(Dataset):
    def __init__(self, cache_path, transform=None):
        data = torch.load(cache_path, map_location='cpu')
        self.images = data['images']       # [N,1,H,W]
        self.labels = torch.tensor(data['group_ids'], dtype=torch.long)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]            # [1,H,W]
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

# =========================
# 4. 模型定义（仅分类）
# =========================
class CNNClassifier(nn.Module):
    def __init__(self, in_shape, n_classes):
        super().__init__()
        H, W = in_shape
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Conv2d(32, n_classes, 1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        logits = self.classifier(x)
        return logits

# =========================
# 5. 训练/验证函数
# =========================
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
    return running_loss / len(loader.dataset)


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct = 0.0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            total_loss += criterion(logits, labels).item() * imgs.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
    avg_loss = total_loss / len(loader.dataset)
    acc = correct / len(loader.dataset)
    return avg_loss, acc

# =========================
# 6. 主流程
# =========================
def main():
    set_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Normalize((0.5,), (0.5,))
    dataset = CachedImageDataset(CACHE_FILE, transform=transform)
    N = len(dataset)
    n_train = int(SPLIT_TRAIN * N)
    n_val = int((SPLIT_TRAIN + SPLIT_VAL) * N) - n_train
    n_test = N - n_train - n_val
    train_ds, val_ds, test_ds = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(SEED)
    )
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=NUM_WORKERS, pin_memory=True)

    labels = torch.load(CACHE_FILE, map_location='cpu')['group_ids']
    n_classes = len(set(labels))
    model = CNNClassifier(IN_SIZE, n_classes).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    criterion = nn.CrossEntropyLoss()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ckpt_path = os.path.join(OUTPUT_DIR, 'best_model.pth')

    best_val_loss = float('inf')
    patience_cnt = 0
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_dl, optimizer, criterion, device)
        val_loss, val_acc = eval_epoch(model, val_dl, criterion, device)
        scheduler.step(val_loss)

        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_cnt = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print("Early stopping triggered.")
                break

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    test_loss, test_acc = eval_epoch(model, test_dl, criterion, device)
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

if __name__ == '__main__':
    main()
