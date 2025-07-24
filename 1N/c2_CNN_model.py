#!/usr/bin/env python3
import os, random, cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from sklearn.decomposition import PCA
from scipy.stats import pearsonr

# =========================
# 1. 全局确定性设置
# =========================
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# =========================
# 2. 参数配置
# =========================
cache_file  = r'E:\Project_SNV\1N\1_Pack\h_c_l\data_cache.pt'
data_save   = r'E:\Project_SNV\1N\c1_cache'
splits      = (0.7, 0.15, 0.15)
batch_size  = 32
num_workers = 0    # 禁用并行 worker 以提升确定性
in_shape    = (50, 50)
device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =========================
# 3. 自定义 Dataset
# =========================
class CachedImageDataset(Dataset):
    def __init__(self, cache_path, transform=None, return_path=False):
        cache = torch.load(cache_path, map_location='cpu')
        self.images      = cache['images']         # Tensor [N,1,50,50]
        self.group_ids   = torch.tensor(cache['group_ids'], dtype=torch.long)
        # 如果 data_cache.pt 里有 paths 字段，就拿出来，否则全填 None
        self.paths       = cache.get('paths', [None] * len(self.images))
        self.transform   = transform
        self.return_path = return_path

    def __len__(self):
        return len(self.group_ids)

    def __getitem__(self, idx):
        x = self.images[idx]
        if self.transform:
            x = self.transform(x)
        y = self.group_ids[idx]
        p = self.paths[idx]
        if self.return_path:
            return x, y, p
        else:
            return x, y


# =========================
# 4. CNN 定义
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
        # 先把 x 送到倒数第二层，得到隐藏向量 h
        h = self.model[:-1](x)
        # 再通过最后一层得到 logits
        logits = self.model[-1](h)
        return logits, h

# =========================
# 5. EarlyStopping
# =========================
class EarlyStopping:
    def __init__(self, patience=7, delta=0.0):
        self.patience = patience
        self.delta    = delta
        self.best_loss= np.inf
        self.counter  = 0
        self.best_state = None

    def step(self, loss, model):
        if loss < self.best_loss - self.delta:
            self.best_loss = loss
            self.counter   = 0
            self.best_state= copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
        return self.counter >= self.patience

    def restore(self, model):
        if self.best_state:
            model.load_state_dict(self.best_state)

# =========================
# 6. 训练与评估函数（含PC1–Laplacian监控）
# =========================
def train_and_evaluate(cache_path):
    transform = None
    ds = CachedImageDataset(cache_path, transform=transform)
    N = len(ds)
    n1 = int(splits[0]*N)
    n2 = int((splits[0]+splits[1])*N)
    train_ds, val_ds, test_ds = random_split(
        ds, [n1, n2-n1, N-n2], generator=torch.Generator().manual_seed(seed)
    )

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True)
    test_dl  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True)
    monitor_dl = DataLoader(train_ds, batch_size=batch_size,shuffle=False, 
                          num_workers=num_workers, pin_memory=True)
    num_groups = len(set(torch.load(cache_path)['group_ids']))
    model = CNN(in_shape=in_shape, n_classes=num_groups).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=7)
    early = EarlyStopping(patience=7)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, 51):
        # --- 训练 ---
        model.train()
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, _ = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        # --- 验证 ---
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

        # --- 监控 PC1–Laplacian 相关度 ---
        all_feats = []
        all_laps  = []
        with torch.no_grad():
            for x, _ in monitor_dl:
                x = x.to(device)
                # 特征向量是除去最后一层的输出
                feat = model.model[:-1](x)
                all_feats.append(feat.cpu().numpy())
                imgs = (x.cpu().numpy() * 255).astype(np.uint8)
                for im in imgs:
                    gray = im.squeeze()
                    lap = cv2.Laplacian(gray, cv2.CV_64F).var()
                    all_laps.append(lap)
        F = np.vstack(all_feats)
        pc1 = PCA(n_components=1, random_state=seed).fit_transform(F).squeeze()
        r, _ = pearsonr(pc1, np.array(all_laps))
        print(f'Epoch {epoch:02d}  ▶ PC1–Lap r = {r:.4f}')

        scheduler.step(val_loss)
        if early.step(val_loss, model):
            print('Early stopping')
            break

    # --- 恢复 & 测试 ---
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

    return model

# =========================
# 7. 主程序
# =========================
if __name__ == '__main__':
    os.makedirs(data_save, exist_ok=True)
    best_model = train_and_evaluate(cache_file)
    save_path = os.path.join(data_save, 'best_model.pth')
    torch.save(best_model.state_dict(), save_path)
    print(f'Saved best model to {save_path}')
