import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy

from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image

# =========================
# 1. 全局确定性设置
# =========================
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# 确保 cudnn 确定性
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# =========================
# 2. 参数配置
# =========================
cache_file  = r'E:\Project_CNN\2_Pack\data_cache.pt'
data_save   = r'E:\Project_CNN\c_cache'     # 保存模型权重
splits      = (0.7, 0.15, 0.15)
batch_size  = 32
num_workers = 0    # 禁用并行 worker 以提升确定性
in_shape    = (50, 50)
device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =========================
# 3. 自定义 Dataset
# =========================
class CachedImageDataset(Dataset):
    def __init__(self, cache_path, transform=None):
        cache = torch.load(cache_path, map_location='cpu')
        self.images    = cache['images']         # Tensor [N,1,50,50]
        self.group_ids = torch.tensor(cache['group_ids'], dtype=torch.long)
        self.transform = transform

    def __len__(self):
        return len(self.group_ids)

    def __getitem__(self, idx):
        x = self.images[idx]    # Tensor[1,50,50], float32 in [0,1]
        if self.transform:
            x = self.transform(x)
        y = self.group_ids[idx]
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
        return self.model(x)

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
# 6. 训练与评估函数
# =========================
def train_and_evaluate(cache_path):
    # 数据转换：[0,1] -> [-1,1]
    transform = transforms.Normalize((0.5,), (0.5,))

    # Dataset + 固定划分
    ds = CachedImageDataset(cache_path, transform=transform)
    N = len(ds)
    n1 = int(splits[0]*N)
    n2 = int((splits[0]+splits[1])*N)
    train_ds, val_ds, test_ds = random_split(
        ds, [n1, n2-n1, N-n2], generator=torch.Generator().manual_seed(seed)
    )

    # DataLoader (num_workers=0) 保证确定性
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True)
    test_dl  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True)

    # 模型初始化
    num_groups = len(set(torch.load(cache_path)['group_ids']))
    model = CNN(in_shape=in_shape, n_classes=num_groups).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=7)
    early = EarlyStopping(patience=5)
    criterion = nn.CrossEntropyLoss()

    # 训练 + 验证
    for epoch in range(1, 51):
        model.train()
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

        # 验证
        model.eval()
        val_loss, correct = 0.0, 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                val_loss += criterion(logits, y).item() * x.size(0)
                correct  += (logits.argmax(1)==y).sum().item()
        val_loss /= len(val_dl.dataset)
        val_acc   = correct / len(val_dl.dataset)
        print(f'Epoch {epoch:02d}  Val Loss={val_loss:.4f}  Val Acc={val_acc:.4f}')

        scheduler.step(val_loss)
        if early.step(val_loss, model):
            print('Early stopping')
            break

    # 恢复并测试
    early.restore(model)
    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in test_dl:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1)==y).sum().item()
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
