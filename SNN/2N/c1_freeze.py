#!/usr/bin/env python3
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

# =========================
# 1. 全局确定性设置
# =========================
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

# =========================
# 2. 参数配置
# =========================
file_name       = '1_2_3_4'
cache_file      = rf'E:\Project_SNV\1N\1_Pack\{file_name}\data_cache.pt'
pretrained_path = rf'E:\Project_SNV\1N\c1_cache\{file_name}\best_model.pth'
finetuned_save  = rf'E:\Project_SNV\2N\c2_cache\best_model.pth'
splits          = (0.7, 0.15, 0.15)
batch_size      = 32
num_workers     = 0
in_shape        = (50, 50)
device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# —— 在这里指定你要保留的第一层通道编号 ——#
desired_channels = [15]

# =========================
# 3. Dataset
# =========================
class CachedImageDataset(Dataset):
    def __init__(self, cache_path, transform=None):
        cache = torch.load(cache_path, map_location='cpu')
        self.images    = cache['images']           # Tensor [N,1,H,W]
        self.group_ids = torch.tensor(cache['group_ids'], dtype=torch.long)
        self.transform = transform

    def __len__(self):
        return len(self.group_ids)

    def __getitem__(self, idx):
        x = self.images[idx]
        if self.transform:
            x = self.transform(x)
        y = self.group_ids[idx]
        return x, y

# =========================
# 4. CNN 定义（与预训练脚本一致）
# =========================
class CNN(nn.Module):
    def __init__(self, in_shape, n_classes):
        super().__init__()
        c_list = [1,16,32,64]
        layers = []
        # 两个 conv block
        for i in range(2):
            layers += [
                nn.Conv2d(c_list[i], c_list[i+1], 3, padding='same'),
                nn.BatchNorm2d(c_list[i+1]), nn.ELU(),
                nn.Dropout2d(0.2), nn.MaxPool2d(2)
            ]
        layers.append(nn.Flatten())
        with torch.no_grad():
            dummy   = torch.zeros(1,1,*in_shape)
            flat_dim= nn.Sequential(*layers)(dummy).shape[1]
        # 全连接部分
        layers += [
            nn.Linear(flat_dim, 128),
            nn.BatchNorm1d(128), nn.ELU(), nn.Dropout(0.5),
            nn.Linear(128, n_classes)
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# =========================
# 5. 冻结+屏蔽第一层 & 重置第二层
# =========================
def apply_mask_and_reset(model, desired):
    # conv1 是 model.model[0], bn1=model.model[1]
    conv1, bn1 = model.model[0], model.model[1]
    out_ch     = conv1.out_channels
    # 构造 mask：只有 desired 通道保 1，其它都 0
    mask = torch.zeros((out_ch,1,1,1), device=conv1.weight.device)
    mask[desired] = 1.0
    with torch.no_grad():
        conv1.weight.mul_(mask)
        if conv1.bias is not None:
            conv1.bias.mul_(mask.view(-1))
    # 冻结 conv1 + bn1
    for p in conv1.parameters(): p.requires_grad = False
    for p in bn1.parameters():   p.requires_grad = False
    bn1.eval()

    # 重置第二层 conv2（在本 model 中是 model.model[5]）
    conv2 = model.model[5]
    with torch.no_grad():
        conv2.weight.zero_()
        if conv2.bias is not None:
            conv2.bias.zero_()

# =========================
# 6. 训练 & 微调函数（含 Test 评估）
# =========================
def train_and_finetune(cache_path):
    transform = None

    # Dataset + 划分
    ds = CachedImageDataset(cache_path, transform=transform)
    N  = len(ds)
    n1 = int(splits[0]*N)
    n2 = int((splits[0]+splits[1])*N)
    train_ds, val_ds, test_ds = random_split(
        ds, [n1, n2-n1, N-n2],
        generator=torch.Generator().manual_seed(seed)
    )

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True)

    # 模型 & 加载预训练
    num_groups = len(set(torch.load(cache_path)['group_ids']))
    model = CNN(in_shape=in_shape, n_classes=num_groups).to(device)
    state = torch.load(pretrained_path, map_location=device)
    model.load_state_dict(state)

    # 冻结屏蔽第一层 + 重置第二层
    apply_mask_and_reset(model, desired_channels)

    # 优化 & 损失
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable, lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
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

    # —— 在返回前做一次 Test 评估 —— #
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers, pin_memory=True)
    model.eval()
    test_correct = 0
    test_total   = 0
    with torch.no_grad():
        for x, y in test_dl:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            test_correct += (logits.argmax(1)==y).sum().item()
            test_total   += y.size(0)
    test_acc = test_correct / test_total
    print(f'Test Accuracy after finetuning: {test_acc:.4f}')

    return model

# =========================
# 7. 主程序：保存微调权重
# =========================
if __name__ == '__main__':
    os.makedirs(os.path.dirname(finetuned_save), exist_ok=True)
    model_ft = train_and_finetune(cache_file)
    torch.save(model_ft.state_dict(), finetuned_save)
    print(f'Saved finetuned model to {finetuned_save}')
