import os
import shutil
import random
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import numpy as np

# =========================
# 1. 参数配置
# =========================
src_root  = r'E:\Project_CNN\0_image'        # 原始 high/low 文件夹
dst_root  = r'E:\Project_CNN\1_training'     # 输出 train/val/test 根目录
data_root = r'E:\Project_CNN\c_cache'        # 权重要保存到这里
splits    = (0.7, 0.15, 0.15)
seed      = 42
batch_size = 32
num_workers = 4
in_shape   = (50, 50)

# ← 通过这个开关控制是否重新拆分
do_split = False  # True：先执行 split_dataset，False：跳过拆分

# =========================
# 2. 数据拆分函数（如需）
# =========================
def split_dataset(src_root, dst_root, classes=('high','low'),
                  splits=(0.7,0.15,0.15), seed=42):
    random.seed(seed)
    # 如果目标目录已存在，先清空
    if os.path.exists(dst_root):
        shutil.rmtree(dst_root)
    for cls in classes:
        src_dir = os.path.join(src_root, cls)
        imgs = [f for f in os.listdir(src_dir)
                if f.lower().endswith(('.png','.jpg','.jpeg'))]
        random.shuffle(imgs)
        n = len(imgs)
        n1, n2 = int(splits[0]*n), int((splits[0]+splits[1])*n)
        groups = {
            'train': imgs[:n1],
            'val':   imgs[n1:n2],
            'test':  imgs[n2:]
        }
        for grp, files in groups.items():
            for fn in files:
                dst_dir = os.path.join(dst_root, grp, cls)
                os.makedirs(dst_dir, exist_ok=True)
                shutil.copy(
                    os.path.join(src_dir, fn),
                    os.path.join(dst_dir, fn)
                )
    print('Dataset split complete.')

# =========================
# 3. CNN 定义
# =========================
class CNN(nn.Module):
    def __init__(self, in_shape, n_classes,
                 n_conv=2, k_sizes=[3,3,3], n_fc=1, fc_units=[128,0,0],
                 conv_drop=[0.2,0.2,0.0], fc_drop=[0.5,0.0,0.0]):
        super().__init__()
        c_list = [1,16,32,64]
        layers = []
        for i in range(n_conv):
            layers += [
                nn.Conv2d(c_list[i], c_list[i+1],
                          kernel_size=k_sizes[i], padding='same'),
                nn.BatchNorm2d(c_list[i+1]),
                nn.ELU(),
                nn.Dropout2d(conv_drop[i]),
                nn.MaxPool2d(2)
            ]
        layers.append(nn.Flatten())
        # 计算 flatten 后的维度
        with torch.no_grad():
            dummy = torch.zeros(1,1,*in_shape)
            flat_dim = nn.Sequential(*layers)(dummy).shape[1]
        fc_dims = [flat_dim] + fc_units[:n_fc]
        for i in range(n_fc):
            layers += [
                nn.Linear(fc_dims[i], fc_dims[i+1]),
                nn.BatchNorm1d(fc_dims[i+1]),
                nn.ELU(),
                nn.Dropout(fc_drop[i])
            ]
        layers += [nn.Linear(fc_dims[n_fc], n_classes)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# =========================
# 4. EarlyStopping
# =========================
class EarlyStopping:
    def __init__(self, patience=5, delta=0.0):
        self.patience = patience
        self.delta = delta
        self.best_loss = np.inf
        self.counter = 0
        self.best_state = None

    def step(self, loss, model):
        if loss < self.best_loss - self.delta:
            self.best_loss = loss
            self.counter = 0
            self.best_state = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
        return self.counter >= self.patience

    def restore(self, model):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)

# =========================
# 5. 训练与评估（含 EarlyStopping）
# =========================
def train_and_evaluate(data_root, device):
    tfm = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize(in_shape),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_ds = datasets.ImageFolder(
        os.path.join(data_root,'train'), transform=tfm)
    val_ds   = datasets.ImageFolder(
        os.path.join(data_root,'val'), transform=tfm)
    test_ds  = datasets.ImageFolder(
        os.path.join(data_root,'test'), transform=tfm)

    train_dl = DataLoader(train_ds, batch_size=batch_size,
                          shuffle=True, num_workers=num_workers)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size,
                          shuffle=False, num_workers=num_workers)
    test_dl  = DataLoader(test_ds,  batch_size=batch_size,
                          shuffle=False, num_workers=num_workers)

    model     = CNN(in_shape=in_shape, n_classes=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3)
    early     = EarlyStopping(patience=5)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, 31):
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
        print(f'Epoch {epoch:02d}  Val Loss={val_loss:.4f}  Acc={val_acc:.4f}')
        scheduler.step(val_loss)
        if early.step(val_loss, model):
            print('Early stopping')
            break

    # 恢复最优模型并做测试
    early.restore(model)
    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in test_dl:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1)==y).sum().item()
    print(f'Test Accuracy: {correct/len(test_dl.dataset):.4f}')
    return model

# =========================
# 6. 主程序：根据 do_split 决定是否拆分 → 训练 → 保存
# =========================
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if do_split:
        split_dataset(src_root, dst_root, splits=splits, seed=seed)
    else:
        print('Skipping dataset split, assuming data already prepared.')

    model = train_and_evaluate(dst_root, device)

    # 最后把最佳模型权重存到 data_root
    os.makedirs(data_root, exist_ok=True)
    save_path = os.path.join(data_root, 'best_CNN.pth')
    torch.save(model.state_dict(), save_path)
    print(f'Saved best model to {save_path}')
