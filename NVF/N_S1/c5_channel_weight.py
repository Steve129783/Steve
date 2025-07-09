import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy

from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

# =========================
# 全局确定性设置
# =========================
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

# =========================
# 参数配置
# =========================
cache_file          = r'E:\Project_CNN\2_Pack\data_cache.pt'
pretrained_path     = r'E:\Project_CNN\c_cache\best_model.pth'
finetuned_save_path = r'E:\Project_CNN\c2_cache\best_model.pth'

splits      = (0.7, 0.15, 0.15)
batch_size  = 32
num_workers = 0
in_shape    = (50, 50)
device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Weight = 1
desired_channels = [11, 13]  # 要保留的第一层通道

# =========================
# Dataset
# =========================
class CachedImageDataset(Dataset):
    def __init__(self, cache_path, transform=None):
        cache = torch.load(cache_path, map_location='cpu')
        self.images    = cache['images']
        self.group_ids = torch.tensor(cache['group_ids'], dtype=torch.long)
        self.transform = transform

    def __len__(self):
        return len(self.group_ids)

    def __getitem__(self, idx):
        x = self.images[idx]
        if self.transform: x = self.transform(x)
        y = self.group_ids[idx]
        return x, y

# =========================
# CNN
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
                nn.Conv2d(c_list[i], c_list[i+1], k_sizes[i], padding='same'),
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
        layers.append(nn.Linear(fc_dims[-1], n_classes))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# =========================
# EarlyStopping
# =========================
class EarlyStopping:
    def __init__(self, patience=5, delta=0.0):
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
# 训练 & 微调函数
# =========================
def train_finetune():
    # 数据变换
    transform = transforms.Normalize((0.5,), (0.5,))

    # 加载并划分
    ds = CachedImageDataset(cache_file, transform)
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
    test_dl  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True)

    # 模型 & 预训练权重
    num_groups = len(set(torch.load(cache_file)['group_ids']))
    model = CNN(in_shape=in_shape, n_classes=num_groups).to(device)
    state = torch.load(pretrained_path, map_location=device)
    model.load_state_dict(state)

    # 冻结 & 屏蔽第一层
    # 假设 conv1 是 model.model[0]，desired_channels 已经定义好
    conv1 = model.model[0]
    out_ch = conv1.out_channels
    conv1_dev = conv1.weight.device

    # 1) 构造一个 [out_ch,1,1,1] 的乘子张量，默认 0.1
    mask = torch.full((out_ch, 1, 1, 1), Weight, device=conv1_dev)
    # 2) 把你要保留的通道乘子设为 1
    mask[desired_channels] = 1.0

    # 3) 按通道缩放 权重 和 bias
    with torch.no_grad():
        conv1.weight.mul_(mask)     # [out_ch, in_ch, k, k] 广播到每个 in_ch,k,k 元素
        if conv1.bias is not None:
            # bias shape [out_ch], 需 squeeze 掩码
            bias_mask = mask.view(out_ch)
            conv1.bias.mul_(bias_mask)

    # 4) 冻结 conv1
    for p in conv1.parameters():
        p.requires_grad = False


    # 重置第二层（保持可训练）
    conv2 = model.model[5]
    with torch.no_grad():
        conv2.weight.zero_()
        if conv2.bias is not None:
            conv2.bias.zero_()

    # 优化器只包含 Conv2 及之后
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable, lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    early     = EarlyStopping(patience=5)
    criterion = nn.CrossEntropyLoss()

    # 训练循环
    for epoch in range(1, 51):
        model.train()
        for x,y in train_dl:
            x,y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss, correct = 0.0, 0
        with torch.no_grad():
            for x,y in val_dl:
                x,y = x.to(device), y.to(device)
                logits = model(x)
                val_loss += criterion(logits, y).item()*x.size(0)
                correct  += (logits.argmax(1)==y).sum().item()
        val_loss /= len(val_dl.dataset)
        val_acc   = correct / len(val_dl.dataset)
        print(f'Epoch {epoch}  Val Loss={val_loss:.4f}  Val Acc={val_acc:.4f}')

        scheduler.step(val_loss)
        if early.step(val_loss, model):
            print("Early stopping")
            break

    # 恢复 & 测试
    early.restore(model)
    model.eval()
    correct = 0
    with torch.no_grad():
        for x,y in test_dl:
            x,y = x.to(device), y.to(device)
            correct += (model(x).argmax(1)==y).sum().item()
    print("Test Acc:", correct/len(test_dl.dataset))

    # 保存微调后的权重
    os.makedirs(os.path.dirname(finetuned_save_path), exist_ok=True)
    torch.save(model.state_dict(), finetuned_save_path)
    print("Saved finetuned model to", finetuned_save_path)

if __name__ == '__main__':
    train_finetune()
