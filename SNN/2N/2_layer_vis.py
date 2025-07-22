#!/usr/bin/env python3
import os
import re
import random
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import numpy as np

from c1_freeze import CNN  # 你原来的模型定义

# =========================
# 1. 读取要保留的第一层通道编号
# =========================
# 假设你把 layer.txt 放在与 model_path 同目录
model_path = r'E:\Project_SNV\1N\c1_cache\1_2_3_4\best_model.pth'
layer_file = os.path.join(os.path.dirname(model_path), 'layer.txt')
with open(layer_file, 'r', encoding='utf-8') as f:
    desired_channels = [int(x) for x in re.findall(r'\d+', f.read())]

# =========================
# 2. 屏蔽并冻结第一层 Conv+BN
# =========================
def mask_and_freeze_conv1(model: nn.Module, desired: list[int]):
    # 首两个模块是 Conv2d 和 BatchNorm2d
    conv1, bn1 = model.model[0], model.model[1]
    out_ch = conv1.out_channels
    # 构造掩码，只有 desired 通道为 1
    mask = torch.zeros((out_ch,1,1,1), device=conv1.weight.device)
    mask[desired] = 1.0
    with torch.no_grad():
        conv1.weight.mul_(mask)
        if conv1.bias is not None:
            conv1.bias.mul_(mask.view(-1))
    # 冻结它们
    for p in conv1.parameters(): p.requires_grad = False
    for p in bn1.parameters():   p.requires_grad = False
    bn1.eval()

# =========================
# 3. 可视化函数（保持原样）
# =========================
def visualize_feature_and_heatmaps(
    model: nn.Module,
    img_path: str,
    layer_idx: int = 0,
    in_shape: tuple = (50,50),
    device: torch.device = torch.device('cpu'),
    n_cols: int = 4
):
    # 1) 读图 & 预处理
    img = Image.open(img_path).convert('L')
    tfm = transforms.Compose([
        transforms.Resize(in_shape),
        transforms.ToTensor()
    ])
    x = tfm(img).unsqueeze(0).to(device)

    # 2) 钩子取出指定卷积层输出
    convs = [m for m in model.model if isinstance(m, nn.Conv2d)]
    conv = convs[layer_idx]
    feats = {}
    def hook_fn(m, i, o): 
        feats['maps'] = o.detach().cpu()[0]
    h = conv.register_forward_hook(hook_fn)

    model.eval()
    with torch.no_grad():
        _ = model(x)
    h.remove()

    fmap = feats['maps']            # Tensor (C, Hf, Wf)
    n_maps, Hf, Wf = fmap.shape
    n_rows = (n_maps + n_cols - 1) // n_cols

    # 3) 原图展示
    plt.figure(figsize=(3,3))
    plt.imshow(img, cmap='gray')
    plt.title('Original Patch')
    plt.axis('off')
    plt.show(block=False)

    # 4) 灰度特征图
    plt.figure(figsize=(n_cols*2, n_rows*2))
    for i in range(n_maps):
        ax = plt.subplot(n_rows, n_cols, i+1)
        ax.imshow(fmap[i], cmap='gray')
        ax.axis('off')
        ax.text(0.95, 0.05, str(i),
                color='white', fontsize=8,
                va='bottom', ha='right',
                transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.5))
    plt.suptitle(f'Layer {layer_idx} Feature Maps (Gray)')
    plt.tight_layout()
    plt.show(block=False)

    # 5) 热力图 + colorbar
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*2, n_rows*2))
    axes = axes.flatten()
    for i in range(n_maps):
        axes[i].imshow(fmap[i], cmap='jet')
        axes[i].axis('off')
        axes[i].text(0.95, 0.05, str(i),
                     color='white', fontsize=8,
                     va='bottom', ha='right',
                     transform=axes[i].transAxes,
                     bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.5))
    for j in range(n_maps, len(axes)):
        axes[j].axis('off')
    plt.suptitle(f'Layer {layer_idx} Feature Maps (Heat)')
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    mappable = plt.cm.ScalarMappable(cmap='jet')
    mappable.set_array(fmap.numpy())
    fig.colorbar(mappable, cax=cbar_ax)
    fig.tight_layout(rect=[0,0,0.9,1])
    plt.show(block=False)

    # 防止脚本立即退出
    plt.pause(0.1)
    input("按回车键关闭所有窗口…")

# =========================
# 4. 主程序
# =========================
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    in_shape = (50,50)

    # 1) 载入 checkpoint 并自动推断 n_classes
    state_dict = torch.load(model_path, map_location='cpu')
    linear_keys = [k for k,v in state_dict.items() if k.endswith('weight') and v.dim()==2]
    last_lin_key = sorted(linear_keys)[-1]
    n_classes = state_dict[last_lin_key].shape[0]
    print(f"Detected {n_classes} classes from checkpoint key '{last_lin_key}'")

    # 2) 构建模型并加载权重
    model = CNN(in_shape=in_shape, n_classes=n_classes).to(device)
    model.load_state_dict(state_dict)
    # 屏蔽并冻结第一层
    mask_and_freeze_conv1(model, desired_channels)

    # 3) 指定要可视化的 patch 路径 & 卷积层索引
    img_path = r'E:\Project_SNV\0S\6_patch\1\0_2_2.png'
    layer_idx = 1   # 第一层

    # 4) 可视化
    visualize_feature_and_heatmaps(
        model, img_path,
        layer_idx=layer_idx,
        in_shape=in_shape,
        device=device
    )
