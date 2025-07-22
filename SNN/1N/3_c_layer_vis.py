import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import numpy as np
from c2_CNN_model import CNN


# --- 可视化函数 ---
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
    def hook_fn(m, i, o): feats['maps'] = o.detach().cpu()[0]
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
                bbox=dict(boxstyle="round,pad=0.2", fc="black", ec="none", alpha=0.5))
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
                     bbox=dict(boxstyle="round,pad=0.2", fc="black", ec="none", alpha=0.5))
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


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    in_shape = (50,50)

    # 1) 载入 checkpoint 并自动推断 n_classes
    ckpt_path = r'E:\Project_SNV\1N\c1_cache\1_2_3_4\best_model.pth'
    state_dict = torch.load(ckpt_path, map_location='cpu')
    # 找出所有 Linear 层的 weight（二维张量）
    linear_keys = [k for k,v in state_dict.items() if k.endswith('weight') and v.dim()==2]
    last_lin_key = sorted(linear_keys)[-1]
    n_classes = state_dict[last_lin_key].shape[0]
    print(f"Detected {n_classes} classes from checkpoint key '{last_lin_key}'")

    # 2) 构建模型并加载权重
    model = CNN(in_shape=in_shape, n_classes=n_classes)
    model.load_state_dict(state_dict)
    model.to(device)

    # 3) 指定要可视化的 patch 路径 & 卷积层索引
    img_path = r'E:\Project_SNV\0S\6_patch\1\0_2_2.png'
    layer_idx = 1   # 0 表示第一层 Conv2d，1 表示第二层，以此类推

    # 4) 可视化
    visualize_feature_and_heatmaps(
        model, img_path,
        layer_idx=layer_idx,
        in_shape=in_shape,
        device=device
    )
