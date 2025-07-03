import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import numpy as np

# --- 1. 定义 CNN 模型类 (同训练脚本中) ---
class CNN(nn.Module):
    def __init__(self, in_shape, n_classes,
                 n_conv=2, k_sizes=[3,3,3], n_fc=1, fc_units=[128,0,0],
                 conv_drop=[0.2,0.2,0.0], fc_drop=[0.5,0.0,0.0]):
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

    def forward(self, x): return self.model(x)


# --- 2. 同时可视化原图、特征图和热力图（含 colorbar） ---
def visualize_feature_and_heatmaps(
    model: nn.Module,
    img_path: str,
    layer_idx: int = 1,
    in_shape: tuple = (50,50),
    device: torch.device = torch.device('cpu'),
    n_cols: int = 4
):
    # 1) 读图 & 预处理
    img = Image.open(img_path).convert('L')
    tfm = transforms.Compose([
        transforms.Resize(in_shape),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
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
    n_rows = (n_maps + n_cols - 1)//n_cols

    # 3) 原图展示
    plt.figure(figsize=(3,3))
    plt.imshow(img, cmap='gray')
    plt.title('Original Patch')
    plt.axis('off')
    plt.show()

    # 4) 窗口一：灰度特征图
    fig1 = plt.figure(figsize=(n_cols*2, n_rows*2))
    for i in range(n_maps):
        ax = fig1.add_subplot(n_rows, n_cols, i+1)
        ax.imshow(fmap[i], cmap='gray')
        ax.axis('off')
    fig1.suptitle(f'Layer {layer_idx} Feature Maps (Gray)')
    fig1.tight_layout()
    plt.show()

    # 5) 窗口二：伪彩色热力图 + 独立 colorbar
    fig2, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*2, n_rows*2))
    axes = axes.flatten()
    for i in range(n_maps):
        axes[i].imshow(fmap[i], cmap='jet')
        axes[i].axis('off')
    for j in range(n_maps, len(axes)):
        axes[j].axis('off')
    fig2.suptitle(f'Layer {layer_idx} Feature Maps (Heat)')
    # 新增独立 colorbar 区域
    cbar_ax = fig2.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    mappable = plt.cm.ScalarMappable(cmap='jet')
    mappable.set_array(fmap.numpy())
    fig2.colorbar(mappable, cax=cbar_ax)
    fig2.tight_layout(rect=[0,0,0.9,1])
    plt.show()

# --- 3. 主程序 ---
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    in_shape = (50,50)
    model = CNN(in_shape=in_shape, n_classes=2)
    model.load_state_dict(torch.load(r'E:\Project_CNN\c_cache\best_CNN.pth', map_location=device))
    model.to(device)

    img_path = r'E:\Project_CNN\0_image\low\000_x150_y100.png'
    visualize_feature_and_heatmaps(model, img_path,
                                   layer_idx=0, #0/1 0 is layer 1
                                   in_shape=in_shape,
                                   device=device)
