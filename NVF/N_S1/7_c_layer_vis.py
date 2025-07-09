import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import numpy as np
from c5_channel_weight import CNN  # 同上模型定义

# =========================
# 全局设置
# =========================
desired_channels = [11, 13]
model_path       = r'E:\Project_CNN\c2_cache\best_model.pth'
in_shape         = (50,50)
device           = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Weight = 0.1

# 屏蔽第一层但不重置第二层
def apply_mask_conv1(model):
    conv1 = model.model[0]
    out_ch = conv1.out_channels
    undesired = list(set(range(out_ch)) - set(desired_channels))
    scale = Weight  # 0.1

    with torch.no_grad():
        # 正确的“弱化”方式：乘以一个系数
        conv1.weight[undesired].mul_(scale)
        if conv1.bias is not None:
            conv1.bias[undesired].mul_(scale)

    # 冻结第一层
    for p in conv1.parameters():
        p.requires_grad = False

# 可视化函数同之前，只在加载后调用 mask
def visualize_feature_and_heatmaps(model, img_path, layer_idx=1, n_cols=4):
    img = Image.open(img_path).convert('L')
    tfm = transforms.Compose([
        transforms.Resize(in_shape),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    x = tfm(img).unsqueeze(0).to(device)

    convs = [m for m in model.model if isinstance(m, nn.Conv2d)]
    conv = convs[layer_idx]
    feats = {}
    def hook_fn(m, i, o): feats['maps'] = o.detach().cpu()[0]
    h = conv.register_forward_hook(hook_fn)
    model.eval()
    with torch.no_grad(): _ = model(x)
    h.remove()

    fmap = feats['maps']
    n_maps, Hf, Wf = fmap.shape
    n_rows = (n_maps + n_cols-1)//n_cols

    # 原图
    plt.figure(figsize=(3,3))
    plt.imshow(img, cmap='gray'); plt.axis('off'); plt.title("Original")
    plt.show(block=False)

    # 灰度特征
    fig1 = plt.figure(figsize=(n_cols*2, n_rows*2))
    for i in range(n_maps):
        ax=fig1.add_subplot(n_rows, n_cols, i+1)
        ax.imshow(fmap[i], cmap='gray'); ax.axis('off')
        ax.text(0.02,0.98,str(i),color='white',fontsize=8,
                va='top',ha='left',transform=ax.transAxes,
                backgroundcolor='black')
    fig1.suptitle(f'Layer {layer_idx} Gray'); fig1.tight_layout(); plt.show(block=False)

    # 彩色热力
    # ——— 计算 alpha_vals ———
    # 这里举个例子：直接用你在 apply_mask_conv1 里用的同一份 Weight
    if layer_idx == 0:
        alpha_vals = [
            1.0 if i in desired_channels else Weight
            for i in range(n_maps)
        ]
    else:
        alpha_vals = [1.0] * n_maps

    # ——— 绘制伪彩色热力图 ———
    fig2, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*2, n_rows*2))
    axes = axes.flatten()
    for i in range(n_maps):
        axes[i].imshow(
            fmap[i],
            cmap='jet',
            alpha=alpha_vals[i]   # <-- 这里传入每个通道的 alpha
        )
        axes[i].axis('off')
        axes[i].text(
            0.02, 0.98, str(i),
            color='white', fontsize=8,
            va='top', ha='left',
            transform=axes[i].transAxes,
            backgroundcolor='black'
        )
    for j in range(n_maps, len(axes)):
        axes[j].axis('off')

    fig2.suptitle(f'Layer {layer_idx} Heat')
    cbar = fig2.add_axes([0.92, 0.15, 0.02, 0.7])
    m = plt.cm.ScalarMappable(cmap='jet')
    m.set_array(fmap.numpy())
    fig2.colorbar(m, cax=cbar)
    fig2.tight_layout(rect=[0,0,0.9,1])
    plt.show(block=False)

    input("按回车关闭…")

if __name__ == '__main__':
    # 加载微调权重
    ckpt = torch.load(model_path, map_location='cpu')
    n_classes = ckpt['model.15.weight'].shape[0]
    model = CNN(in_shape=in_shape, n_classes=n_classes)
    model.load_state_dict(ckpt)
    model.to(device)

    apply_mask_conv1(model)   # 只冻结+屏蔽第一层
    # 不要在这里 reset 第二层！

    img_path = r'E:\Project_CNN\0_image\low\000_x150_y100.png'
    visualize_feature_and_heatmaps(model, img_path, layer_idx=0)
