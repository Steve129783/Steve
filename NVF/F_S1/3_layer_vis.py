import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import numpy as np
from f1_FCN import FCNSegmenter   # 或者你的模型定义文件

# ——— 可视化函数 ———
def visualize_layer_maps(
    model: nn.Module,
    img_path: str,
    layer_name: str = 'conv1',       # 'conv1' 或 'conv2'
    in_shape: tuple = (50,50),
    desired_channel: int = None,     # 若为 None，显示所有通道
    device: torch.device = torch.device('cpu'),
    n_cols: int = 8
):
    # 1. 读图 & 预处理
    pil = Image.open(img_path).convert('L')
    tfm = transforms.Compose([
        transforms.Resize(in_shape),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    x = tfm(pil).unsqueeze(0).to(device)  # [1,1,H,W]

    # 2. 钩子拿到指定层输出
    fmap_dict = {}
    def hook_fn(m, i, o):
        fmap_dict['maps'] = o.detach().cpu()[0]  # [C, h, w]
    # 注册钩子
    layer = getattr(model, layer_name)
    handle = layer.register_forward_hook(hook_fn)

    model.eval()
    with torch.no_grad():
        _ = model(x)
    handle.remove()

    fmap = fmap_dict['maps']   # Tensor C×h×w
    C, Hf, Wf = fmap.shape

    # 如果 desired_channel 指定了单通道，就只看那一个
    channels = [desired_channel] if desired_channel is not None else list(range(C))
    n_maps = len(channels)
    n_rows = (n_maps + n_cols - 1) // n_cols

    # 3. 展示 原图
    fig0 = plt.figure(figsize=(3,3))
    plt.imshow(pil, cmap='gray')
    plt.title('Input Image')
    plt.axis('off')
    plt.show()

    # 4. 灰度特征图
    fig1, axes1 = plt.subplots(n_rows, n_cols, figsize=(n_cols*1.8, n_rows*1.8))
    axes1 = axes1.flatten()
    for idx, ch in enumerate(channels):
        ax = axes1[idx]
        ax.imshow(fmap[ch], cmap='gray')
        ax.set_title(f'ch{ch}', fontsize=6)
        ax.axis('off')
    for idx in range(n_maps, len(axes1)):
        axes1[idx].axis('off')
    fig1.suptitle(f'{layer_name} Feature Maps (gray)', fontsize=10)
    plt.tight_layout()
    plt.show()

    # 5. 伪彩色热力图
    fig2, axes2 = plt.subplots(n_rows, n_cols, figsize=(n_cols*1.8, n_rows*1.8))
    axes2 = axes2.flatten()
    vmin, vmax = fmap.min().item(), fmap.max().item()
    for idx, ch in enumerate(channels):
        ax = axes2[idx]
        im = ax.imshow(fmap[ch], cmap='jet', vmin=vmin, vmax=vmax)
        ax.set_title(f'ch{ch}', fontsize=6)
        ax.axis('off')
    for idx in range(n_maps, len(axes2)):
        axes2[idx].axis('off')
    fig2.suptitle(f'{layer_name} Feature Maps (heat)', fontsize=10)
    # colorbar
    cax = fig2.add_axes([0.92, 0.15, 0.02, 0.7])
    fig2.colorbar(im, cax=cax)
    plt.tight_layout(rect=[0,0,0.9,1])
    plt.show()


# ——— 主程序 ———
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    in_shape = (50,50)
    # 1) 加载模型
    #    注意：这里需要传一个 dummy desired_layer/desired_channel，后面钩子处才生效
    model = FCNSegmenter(
        in_shape=in_shape,
        n_classes=3,               # 训练时的类别数
        desired_layer='conv1',
        desired_channel=0
    )
    ckpt = torch.load(r'E:\Project_CNN\f_cache\best_model.pth', map_location=device)
    model.load_state_dict(ckpt)
    model.to(device)

    # 2) 可视化整个 conv2 层的所有通道
    visualize_layer_maps(
        model,
        img_path=r'E:\Project_CNN\0_image\low\000_x150_y100.png',
        layer_name='conv1',
        in_shape=in_shape,
        desired_channel=None,      # None 就显示所有通道
        device=device,
        n_cols=8                   # 每行放 8 张图，你可以调
    )

    # 3) 如果你只想看某个通道，比如第 5 个：
    # visualize_layer_maps(model, img_path, layer_name='conv1', desired_channel=5, device=device)
