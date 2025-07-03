import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from c4_CVAE_model import CVAE_ResSimple_50  # 确保路径和模型类名正确

def visualize_cvae_feature_maps_nonblocking(
    model: nn.Module,
    img_path: str,
    layer_name: str,          # 'enc1','enc2' 或 'enc3'
    in_shape=(50,50),
    device=torch.device('cpu'),
    n_cols=8                  # 每行显示多少张通道图
):
    # 1) 读图 & 预处理
    img = Image.open(img_path).convert('L')
    tfm = transforms.Compose([
        transforms.Resize(in_shape),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    x = tfm(img).unsqueeze(0).to(device)
    y = torch.zeros(1,1, device=device)

    # 2) hook
    feats = {}
    def hook_fn(m,i,o): feats['maps'] = o.detach().cpu()[0]
    conv_layer = getattr(model, layer_name)[0]
    h = conv_layer.register_forward_hook(hook_fn)
    model.eval()
    with torch.no_grad():
        _ = model.encode(x, y)
    h.remove()

    fmap = feats['maps']  # [C, H, W]
    C, Hf, Wf = fmap.shape
    n_rows = (C + n_cols - 1)//n_cols

    # --- 原图窗口 ---
    fig1 = plt.figure(figsize=(4,4))
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    fig1.canvas.manager.set_window_title('Original')
    plt.show(block=False)

    # --- 灰度特征图窗口 ---
    fig2 = plt.figure(figsize=(n_cols*2, n_rows*2))
    for i in range(C):
        ax = fig2.add_subplot(n_rows, n_cols, i+1)
        ax.imshow(fmap[i], cmap='gray')
        ax.axis('off')
    fig2.suptitle(f'{layer_name} Feature Maps (Gray)')
    fig2.canvas.manager.set_window_title('Gray Feature Maps')
    plt.show(block=False)

    # --- 热力图窗口 ---
    fig3, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*2, n_rows*2))
    axes = axes.flatten()
    for i in range(C):
        axes[i].imshow(fmap[i], cmap='jet')
        axes[i].axis('off')
    for j in range(C, len(axes)):
        axes[j].axis('off')
    fig3.suptitle(f'{layer_name} Feature Maps (Heat)')
    cbar_ax = fig3.add_axes([0.92, 0.15, 0.02, 0.7])
    mappable = plt.cm.ScalarMappable(cmap='jet')
    mappable.set_array(fmap.numpy())
    fig3.colorbar(mappable, cax=cbar_ax)
    fig3.canvas.manager.set_window_title('Heat Feature Maps')
    plt.show(block=False)

    # 防止脚本立刻退出：简单的事件循环或暂停
    plt.pause(0.1)   # 确保所有窗口都画出来
    input("按回车键关闭所有窗口并退出……")

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载模型（保证 num_groups 与训练时一致）
    num_groups = 3
    model = CVAE_ResSimple_50(img_channels=1, latent_dim=32, label_dim=1, num_groups=num_groups)
    ckpt = r'E:\Project_CNN\v_cache\best_model.pth'
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device).eval()

    img_path = r'E:\Project_VAE\V1\50_slice_padding\1_2_3_4\2\crop_Componenets-2.transformed079_50_150.png'

    visualize_cvae_feature_maps_nonblocking(
        model=model,
        img_path=img_path,
        layer_name='enc1',
        in_shape=(50,50),
        device=device,
        n_cols=8
    )
