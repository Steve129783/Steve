import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from f1_FCN import FCNSegmenter

# —— 1. 设备 & 模型加载 —— #
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
in_shape      = (50, 50)
num_classes   = 3               # 请改成你训练时的类别数
desired_layer = 'conv2'         # 'conv1' 或 'conv2'
desired_channel = 13            # 你想可视化的通道索引

model = FCNSegmenter(
    in_shape,
    n_classes=num_classes,
    desired_layer=desired_layer,
    desired_channel=desired_channel
)
checkpoint = torch.load(r'E:\Project_CNN\f_cache\best_model.pth', map_location=device)
model.load_state_dict(checkpoint)
model.to(device)
model.eval()

# —— 2. 单张图像预处理 —— #
transform = transforms.Compose([
    transforms.Resize(in_shape),      # 50×50
    transforms.ToTensor(),            # [H,W] → [C,H,W]
    transforms.Normalize((0.5,), (0.5,))
])

img_path = r'E:\Project_CNN\0_image\low\000_x150_y100.png'
pil     = Image.open(img_path).convert('L')  # 转灰度
tensor  = transform(pil)                     # [1,50,50]
new_img = tensor.unsqueeze(0).to(device)     # → [1,1,50,50]

# —— 3. 前向 & 掩码生成 —— #
with torch.no_grad():
    _, seg_logits = model(new_img)           # [1,1,50,50]
    prob_map = torch.sigmoid(seg_logits)     # 0–1 归一化
    mask     = (prob_map >= 0.5).float()     # 二值掩码

# —— 4. 可视化 —— #
heat     = prob_map[0,0].cpu().numpy()
bin_mask = mask[0,0].cpu().numpy()
orig     = tensor[0].cpu().numpy()         # 归一化后的原图（值在 -1 到 1）

fig, axes = plt.subplots(1, 3, figsize=(9, 3))
axes[0].imshow((orig * 0.5 + 0.5), cmap='gray')  
axes[0].set_title('Input')
axes[1].imshow(heat, cmap='jet')
axes[1].set_title(f'{desired_layer} ch{desired_channel} Heatmap')
axes[2].imshow(bin_mask, cmap='gray')
axes[2].set_title('Binary Mask')
for ax in axes:
    ax.axis('off')
plt.tight_layout()
plt.show()
