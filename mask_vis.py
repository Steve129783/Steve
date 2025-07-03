import torch
import matplotlib.pyplot as plt

# 加载缓存文件
cache = torch.load(r"E:\Project_CNN\3_Pack\data_cache.pt")
masks = cache["masks"]  # shape: [N, 1, 50, 50]

# 显示前 16 张 mask
n_show = 16
plt.figure(figsize=(12, 12))
for i in range(n_show):
    plt.subplot(4, 4, i+1)
    plt.imshow(masks[i][0], cmap='gray')  # [0,1] 之间，直接显示
    plt.title(f"Mask {i}")
    plt.axis('off')
plt.tight_layout()
plt.show()
