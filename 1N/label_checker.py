import torch
from collections import Counter
import matplotlib.pyplot as plt

# 载入 cache
data = torch.load(r"E:\Project_SNV\1N\1_Pack\1_2_3_4\data_cache.pt", map_location="cpu")
images   = data["images"]
group_ids = data["group_ids"]
paths    = data["paths"]

# 1. 看一下前几个 paths 和对应的 group_id
for i in range(5):
    print(f"{i:03d} | group {group_ids[i]} | {paths[i]}")

# 2. 看各个 group 里有多少张图
print("Group 分布：", Counter(group_ids))

# 3. 随便展示一张 patch 确认一下像素范围
idx = 42
patch = images[idx][0].numpy()   # [1,50,50] → [50,50]
plt.imshow(patch, cmap="gray")
plt.title(f"{idx}: group {group_ids[idx]}")
plt.axis("off")
plt.show()
