import os
import torch
from collections import defaultdict

# 载入缓存
cache = torch.load(r"E:\Project_CNN\2_Pack\data_cache.pt", map_location="cpu")
paths     = cache['paths']      # list[str]
group_ids = cache['group_ids']  # list[int]

# 按 group_id 收集文件夹名
mp = defaultdict(set)
for p, g in zip(paths, group_ids):
    # 取文件所在的上级目录名作为标签
    label = os.path.basename(os.path.dirname(p))
    mp[int(g)].add(label)

# 打印结果
for gid, labels in sorted(mp.items()):
    print(f"group {gid} 对应 标签：{labels}")
