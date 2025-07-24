#!/usr/bin/env python3
import os
import torch
from PIL import Image
import torchvision.transforms.functional as TF
from concurrent.futures import ProcessPoolExecutor, as_completed

# ================ 配置 ===================
root_dir        = r"E:\Project_SNV\0S\6_patch"
output_cache    = r"E:\Project_SNV\1N\1_Pack\17_18_19_20\data_cache.pt"
num_workers     = os.cpu_count() or 4

# 只处理这些子文件夹；填写它们的精确名称
include_groups  = ["17", "18", "19", '20']

# 处理单张图像，返回张量与所属组ID
def process_patch(args):
    img_path, gid = args
    img = Image.open(img_path).convert("L")
    tensor = TF.to_tensor(img)
    if tensor.shape[1:] != (50,50):
        tensor = TF.resize(tensor, (50,50))
    return tensor, gid

# 并行读取并缓存图像函数
def load_patches_parallel(root_dir, include_groups=None):
    # 先收集所有子文件夹，并按 include_groups 过滤
    all_dirs = sorted(
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    )
    if include_groups is not None:
        groups = [d for d in all_dirs if d in include_groups]
    else:
        groups = all_dirs

    tasks = []
    group_map = {}
    # 对过滤后的组进行 enumerate，保证 gid 从 0 开始
    for gid, group in enumerate(groups):
        gp = os.path.join(root_dir, group)
        group_map[gid] = group
        for fname in sorted(os.listdir(gp)):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                continue
            tasks.append((os.path.join(gp, fname), gid))

    images, group_ids, paths = [], [], []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_patch, t): t for t in tasks}
        for future in as_completed(futures):
            tensor, gid = future.result()
            img_path, _ = futures[future]
            images.append(tensor)
            group_ids.append(gid)
            paths.append(img_path)

    images_tensor = torch.stack(images) if images else torch.empty(0)
    return images_tensor, group_ids, group_map, paths

if __name__ == '__main__':
    images, group_ids, group_map, paths = load_patches_parallel(
        root_dir, include_groups=include_groups
    )
    cache = {"images": images, "group_ids": group_ids, "paths": paths}
    os.makedirs(os.path.dirname(output_cache), exist_ok=True)
    torch.save(cache, output_cache)

    print(f"Packed {len(images)} patches from {len(group_map)} groups into {output_cache}")
    print("Group mapping:")
    for gid, name in group_map.items():
        count = group_ids.count(gid)
        print(f"  {gid}: {name} ({count} patches)")
