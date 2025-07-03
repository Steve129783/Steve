import os
import torch
from PIL import Image
import torchvision.transforms.functional as TF
from concurrent.futures import ProcessPoolExecutor, as_completed

# 配置：
root_dir = r"E:\Project_CNN\0_image"   # 包含 high/, low/, correct/ 子文件夹
output_cache = r"E:\Project_CNN\2_Pack\data_cache.pt"
num_workers = os.cpu_count() or 4  # 并行进程数

# 单图像加载和预处理函数
def process_patch(args):
    img_path, gid = args
    # 读取灰度图
    img = Image.open(img_path).convert("L")
    # 转 tensor 并 resize
    tensor = TF.to_tensor(img)
    if tensor.shape[1:] != (50, 50):
        tensor = TF.resize(tensor, (50, 50))
    # 掩码
    mask = (tensor > 0).float()
    return tensor, mask, gid

# 批量并行加载
def load_patches_parallel(root_dir):
    # 收集所有图像路径及其 group id
    tasks = []  # list of (path, gid)
    group_map = {}
    for gid, group in enumerate(sorted(os.listdir(root_dir))):
        group_path = os.path.join(root_dir, group)
        if not os.path.isdir(group_path):
            continue
        group_map[gid] = group
        for fname in sorted(os.listdir(group_path)):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                continue
            tasks.append((os.path.join(group_path, fname), gid))

    images, masks, group_ids = [], [], []
    # 并行执行
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_patch, t): t for t in tasks}
        for future in as_completed(futures):
            tensor, mask, gid = future.result()
            images.append(tensor)
            masks.append(mask)
            group_ids.append(gid)

    # 堆叠成大张量
    images_tensor = torch.stack(images)
    masks_tensor = torch.stack(masks)
    return images_tensor, masks_tensor, group_ids, group_map

if __name__ == '__main__':
    images, masks, group_ids, group_map = load_patches_parallel(root_dir)
    cache = {
        "images": images,      # [N,1,50,50]
        "masks": masks,        # [N,1,50,50]
        "group_ids": group_ids # list[int]
    }
    os.makedirs(os.path.dirname(output_cache), exist_ok=True)
    torch.save(cache, output_cache)
    print(f"Packed {len(images)} patches from {len(group_map)} groups into {output_cache}")
    print("Group mapping:")
    for gid, name in group_map.items():
        print(f"  {gid}: {name} ({group_ids.count(gid)} patches)")
