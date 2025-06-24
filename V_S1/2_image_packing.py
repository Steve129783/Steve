import os
import cv2
import torch
import numpy as np
import torchvision.transforms.functional as TF
import random

# ─── 配置 ───
images_folder        = r"E:\Project_VAE\V2\Sliced_png\1_2_3_4\1"
save_path            = r"E:\Project_VAE\V2\Ori_img\Val\data_cache.pt"
augment_all          = False    # 是否对所有保留的图做一次随机增强
brightness_max_thres = 100      # mx < 100 的图据为“纯 BG＋padding”
brightness_min_thres = 100       # 在 mx ≥ 100 的图中，mn < 60 的视为“dark_pad”
p_keep_dark          = 1      # 对 dark_pad 只保留 10%
random.seed(42)
# ───────────

# 临时存放
dark_pad_paths   = []  # mx≥100 且 mn<60
standard_paths   = []  # mx≥100 且 mn≥60

# 1) 扫描并初步分类
for root, dirs, files in os.walk(images_folder):
    for fname in files:
        if not fname.lower().endswith((".png",".jpg",".jpeg",".tif",".tiff")):
            continue
        full_path = os.path.join(root, fname)
        gray = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
        if gray is None or gray.shape != (50,50):
            continue

        mn = int(gray.min())
        mx = int(gray.max())

        # 丢掉所有 mx < 100（纯 BG＋padding）的图
        if mx < brightness_max_thres:
            continue

        # 在剩下 mx ≥ 100 的图里，按 mn 分类
        if mn < brightness_min_thres:
            dark_pad_paths.append(full_path)
        else:
            standard_paths.append(full_path)

# 2) 从 dark_pad_paths 中随机抽 10% 保留
n_keep_dark = int(len(dark_pad_paths) * p_keep_dark)
keep_dark   = set(random.sample(dark_pad_paths, n_keep_dark))

# 最终保留列表 = standard_paths + keep_dark
keep_paths = standard_paths + list(keep_dark)

print(f"总图像数（mx≥100）: {len(standard_paths) + len(dark_pad_paths)}")
print(f"  standard (mn≥{brightness_min_thres}): {len(standard_paths)}")
print(f"  dark_pad (mn<{brightness_min_thres}): {len(dark_pad_paths)} → 保留 {len(keep_dark)}")
print(f"最终保留用于打包: {len(keep_paths)} 张")

# 3) 打包成 data_cache.pt
image_tensors = []
paths = []

for full_path in keep_paths:
    # 3.1 读取并归一化
    gray = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
    img_norm   = gray.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_norm).unsqueeze(0)  # (1,50,50)

    # 3.2 添加原图
    image_tensors.append(img_tensor)
    paths.append(full_path)

    # 3.3 可选增强
    if augment_all:
        choice = random.choice(["rot","hflip","vflip","bright"])
        if choice=="rot":
            angle = random.choice([10,-10])
            img_aug = TF.rotate(img_tensor, angle=angle, expand=False, fill=0)
            tag = f"__rot{angle}"
        elif choice=="hflip":
            img_aug = TF.hflip(img_tensor)
            tag = "__hflip"
        elif choice=="vflip":
            img_aug = TF.vflip(img_tensor)
            tag = "__vflip"
        else:
            frac = random.uniform(0.9,1.1)
            img_aug = TF.adjust_brightness(img_tensor, frac)
            tag = f"__bright{frac:.2f}"

        image_tensors.append(img_aug)
        paths.append(full_path + tag)

# 4) 保存
if not image_tensors:
    raise RuntimeError("没有保留任何图像，请检查过滤条件。")

cache_dict = {
    "images": torch.stack(image_tensors, 0),  # (N,1,50,50)
    "paths":  paths
}
os.makedirs(os.path.dirname(save_path), exist_ok=True)
torch.save(cache_dict, save_path)
print(f"打包完成：共 {len(image_tensors)} 张（含增强），保存在\n  {save_path}")
