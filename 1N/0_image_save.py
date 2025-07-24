from PIL import Image, ImageSequence
import os, math

# 输入 TIFF 路径
tif_path  = r'E:\Project_CNN\CT_DATA\OneDrive_1_2025-6-30\14_LowEnergyCropped.tif'
# 保存裁剪后整张图的目录
full_dir  = r'E:\Project_SNV\0S\3_full_img\l'
# 保存切分 patch 的目录
patch_dir = r'E:\Project_SNV\0S\6_patch\l'

os.makedirs(full_dir,  exist_ok=True)
os.makedirs(patch_dir, exist_ok=True)

# 目标 patch 大小
base_h, base_w = 50, 50

with Image.open(tif_path) as im:
    for idx, page in enumerate(ImageSequence.Iterator(im)):
        w, h = page.size
        print(f"Frame {idx:03d}: original size = {w}×{h}")

        # 计算可整除的区域
        n_cols = w  // base_w
        n_rows = h  // base_h
        crop_w = n_cols * base_w
        crop_h = n_rows * base_h

        if n_cols == 0 or n_rows == 0:
            print("  Skipped: 图像尺寸不足 50×50")
            continue

        # 裁剪左上角的整除区块
        img_cropped = page.crop((0, 0, crop_w, crop_h))
        print(f"  Cropped area = {crop_w}×{crop_h} (丢弃右 {w-crop_w}px，底 {h-crop_h}px)")

        # 保存裁剪后的整图
        full_name = f"{idx:03d}_cropped_{crop_w}x{crop_h}.png"
        img_cropped.save(os.path.join(full_dir, full_name))
        print(f"    Saved full cropped image: {full_name}")

        # 切块并保存每个 50×50 patch
        count = 0
        for y in range(0, crop_h, base_h):
            for x in range(0, crop_w, base_w):
                patch = img_cropped.crop((x, y, x + base_w, y + base_h))
                name  = f"{idx:03d}_x{x}_y{y}.png"
                patch.save(os.path.join(patch_dir, name))
                count += 1

        print(f"    Saved {count} patches to {patch_dir}")
