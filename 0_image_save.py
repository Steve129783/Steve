from PIL import Image, ImageSequence, ImageOps
import os, math

# 输入 TIFF 路径
tif_path = r'E:\Project_CNN\CT_DATA\OneDrive_1_2025-6-30\12_HighEnergyCropped.tif'
# 保存 padded 完整帧的目录
full_dir  = r'E:\Project_CNN\1_full\image\high'
# 保存切分 patch 的目录
patch_dir = r'E:\Project_CNN\0_image\high'

os.makedirs(full_dir, exist_ok=True)
os.makedirs(patch_dir, exist_ok=True)

# 目标 patch 和对齐的基准尺寸
base_h, base_w = 50, 50

with Image.open(tif_path) as im:
    for idx, page in enumerate(ImageSequence.Iterator(im)):
        w, h = page.size
        print(f"Frame {idx:03d}: original size = {w}×{h}")
        
        # 计算要填充到的最小 50 整数倍
        target_w = math.ceil(w  / base_w) * base_w
        target_h = math.ceil(h  / base_h) * base_h

        # 计算每边要 pad 的像素数，保证居中
        pad_left   = (target_w - w) // 2
        pad_right  = target_w - w - pad_left
        pad_top    = (target_h - h) // 2
        pad_bottom = target_h - h - pad_top

        # 居中填充黑边
        padded = ImageOps.expand(page,
                                 border=(pad_left, pad_top, pad_right, pad_bottom),
                                 fill=0)
        W, H = padded.size
        print(f"  → centered padded size = {W}×{H}"
              f"  (L={pad_left}, T={pad_top}, R={pad_right}, B={pad_bottom})")

        # 保存整帧 padded 图
        full_name = f"{idx:03d}_centered_{W}x{H}.png"
        padded.save(os.path.join(full_dir, full_name))
        print(f"    Saved centered padded image: {full_name}")

        # 按 50×50 均匀切分
        for y in range(0, H, base_h):
            for x in range(0, W, base_w):
                box   = (x, y, x + base_w, y + base_h)
                patch = padded.crop(box)
                name  = f"{idx:03d}_x{x}_y{y}.png"
                patch.save(os.path.join(patch_dir, name))
        print(f"    Saved patches in {patch_dir}")
