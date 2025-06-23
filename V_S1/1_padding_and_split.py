import cv2
import numpy as np
import os

def pad_to_center_multiple_of_50(img):
    """
    对一张灰度图 img (H,W)，以图像中心为基准，
    在上下左右均匀补黑边，使其新尺寸 (new_H, new_W) 恰好是 50 的倍数。
    返回补好边的图。
    """
    H, W = img.shape[:2]

    new_H = ((H + 49) // 50) * 50
    new_W = ((W + 49) // 50) * 50

    pad_total_vert = new_H - H
    pad_total_horz = new_W - W

    pad_top    = pad_total_vert // 2
    pad_bottom = pad_total_vert - pad_top

    pad_left   = pad_total_horz // 2
    pad_right  = pad_total_horz - pad_left

    padded = cv2.copyMakeBorder(
        img,
        top=pad_top, bottom=pad_bottom,
        left=pad_left, right=pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=0
    )
    return padded

def crop_50x50_from_center_padded(img, output_folder, prefix="patch"):
    """
    1. 先把 img pad 到中心对齐的 50 倍数尺寸
    2. 从左上角(0,0)开始每隔 50 px 切一个 50×50 的小图
    3. 不做任何亮度检查，直接保存所有切片
    """
    padded = pad_to_center_multiple_of_50(img)
    H_p, W_p = padded.shape[:2]

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for y in range(0, H_p, 50):
        for x in range(0, W_p, 50):
            patch = padded[y:y+50, x:x+50]
            out_name = f"{prefix}_{y}_{x}.png"
            cv2.imwrite(os.path.join(output_folder, out_name), patch)

# 批量处理示例
input_folder  = r"D:\Study\Postgraduate\S2\Project\Code\Resource\Original\1_2_3_4\2"
output_folder = r"E:\Project_VAE\V1\50_slice_padding\1_2_3_4\2"

for fname in os.listdir(input_folder):
    if not fname.lower().endswith(('.png', '.jpg', '.tif')):
        continue
    img_path = os.path.join(input_folder, fname)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue

    prefix = os.path.splitext(fname)[0]
    crop_50x50_from_center_padded(img, output_folder, prefix=prefix)
