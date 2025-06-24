#!/usr/bin/env python3
# reassemble_defects_with_padding_rgb_with_type.py

import os
import re
import numpy as np
from PIL import Image, ImageDraw
import cv2
import torch
from skimage.measure import label, regionprops
import json
import pandas as pd
from pathlib import Path

# Helper: 检测图像类型
def detect_image_type(image_path, tol=0):
    img = Image.open(image_path)
    mode = img.mode
    # 1-bit 二值
    if mode == '1':
        return 'binary_1bit'
    # 灰度模式
    if mode == 'L' or mode == 'LA':
        arr = np.array(img.convert('L'))
        unique = np.unique(arr)
        if set(unique.tolist()).issubset({0, 255}):
            return 'binary'  # 只有黑白
        return 'grayscale'
    # 调色板或彩色模式
    if mode in ('P', 'RGB', 'RGBA'):
        rgb = img.convert('RGB')
        arr = np.array(rgb)
        r, g, b = arr[...,0], arr[...,1], arr[...,2]
        if np.all(np.abs(r - g) <= tol) and np.all(np.abs(r - b) <= tol):
            return 'pseudo_gray'  # RGB 通道相等
        return 'color'
    return mode

# 主函数：统计主体亮度并检测类型
if __name__ == '__main__':
    tif_dir = Path(r'E:\Project_VAE\V2\padded_image\1_2_3_4\1')
    json_dir = Path(r'D:\Study\Postgraduate\S2\Project\Code\Resource\Middle Stage B\1_2_3_4\1')

    fnames = sorted([p for p in tif_dir.iterdir() if p.suffix.lower() in ('.tif', '.tiff', '.png', '.jpg', '.jpeg')])
    results = []

    for image_path in fnames:
        # 检测图像类型
        img_type = detect_image_type(image_path)

        # 读取灰度图
        gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            print(f"Failed to load image {image_path.name}")
            continue
        h, w = gray.shape

        # JSON 标注
        json_path = json_dir / f"{image_path.stem}.json"
        if not json_path.exists():
            body_pixels = gray.flatten()
        else:
            with json_path.open('r', encoding='utf-8') as f:
                ann = json.load(f)
            mask = np.zeros((h, w), dtype=np.uint8)
            for shape in ann.get('shapes', []):
                if shape.get('label', '').lower() == 'main body':
                    tmp = Image.new('L', (w, h), 0)
                    draw = ImageDraw.Draw(tmp)
                    pts = [tuple(p) for p in shape.get('points', [])]
                    draw.polygon(pts, fill=1)
                    mask[np.array(tmp, bool)] = 1
            body_pixels = gray[mask == 1] or gray.flatten()

        # 计算统计量
        if body_pixels.size == 0:
            stats = dict(mean_brightness=np.nan, max_brightness=np.nan,
                         min_brightness=np.nan, median_brightness=np.nan,
                         percentile_70_brightness=np.nan)
        else:
            stats = dict(
                mean_brightness=float(body_pixels.mean()),
                max_brightness=int(body_pixels.max()),
                min_brightness=int(body_pixels.min()),
                median_brightness=float(np.median(body_pixels)),
                percentile_70_brightness=float(np.percentile(body_pixels, 70))
            )

        results.append({
            'image_name': image_path.name,
            'image_type': img_type,
            **stats
        })

    # 输出 DataFrame
    df = pd.DataFrame(results)
    print(df)
    # 保存 CSV
    output_csv = Path(r'E:\Project_VAE') / 'brightness_stats_with_type.csv'
    df.to_csv(output_csv, index=False)
    print(f"Saved stats to: {output_csv}")
