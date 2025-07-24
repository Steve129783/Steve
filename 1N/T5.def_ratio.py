#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import torch
import csv
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

# -------- 配置区 --------
group_n    = '1_2_3_4'
CACHE_FILE = rf'E:\Project_SNV\1N\1_Pack\{group_n}\data_cache.pt'
CSV_OUT    = rf'E:\Project_SNV\1N\c1_cache\{group_n}\defect_ratios.csv'
BATCH_SIZE = 16

# 如果想可视化某个具体 path，就在这里写上它；否则设为 None
# 例如 "idx_42" 或者完整文件名; 设为 None 则只输出 CSV # r"E:\Project_SNV\0S\6_patch\2\564_5_2.png"
SHOW_PATH = r'E:\Project_SNV\0S\6_patch\2\34_6_5.png'   

# -------- 定义 Dataset --------
class CachedImageDataset(Dataset):
    def __init__(self, cache_path):
        cache = torch.load(cache_path, map_location='cpu')
        self.images = cache['images']        # Tensor [N,1,H,W]
        self.paths  = cache.get(
            'paths',
            [f"idx_{i}" for i in range(len(self.images))]
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img  = self.images[idx].squeeze().numpy()
        gray = (img * 255).astype(np.uint8)
        path = str(self.paths[idx])
        return gray, path

# -------- MSER Mask 生成 --------
def defect_mask_mser(gray, min_area=8, max_area=70):
    bw = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        51, 5
    )
    mser = cv2.MSER_create()
    mser.setMinArea(min_area)
    mser.setMaxArea(max_area)
    regions, _ = mser.detectRegions(bw)
    mask = np.zeros_like(gray, dtype=np.uint8)
    for pts in regions:
        cv2.fillPoly(mask, [pts.reshape(-1,1,2)], 1)
    return bw, mask

# -------- 可视化单张图 --------
def visualize_single(gray, bw, mask, path):
    img_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    red = np.zeros_like(img_bgr); red[:,:,2] = mask*255
    overlay = cv2.addWeighted(img_bgr, 0.7, red, 0.3, 0)
    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    titles = ['Gray', 'Binary', 'MSER Mask', 'Overlay']
    images = [gray, bw, mask*255, overlay]

    plt.figure(figsize=(12,3))
    for i,(im,t) in enumerate(zip(images,titles),1):
        ax = plt.subplot(1,4,i)
        cmap = 'gray' if im.ndim==2 else None
        plt.imshow(im, cmap=cmap)
        ax.set_title(f"{t}\n{os.path.basename(path)}", fontsize=8)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# -------- 主流程 --------
def main():
    ds = CachedImageDataset(CACHE_FILE)

    # 如果设定了 SHOW_PATH，则只可视化这一张
    if SHOW_PATH is not None:
        if SHOW_PATH not in ds.paths:
            print(f"Path '{SHOW_PATH}' not found in dataset.")
            return
        idx = ds.paths.index(SHOW_PATH)
        gray, _ = ds[idx]
        bw, mask = defect_mask_mser(gray)
        visualize_single(gray, bw, mask, SHOW_PATH)
        return

    # 否则遍历全部，输出 CSV
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda x: x)
    os.makedirs(os.path.dirname(CSV_OUT), exist_ok=True)
    with open(CSV_OUT, 'w', newline='') as fp:
        writer = csv.writer(fp)
        writer.writerow(['path', 'defect_ratio'])
        for batch in loader:
            for gray, path in batch:
                if isinstance(gray, torch.Tensor):
                    gray = gray.cpu().numpy().astype(np.uint8)
                bw, mask = defect_mask_mser(gray)
                ratio = mask.sum() / mask.size
                writer.writerow([path, f"{ratio:.6f}"])
    print(f"CSV written to: {CSV_OUT}")

if __name__ == '__main__':
    main()
