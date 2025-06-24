#!/usr/bin/env python3
# reassemble_defects_with_aspect_padding.py

import os
import re
import numpy as np
from PIL import Image
import torch
from skimage.measure import label, regionprops

def reassemble_defects(data_cache_path, padded_tif_dir, patch_json_dir, output_path):
    """
    重组缺陷实例，保持 ROI 长宽比：
      1) 贴回小 mask 到大图
      2) 连通域提取缺陷区域
      3) 等比缩放至最长边 50px，再在同模式画布上居中 padding 为 50×50
      4) 记录 contributing_patches & patch_jsons
    """
    # 1. 加载缓存
    cache = torch.load(data_cache_path)
    masks = cache['masks'].numpy()   # (N,1,50,50)
    meta  = cache['meta']            # list of dict with 'base','instance_id','patch_row','patch_col'

    print(f"[DEBUG] Loaded {len(meta)} meta entries.")

    # 2. 拼装 full_mask 并记录 json
    instances = {}
    for idx, m in enumerate(meta):
        base_clean = re.sub(r'_[0-9]+_[0-9]+$', '', m['base'])
        inst_id    = m['instance_id']
        if inst_id == -1:
            continue

        key = (base_clean, inst_id)
        if key not in instances:
            tif_path = os.path.join(padded_tif_dir, f"{base_clean}.tif")
            if not os.path.exists(tif_path):
                print(f"[WARN] TIFF not found: {tif_path}")
                continue
            with Image.open(tif_path) as tmp:
                w, h = tmp.size
                mode = tmp.mode
            instances[key] = {
                'full_mask': np.zeros((h, w), dtype=np.uint8),
                'tif_path': tif_path,
                'patch_bases': set(),
                'patch_jsons': [],
                'img_mode': mode
            }

        row, col = m['patch_row'], m['patch_col']
        y0, x0   = row * 50, col * 50
        instances[key]['full_mask'][y0:y0+50, x0:x0+50] |= (masks[idx,0] > 0).astype(np.uint8)

        json_name = f"{base_clean}_{row}_{col}.json"
        json_path = os.path.join(patch_json_dir, json_name)
        if os.path.exists(json_path):
            instances[key]['patch_bases'].add(json_name)
            instances[key]['patch_jsons'].append(json_path)

    # 3. 提取 ROI 并等比缩放＋padding
    TARGET = 50
    defect_data = []
    for (base_clean, inst_id), info in instances.items():
        mask_full = info['full_mask']
        for region in regionprops(label(mask_full)):
            minr, minc, maxr, maxc = region.bbox
            minr, minc = max(0, minr), max(0, minc)
            maxr = min(mask_full.shape[0], maxr)
            maxc = min(mask_full.shape[1], maxc)

            # 打开原图并裁剪
            img = Image.open(info['tif_path']).convert(info['img_mode'])
            roi = img.crop((minc, minr, maxc, maxr))

            # 等比缩放
            w, h = roi.size
            scale = TARGET / max(w, h)
            nw, nh = int(w * scale), int(h * scale)
            resized = roi.resize((nw, nh), Image.BILINEAR)

            # padding 到目标 50×50
            canvas = Image.new(info['img_mode'], (TARGET, TARGET), 0)
            # 对彩色图默认为全 0（黑）；对 L 模式即灰度黑
            offset_x = (TARGET - nw) // 2
            offset_y = (TARGET - nh) // 2
            canvas.paste(resized, (offset_x, offset_y))

            defect_data.append({
                'base': base_clean,
                'instance_id': inst_id,
                'padded_bbox': (minr, minc, maxr, maxc),
                'contributing_patches': list(info['patch_bases']),
                'patch_jsons': info['patch_jsons'],
                'image': canvas,
                'mask': mask_full[minr:maxr, minc:maxc],
                'source_tif': info['tif_path']
            })

    # 4. 保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(defect_data, output_path)

    print("\nReassembly complete:")
    print(f"- TIFF processed:      {len(instances)}")
    print(f"- JSON patches found:  {sum(len(v['patch_jsons']) for v in instances.values())}")
    print(f"- Defect instances:    {len(defect_data)}")
    print(f"- Saved to:            {output_path}")


if __name__ == "__main__":
    data_cache_path = r"E:\Project_VAE\V2\Pack_json_png\Ori_drop\data_cache.pt"
    padded_tif_dir  = r"E:\Project_VAE\V2\padded_image\1_2_3_4\1"
    patch_json_dir  = r"E:\Project_VAE\V2\Sliced_json\1_2_3_4\1"
    output_path     = r"E:\Project_VAE\V2\Reorganization\1_2_3_4\1\reassembled_defects.pt"

    reassemble_defects(
        data_cache_path=data_cache_path,
        padded_tif_dir=padded_tif_dir,
        patch_json_dir=patch_json_dir,
        output_path=output_path
    )
