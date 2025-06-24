import os
import re
import numpy as np
from PIL import Image
import torch
from skimage.measure import label, regionprops

def reassemble_defects(data_cache_path, padded_tif_dir, patch_json_dir, output_path):
    """
    重组缺陷实例，区分原始 TIFF 大图和 patch JSON 输入，
    仅在 JSON 文件存在时记录路径，以兼容部分 patch 无 defect 的情况。
    """
    # 1. 加载缓存
    cache = torch.load(data_cache_path)
    masks = cache['masks'].numpy()   # (N,1,50,50)
    meta  = cache['meta']            # list of dict with 'base','instance_id','patch_row','patch_col'

    print(f"[DEBUG] Loaded {len(meta)} meta entries.")

    instances = {}
    for idx, m in enumerate(meta):
        base_patch = m['base']  # e.g. "crop_Componenets_1.transformed059_0_3"
        base_clean = re.sub(r'_\d+_\d+$', '', base_patch)

        instance_id = m['instance_id']
        if instance_id == -1:
            continue

        key = (base_clean, instance_id)
        if key not in instances:
            tif_path = os.path.join(padded_tif_dir, f"{base_clean}.tif")
            if not os.path.exists(tif_path):
                print(f"[WARN] TIFF not found: {tif_path}")
                continue
            with Image.open(tif_path) as img:
                w, h = img.size
            instances[key] = {
                'full_mask': np.zeros((h, w), dtype=np.uint8),
                'tif_path': tif_path,
                'patch_bases': set(),
                'patch_jsons': []
            }

        # 贴 mask 回全图
        row, col = m['patch_row'], m['patch_col']
        y0, x0   = row * 50, col * 50
        instances[key]['full_mask'][y0:y0+50, x0:x0+50] |= (masks[idx,0] > 0).astype(np.uint8)

        # 记录 JSON 路径
        json_name = f"{base_clean}_{row}_{col}.json"
        json_path = os.path.join(patch_json_dir, json_name)
        if os.path.exists(json_path):
            instances[key]['patch_bases'].add(json_name)
            instances[key]['patch_jsons'].append(json_path)

    # 2. 提取完整缺陷 ROI
    defect_data = []
    for (base_clean, instance_id), info in instances.items():
        mask_full = info['full_mask']
        labeled   = label(mask_full)
        for region in regionprops(labeled):
            minr, minc, maxr, maxc = region.bbox
            # 边界安全裁剪
            minr, minc = max(0, minr), max(0, minc)
            maxr = min(mask_full.shape[0], maxr)
            maxc = min(mask_full.shape[1], maxc)

            with Image.open(info['tif_path']) as img:

                # crop 的参数顺序应为 (left, upper, right, lower)
                roi = img.crop((minc, minr, maxc, maxr)).resize((50, 50), Image.BILINEAR)

            defect_data.append({
                'base': base_clean,
                'instance_id': instance_id,
                'padded_bbox': (minr, minc, maxr, maxc),
                'contributing_patches': list(info['patch_bases']),
                'patch_jsons': info['patch_jsons'],
                'image': roi,
                'mask': mask_full[minr:maxr, minc:maxc],
                'source_tif': info['tif_path']
            })

    # 3. 保存结果
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