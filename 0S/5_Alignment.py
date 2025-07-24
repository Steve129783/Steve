#!/usr/bin/env python3
import os
import json
import numpy as np
from PIL import Image
from skimage.draw import polygon

# —— 在这里显式设定 —— #
N = 20
img_dir      = fr"E:\Project_SNV\0S\3_full_img\{N}"
json_dir     = fr"E:\Project_SNV\0S\4_moved_MB\{N}"
out_dir      = fr"E:\Project_SNV\0S\5_aligned_img\{N}"

# 如果想指定外部 sample JSON，请把路径写在这里；否则设为 None
sample_json_path = r'E:\Project_SNV\0S\4_moved_MB\3\0.json'
# 如果想直接用本系列中的第几张（0-based）做参考，请写在这里；否则设为 None
sample_index     = None
# —— 结束设定 —— #

def load_image_volume(img_dir, ext=".png"):
    files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(ext)])
    arrs  = [np.array(Image.open(os.path.join(img_dir, f))) for f in files]
    return np.stack(arrs, axis=0), files  # (D,H,W)

def load_mb_mask_volume(json_dir, H=None, W=None):
    files = sorted([f for f in os.listdir(json_dir) if f.lower().endswith(".json")])
    # 如果没给 H,W，就用第一个 JSON 里的尺寸
    if H is None or W is None:
        sample = json.load(open(os.path.join(json_dir, files[0]), encoding="utf-8"))
        H, W = sample["imageHeight"], sample["imageWidth"]

    masks = []
    for fn in files:
        data = json.load(open(os.path.join(json_dir, fn), encoding="utf-8"))
        mb   = [s for s in data["shapes"] if s["label"] == "main body"]
        mask = np.zeros((H, W), np.uint8)
        for shp in mb:
            pts     = np.array(shp["points"], float)
            rr, cc  = polygon(pts[:,1], pts[:,0], shape=mask.shape)
            mask[rr, cc] = 1
        masks.append(mask)
    return np.stack(masks, axis=0), files

def ensure_same_shape(imgs, masks):
    D, H, W = imgs.shape
    if masks.shape != imgs.shape:
        from skimage.transform import resize
        new = []
        for m in masks:
            m2 = resize(m, (H, W), order=0, preserve_range=True, anti_aliasing=False)
            new.append((m2 > 0.5).astype(np.uint8))
        masks = np.stack(new, axis=0)
    return masks

def compute_centroids(masks):
    cents = []
    for m in masks:
        ys, xs = np.nonzero(m)
        cents.append((ys.mean(), xs.mean()))
    return np.array(cents)

def translate_slice(slice_img, dy, dx):
    H, W = slice_img.shape
    out = np.zeros_like(slice_img)
    # y-direction
    if dy >= 0:
        y0_src, y0_dst = 0, dy
        h = H - dy
    else:
        y0_src, y0_dst = -dy, 0
        h = H + dy
    # x-direction
    if dx >= 0:
        x0_src, x0_dst = 0, dx
        w = W - dx
    else:
        x0_src, x0_dst = -dx, 0
        w = W + dx
    out[y0_dst:y0_dst+h, x0_dst:x0_dst+w] = slice_img[y0_src:y0_src+h, x0_src:x0_src+w]
    return out

if __name__=="__main__":
    # 1) 读图
    imgs, files = load_image_volume(img_dir, ext=".png")

    # 2) 读掩码，若指定了 sample_json_path，则先从它读 H,W
    if sample_json_path:
        sample = json.load(open(sample_json_path, encoding="utf-8"))
        H, W  = sample["imageHeight"], sample["imageWidth"]
        masks, _ = load_mb_mask_volume(json_dir, H=H, W=W)
    else:
        masks, _ = load_mb_mask_volume(json_dir)

    masks = ensure_same_shape(imgs, masks)
    imgs  = imgs * masks  # 可选：只保留 main body 区域

    # 3) 计算所有切片的质心
    cents = compute_centroids(masks)

    # 4) 确定参考质心
    if sample_index is not None:
        ref_y, ref_x = cents[sample_index]
    elif sample_json_path:
        shp = next(s for s in sample["shapes"] if s["label"]=="main body")
        pts = np.array(shp["points"], float)
        ys, xs = polygon(pts[:,1], pts[:,0], shape=(H,W))
        ref_y, ref_x = ys.mean(), xs.mean()
    else:
        ref_y, ref_x = cents[0]

    # 5) 平移对齐并以 idx 命名保存
    os.makedirs(out_dir, exist_ok=True)
    for i, img in enumerate(imgs):
        dy      = int(round(ref_y - cents[i,0]))
        dx      = int(round(ref_x - cents[i,1]))
        aligned = translate_slice(img, dy, dx)
        out_p   = os.path.join(out_dir, f"{i}.png")
        Image.fromarray(aligned).save(out_p)

    print(f"Done: saved {len(imgs)} aligned slices in {out_dir} as 0.png … {len(imgs)-1}.png")
