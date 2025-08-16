#!/usr/bin/env python3
import os
import json
import numpy as np
from PIL import Image
from skimage.draw import polygon

# —— Explicit configuration section ——
N = 20
# Input folder of original images
img_dir      = fr"E:\Project_SNV\0S\3_full_img\{N}"
# Corresponding JSON annotation folder (with "main body" already moved)
json_dir     = fr"E:\Project_SNV\0S\4_moved_MB\{N}"
# Output folder for aligned images
out_dir      = fr"E:\Project_SNV\0S\5_aligned_img\{N}"

# Optional: use a sample JSON to obtain reference dimensions; set to None if not needed
sample_json_path = r'E:\Project_SNV\0S\4_moved_MB\3\0.json'
# Optional: specify a slice index (0-based) as the reference; set to None if not needed
sample_index     = None
# —— End of configuration ——

def load_image_volume(img_dir, ext=".png"):
    """
    Load a sequence of images from a directory, return a 3D NumPy array (D, H, W)
    and the list of filenames.
    """
    files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(ext)])
    arrs  = [np.array(Image.open(os.path.join(img_dir, f))) for f in files]
    return np.stack(arrs, axis=0), files  # (D, H, W)


def load_mb_mask_volume(json_dir, H=None, W=None):
    """
    Load "main body" masks from LabelMe JSON files.
    Returns a binary mask volume (D, H, W) and the list of filenames.
    If H and W are not provided, they are taken from the first JSON file.
    """
    files = sorted([f for f in os.listdir(json_dir) if f.lower().endswith(".json")])
    if H is None or W is None:
        sample = json.load(open(os.path.join(json_dir, files[0]), encoding="utf-8"))
        H, W = sample["imageHeight"], sample["imageWidth"]

    masks = []
    for fn in files:
        data = json.load(open(os.path.join(json_dir, fn), encoding="utf-8"))
        mb   = [s for s in data["shapes"] if s["label"] == "main body"]
        mask = np.zeros((H, W), np.uint8)
        for shp in mb:
            pts    = np.array(shp["points"], float)
            rr, cc = polygon(pts[:,1], pts[:,0], shape=mask.shape)
            mask[rr, cc] = 1
        masks.append(mask)
    return np.stack(masks, axis=0), files  # (D, H, W)


def ensure_same_shape(imgs, masks):
    """
    Ensure that the mask volume and the image volume have the same shape.
    If not, resize the masks accordingly.
    """
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
    """
    Compute centroid (row, col) for each mask slice.
    Returns an array of shape (D, 2).
    """
    cents = []
    for m in masks:
        ys, xs = np.nonzero(m)
        cents.append((ys.mean(), xs.mean()))
    return np.array(cents)


def translate_slice(slice_img, dy, dx):
    """
    Translate a 2D image slice by (dy, dx).
    Empty regions are filled with zeros.
    """
    H, W = slice_img.shape
    out = np.zeros_like(slice_img)
    # Y-axis source/target ranges
    if dy >= 0:
        y0_src, y0_dst = 0, dy
        h = H - dy
    else:
        y0_src, y0_dst = -dy, 0
        h = H + dy
    # X-axis source/target ranges
    if dx >= 0:
        x0_src, x0_dst = 0, dx
        w = W - dx
    else:
        x0_src, x0_dst = -dx, 0
        w = W + dx
    # Copy
    out[y0_dst:y0_dst+h, x0_dst:x0_dst+w] = slice_img[y0_src:y0_src+h, x0_src:x0_src+w]
    return out

if __name__ == "__main__":
    # 1) Load image volume
    imgs, files = load_image_volume(img_dir, ext=".png")

    # 2) Load mask volume and dimensions
    if sample_json_path:
        sample = json.load(open(sample_json_path, encoding="utf-8"))
        H, W  = sample["imageHeight"], sample["imageWidth"]
        masks, _ = load_mb_mask_volume(json_dir, H=H, W=W)
    else:
        masks, _ = load_mb_mask_volume(json_dir)

    # 3) Ensure shape consistency and optionally mask background
    masks = ensure_same_shape(imgs, masks)
    imgs  = imgs * masks  # Optional: keep only main body, zero out background

    # 4) Compute centroids
    cents = compute_centroids(masks)

    # 5) Select reference centroid
    if sample_index is not None:
        ref_y, ref_x = cents[sample_index]
    elif sample_json_path:
        shp = next(s for s in sample["shapes"] if s["label"]=="main body")
        pts = np.array(shp["points"], float)
        ys, xs = polygon(pts[:,1], pts[:,0], shape=(H, W))
        ref_y, ref_x = ys.mean(), xs.mean()
    else:
        ref_y, ref_x = cents[0]

    # 6) Translate slices for alignment and save
    os.makedirs(out_dir, exist_ok=True)
    for i, img in enumerate(imgs):
        dy      = int(round(ref_y - cents[i,0]))
        dx      = int(round(ref_x - cents[i,1]))
        aligned = translate_slice(img, dy, dx)
        out_p   = os.path.join(out_dir, f"{i}.png")
        Image.fromarray(aligned).save(out_p)

    print(f"Done: saved {len(imgs)} aligned slices to {out_dir} (0.png … {len(imgs)-1}.png)")
