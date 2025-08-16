#!/usr/bin/env python3
import os
import torch
import numpy as np
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.pyplot as plt

# ================ Configuration ===================
root_dir        = r"E:\Project_SNV\0S\6_patch"
output_cache    = r"E:\Project_SNV\1N\1_Pack\120\data_cache.pt"
num_workers     = os.cpu_count() or 4
include_groups  = ['1','2','3','4','9','10','11','12','13','14','15','16','17','18','19','20']

def process_patch(args):
    img_path, gid = args
    img = Image.open(img_path)
    arr = np.array(img)
    if arr.dtype not in (np.uint8, np.uint16):
        raise ValueError(f"Unsupported image dtype {arr.dtype} in {img_path!r}")
    
    # per‐patch min–max normalization
    max_val = np.iinfo(arr.dtype).max
    arr = arr.astype(np.float32) / max_val

    h, w = arr.shape[:2]
    assert (h, w) == (50, 50), f"Unexpected patch size {h}×{w} in {img_path}"
    tensor = torch.from_numpy(arr)[None, ...]
    return img_path, tensor, gid

def load_patches_parallel(root_dir, include_groups=None):
    # collect tasks
    all_dirs = sorted(
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    )
    groups = [d for d in all_dirs if include_groups and d in include_groups] or all_dirs
    tasks, group_map = [], {}
    for gid, group in enumerate(groups):
        group_map[gid] = group
        subdir = os.path.join(root_dir, group)
        for fname in sorted(os.listdir(subdir)):
            if fname.lower().endswith(".png"):
                tasks.append((os.path.join(subdir, fname), gid))

    # process in parallel
    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_patch, t): t for t in tasks}
        for fut in as_completed(futures):
            img_path, tensor, gid = fut.result()
            results.append((img_path, tensor, gid))

    # sort by filepath to ensure deterministic order
    results.sort(key=lambda x: x[0])

    # unpack into lists
    paths      = [r[0] for r in results]
    images     = [r[1] for r in results]
    group_ids  = [r[2] for r in results]
    images_tensor = torch.stack(images) if images else torch.empty(0)
    return images_tensor, group_ids, group_map, paths

if __name__ == '__main__':
    # 1. Load and normalize patches (in deterministic order)
    images, group_ids, group_map, paths = load_patches_parallel(root_dir, include_groups)
    arrs = images.numpy().squeeze(1)  # shape (N,50,50)

    # 2. Plot original (normalized) histograms
    plt.figure(figsize=(8, 6))
    for gid, name in group_map.items():
        pixels = arrs[np.array(group_ids) == gid].ravel()
        plt.hist(pixels, bins=50, range=(0, 1), density=True,
                 histtype='step', label=f"Group {name}")
    plt.xlabel("Normalized brightness")
    plt.ylabel("Density")
    plt.title("Original per-group brightness histograms")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 3. Compute per-patch global statistics
    N = arrs.shape[0]
    flat = arrs.reshape(N, -1)
    stat_min    = flat.min(axis=1)
    stat_max    = flat.max(axis=1)
    stat_median = np.median(flat, axis=1)
    stat_var    = flat.var(axis=1)

    stats = {
        'Min':    stat_min,
        'Max':    stat_max,
        'Median': stat_median,
        'Variance': stat_var,
    }

    # 4. Plot histograms of these stats per group
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    for ax, (stat_name, values) in zip(axes, stats.items()):
        for gid, gname in group_map.items():
            grp_vals = values[np.array(group_ids) == gid]
            ax.hist(grp_vals, bins=30, density=True,
                    histtype='step', label=f"Group {gname}")
        ax.set_title(f"{stat_name} Distribution")
        ax.set_xlabel(stat_name)
        ax.set_ylabel("Density")
        ax.legend()
    plt.tight_layout()
    plt.show()

    # 5. Save cache (with deterministic ordering)
    cache = {"images": images, "group_ids": group_ids, "paths": paths}
    os.makedirs(os.path.dirname(output_cache), exist_ok=True)
    torch.save(cache, output_cache)
    print(f"Packed {len(images)} patches into {output_cache}")
    for gid, name in group_map.items():
        count = group_ids.count(gid)
        print(f"  {gid}: {name} ({count} patches)")
