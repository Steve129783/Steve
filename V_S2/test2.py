#!/usr/bin/env python3
# preview_padded_rois.py

import torch
import PIL.Image
import numpy as np
import matplotlib.pyplot as plt

DATA_PATH = r"E:\Project_VAE\V2\Reorganization\1_2_3_4\1\reassembled_defects.pt"

def main():
    # 1) allow PIL.Image unpickling
    torch.serialization.add_safe_globals([PIL.Image.Image])
    data = torch.load(DATA_PATH, map_location='cpu', weights_only=False)
    
    # 2) print a brief demo of what's inside the .pt
    print(f"Loaded {len(data)} instances. Showing demo of first 3 entries:\n")
    for i in range(min(3, len(data))):
        entry = data[i]
        print(f"Instance #{i}:")
        print("  keys:", list(entry.keys()))
        print("  base:", entry['base'])
        print("  instance_id:", entry['instance_id'])
        print("  padded_bbox:", entry['padded_bbox'])
        print("  contributing_patches:", entry['contributing_patches'])
        print("  patch_jsons:", entry['patch_jsons'])
        print("  source_tif:", entry['source_tif'])
        # inspect image
        img = entry['image']
        arr = np.array(img)
        print("  image.mode:", img.mode, "shape:", arr.shape, 
              "unique pixel values (first 10):", np.unique(arr)[:10])
        # inspect mask
        mask = entry['mask']
        print("  mask shape:", mask.shape, 
              "unique mask values:", np.unique(mask))
        print()

    # 3) extract ROI images for visualization
    rois = [item['image'] for item in data]
    N = min(16, len(rois))

    # 4) plot first 16
    fig, axes = plt.subplots(4, 4, figsize=(8,8))
    for i in range(N):
        ax = axes[i//4, i%4]
        ax.imshow(rois[i], cmap='gray')
        ax.set_title(f"#{i}")
        ax.axis('off')
    for j in range(N, 16):
        axes[j//4, j%4].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
