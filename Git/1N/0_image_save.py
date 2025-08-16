from PIL import Image, ImageSequence
import os
import math

# Input path to the multi-page TIFF file
tif_path  = r'E:\Project_CNN\CT_DATA\Sam\14_LowEnergyCropped.tif'
# Directory to save each full cropped frame
full_dir  = r'E:\Project_SNV\0S\3_full_img\l'
# Directory to save 50×50 patches from each frame
patch_dir = r'E:\Project_SNV\0S\6_patch\l'

os.makedirs(full_dir, exist_ok=True)
os.makedirs(patch_dir, exist_ok=True)

# Target patch dimensions
base_h, base_w = 50, 50

# Open the multi-page TIFF
with Image.open(tif_path) as im:
    # Iterate over each frame/page
    for idx, page in enumerate(ImageSequence.Iterator(im)):
        w, h = page.size
        print(f"Frame {idx:03d}: original size = {w}×{h}")

        # Calculate the maximal region divisible by the patch size
        n_cols = w // base_w
        n_rows = h // base_h
        crop_w = n_cols * base_w
        crop_h = n_rows * base_h

        # Skip if the image is smaller than one patch
        if n_cols == 0 or n_rows == 0:
            print("  Skipped: image smaller than 50×50")
            continue

        # Crop the top-left region that divides evenly into 50×50 patches
        img_cropped = page.crop((0, 0, crop_w, crop_h))
        print(f"  Cropped area = {crop_w}×{crop_h} (discarded right {w-crop_w}px, bottom {h-crop_h}px)")

        # Save the cropped full-frame image
        full_name = f"{idx:03d}_cropped_{crop_w}x{crop_h}.png"
        img_cropped.save(os.path.join(full_dir, full_name))
        print(f"    Saved full cropped image: {full_name}")

        # Generate and save each 50×50 patch
        count = 0
        for y in range(0, crop_h, base_h):
            for x in range(0, crop_w, base_w):
                patch = img_cropped.crop((x, y, x + base_w, y + base_h))
                name  = f"{idx:03d}_x{x}_y{y}.png"
                patch.save(os.path.join(patch_dir, name))
                count += 1

        print(f"    Saved {count} patches to {patch_dir}")