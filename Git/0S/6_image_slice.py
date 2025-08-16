#!/usr/bin/env python3
import os
import cv2

def pad_to_center_multiple_of_50(img):
    """
    Given a grayscale image `img` of shape (H, W),
    pad the borders with black (pixel value 0) so that
    the output size (new_H, new_W) are both multiples of 50,
    while keeping the original image centered.
    Returns the padded image.
    """
    H, W = img.shape[:2]

    # Compute target dimensions (round up to next multiple of 50)
    new_H = ((H + 49) // 50) * 50
    new_W = ((W + 49) // 50) * 50

    pad_vert = new_H - H
    pad_horz = new_W - W

    # Split padding evenly between top/bottom and left/right
    pad_top    = pad_vert // 2
    pad_bottom = pad_vert - pad_top
    pad_left   = pad_horz // 2
    pad_right  = pad_horz - pad_left

    # Apply constant black border
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
    1. Pad `img` so that both dimensions are multiples of 50.
    2. Slide a 50×50 window from the top-left corner (0,0) with step size 50.
    3. Save each patch into `output_folder` with filename `{prefix}_{row}_{col}.png`.
    """
    padded = pad_to_center_multiple_of_50(img)
    H_p, W_p = padded.shape[:2]

    os.makedirs(output_folder, exist_ok=True)

    for y in range(0, H_p, 50):
        for x in range(0, W_p, 50):
            patch = padded[y:y+50, x:x+50]
            row = y // 50
            col = x // 50
            out_name = f"{prefix}_{row}_{col}.png"
            out_path = os.path.join(output_folder, out_name)
            cv2.imwrite(out_path, patch)


def main():
    # —— Modify these paths as needed —— #
    N = 1
    input_folder  = rf"E:\Project_SNV\0S\5_aligned_img\{N}"
    output_folder = rf"E:\Project_SNV\0S\6_patch\T{N}"

    # Traverse all image files in the input folder
    for fname in os.listdir(input_folder):
        # Only process common image formats
        if not fname.lower().endswith(('.png', '.jpg', '.tif', '.jpeg')):
            continue
        img_path = os.path.join(input_folder, fname)
        # Read image in grayscale (unchanged if already grayscale)
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"[WARN] Failed to read image: {img_path}")
            continue

        prefix = os.path.splitext(fname)[0]
        crop_50x50_from_center_padded(img, output_folder, prefix=prefix)

if __name__ == "__main__":
    main()
