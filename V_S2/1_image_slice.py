import os
import cv2

def pad_to_center_multiple_of_50(img):
    """
    对一张灰度图 img (H,W)，以图像中心为基准，
    在上下左右均匀补黑边，使其新尺寸 (new_H, new_W) 恰好是 50 的倍数。
    返回补好边后的图像。
    """
    H, W = img.shape[:2]

    # 计算补齐到 50 的倍数后的尺寸
    new_H = ((H + 49) // 50) * 50
    new_W = ((W + 49) // 50) * 50

    pad_vert = new_H - H
    pad_horz = new_W - W

    pad_top    = pad_vert // 2
    pad_bottom = pad_vert - pad_top
    pad_left   = pad_horz // 2
    pad_right  = pad_horz - pad_left

    # 用黑色（0）边框补齐
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
    1. 先把 img pad 到中心对齐、边长都是 50 的倍数
    2. 从左上角(0,0)开始，每隔 50px 切一个 50×50 的小图
    3. 按 (row, col) 命名：prefix_row_col.png
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
    # —— 根据你的实际路径修改 —— #
    input_folder  = r"D:\Study\Postgraduate\S2\Project\Code\Resource\Original\1_2_3_4\1"
    output_folder = r"E:\Project_VAE\V2\Sliced_png\1_2_3_4\1"

    # 批量处理所有图像文件
    for fname in os.listdir(input_folder):
        if not fname.lower().endswith(('.png', '.jpg', '.tif', '.jpeg')):
            continue
        img_path = os.path.join(input_folder, fname)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[WARN] 无法读取图像: {img_path}")
            continue

        prefix = os.path.splitext(fname)[0]
        crop_50x50_from_center_padded(img, output_folder, prefix=prefix)

if __name__ == "__main__":
    main()
