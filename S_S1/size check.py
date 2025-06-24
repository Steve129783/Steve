import os
import cv2

def detect_jump_images(image_dir, threshold=10):
    image_paths = sorted([
        os.path.join(image_dir, fname)
        for fname in os.listdir(image_dir)
        if fname.endswith((".tif", ".png", ".jpg"))
    ])
    
    prev_size = None
    for path in image_paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[⚠️] Failed to read image: {path}")
            continue
        h, w = img.shape
        if prev_size:
            ph, pw = prev_size
            if abs(h - ph) > threshold or abs(w - pw) > threshold:
                print(f" Jump detected: {os.path.basename(path)} "
                      f"({h}x{w}) vs prev ({ph}x{pw})")
        prev_size = (h, w)

# 示例用法
detect_jump_images(
    image_dir=r"D:\Study\Postgraduate\S2\Project\R\Important materials for  Dissertation\Raw data part\ALLTIF\1_2_3_4\1",
    threshold=5  # 你可以调整这个跳变阈值
)
