import os

def delete_all_images_in_folder(folder):
    """
    递归遍历 folder 及其所有子目录，
    将所有扩展名为 .png/.jpg/.jpeg/.tif/.json/.pt/.7z/.001/.002/.npz 的文件永久删除（不进回收站）。
    """
    for root, dirs, files in os.walk(folder):
        for fname in files:
            if not fname.lower().endswith((
                '.png', '.jpg', '.jpeg', '.tif', '.json',
                '.pt', '.7z', '.001', '.002', '.npz'
            )):
                continue
            path = os.path.join(root, fname)
            if os.path.isfile(path):
                try:
                    os.remove(path)  # 直接永久删除
                    print(f"Deleted: {path}")
                except Exception as e:
                    print(f"无法删除 {path}：{e}")

# 用法示例
output_folder = r"E:\Project_SNV\0S\5_aligned_img"
delete_all_images_in_folder(output_folder)
