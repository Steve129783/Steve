import os

def delete_all_images_in_folder(folder):
    """
    直接将 folder 下的所有 .png/.jpg/.jpeg/.tif 文件永久删除（不进回收站）。
    """
    for fname in os.listdir(folder):
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.json', '.pt', '.7z','.001','.002', '.npy')):
            continue
        path = os.path.join(folder, fname)
        if os.path.isfile(path):
            try:
                os.remove(path)  # 直接永久删除
            except Exception as e:
                print(f"无法删除 {path}：{e}")

# 用法示例
output_folder = r"E:\Project_VAE\Sliced_json\1_2_3_4\1"
delete_all_images_in_folder(output_folder)
