from PIL import Image
import os

def clean_corrupted(folder):
    bad = []
    # 1) 遍历所有 PNG/JPG 文件，检测损坏
    for root, _, files in os.walk(folder):
        for fn in files:
            if not fn.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            path = os.path.join(root, fn)
            try:
                with Image.open(path) as im:
                    im.verify()
            except Exception:
                bad.append(path)
    # 2) 打印并删除
    if bad:
        print('Removing corrupted files:')
        for p in bad:
            print('  ', p)
            try:
                os.remove(p)
            except Exception as e:
                print(f'Failed to delete {p}: {e}')
    else:
        print('No corrupted files found in', folder)

# 针对你的各个子目录调用
clean_corrupted(r'E:\Project_CNN\1_training\train\low')
clean_corrupted(r'E:\Project_CNN\1_training\train\high')
clean_corrupted(r'E:\Project_CNN\1_training\val')
clean_corrupted(r'E:\Project_CNN\1_training\test')
