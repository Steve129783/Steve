import os
from pathlib import Path
from PIL import Image
import numpy as np

def analyze_sizes_in_folder(folder_path):
    """
    扫描文件夹内所有图片文件，读取它们的宽度和高度并计算统计信息及 90% 分位数。
    """
    img_exts = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
    widths, heights = [], []
    
    folder = Path(folder_path)
    for img_path in folder.iterdir():
        if img_path.suffix.lower() in img_exts:
            try:
                with Image.open(img_path) as img:
                    w, h = img.size  # PIL.Image.size 返回 (width, height)
                    widths.append(w)
                    heights.append(h)
            except Exception as e:
                print(f"⚠️ 无法打开 {img_path.name}：{e}")
    
    if not heights:
        print("在指定文件夹中未检测到任何支持的图片。")
        return None, None
    
    widths  = np.array(widths)
    heights = np.array(heights)
    
    def stats(arr):
        return {
            'min':    arr.min(),
            'max':    arr.max(),
            'mean':   float(arr.mean()),
            'median': float(np.median(arr)),
            'p90':    float(np.percentile(arr, 90))
        }
    
    w_stats = stats(widths)
    h_stats = stats(heights)
    
    print("宽度统计信息（单位：像素）：")
    print(f"  最小值: {w_stats['min']}")
    print(f"  最大值: {w_stats['max']}")
    print(f"  平均值: {w_stats['mean']:.2f}")
    print(f"  中位数: {w_stats['median']}")
    print(f"  90% 分位数: {w_stats['p90']}")
    
    print("\n高度统计信息（单位：像素）：")
    print(f"  最小值: {h_stats['min']}")
    print(f"  最大值: {h_stats['max']}")
    print(f"  平均值: {h_stats['mean']:.2f}")
    print(f"  中位数: {h_stats['median']}")
    print(f"  90% 分位数: {h_stats['p90']}")
    
    # 返回 (width_90pct, height_90pct)
    return int(w_stats['p90']), int(h_stats['p90'])

if __name__ == '__main__':
    folder = r"D:\Study\Postgraduate\S2\Project\Code\Resource\Original\9_10_11_12\9"
    w90, h90 = analyze_sizes_in_folder(folder)
    if w90 and h90:
        print(f"\n建议统一的尺寸（参考 90% 分位数）: {w90}×{h90} px")
