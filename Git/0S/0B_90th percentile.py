import os
from pathlib import Path
from PIL import Image
import numpy as np

def analyze_sizes_in_folder(folder_path):
    """
    Scan all image files in a folder, retrieve their widths and heights,
    and compute statistics including the 90th percentile.
    """
    # Supported image file extensions
    img_exts = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
    widths, heights = [], []
    
    folder = Path(folder_path)
    for img_path in folder.iterdir():
        if img_path.suffix.lower() in img_exts:
            try:
                # Open the image and get its size (width, height)
                with Image.open(img_path) as img:
                    w, h = img.size
                    widths.append(w)
                    heights.append(h)
            except Exception as e:
                print(f"⚠️ Could not open {img_path.name}: {e}")
    
    if not heights:
        print("No supported images found in the specified folder.")
        return None, None
    
    # Convert lists to NumPy arrays for statistical calculations
    widths = np.array(widths)
    heights = np.array(heights)
    
    def stats(arr):
        return {
            'min': arr.min(),
            'max': arr.max(),
            'mean': float(arr.mean()),
            'median': float(np.median(arr)),
            'p90': float(np.percentile(arr, 90))
        }
    
    w_stats = stats(widths)
    h_stats = stats(heights)
    
    # Print width statistics
    print("Width statistics (pixels):")
    print(f"  Min: {w_stats['min']}")
    print(f"  Max: {w_stats['max']}")
    print(f"  Mean: {w_stats['mean']:.2f}")
    print(f"  Median: {w_stats['median']}")
    print(f"  90th percentile: {w_stats['p90']}")
    
    # Print height statistics
    print("\nHeight statistics (pixels):")
    print(f"  Min: {h_stats['min']}")
    print(f"  Max: {h_stats['max']}")
    print(f"  Mean: {h_stats['mean']:.2f}")
    print(f"  Median: {h_stats['median']}")
    print(f"  90th percentile: {h_stats['p90']}")
    
    # Return the 90th percentile dimensions as integers
    return int(w_stats['p90']), int(h_stats['p90'])

if __name__ == '__main__':
    folder = r"D:\Study\Postgraduate\S2\Project\Code\Resource\Original\17_18_19_20\19"
    w90, h90 = analyze_sizes_in_folder(folder)
    if w90 and h90:
        print(f"\nRecommended uniform size (based on 90th percentile): {w90}×{h90} px")