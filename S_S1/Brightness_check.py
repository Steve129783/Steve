import numpy as np
import cv2
import json
import pandas as pd
from pathlib import Path
from PIL import Image, ImageDraw

# 1. 指定两个文件夹：一个存放 .tif/.png 图像，一个存放对应的 .json 标注
tif_dir = Path(r'E:\Project_VAE\50_slice_padding\1_2_3_4\1c')        # 存放图像
json_dir = Path(r'D:\Study\Postgraduate\S2\Project\Code\Resource\Middle Stage B\1_2_3_4\1')  # 存放标注

# 收集所有图像文件名
fnames = sorted([p for p in tif_dir.iterdir() if p.suffix.lower() in ('.tif', '.tiff', '.png', '.jpg', '.jpeg')])

# 2. 用于存储结果的列表
results = []

for image_path in fnames:
    # 2.1 根据图像名称在 json_dir 中寻找同名 .json
    json_path = json_dir / f"{image_path.stem}.json"

    # 2.2 读取灰度图
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        print(f"Failed to load image {image_path.name}")
        continue
    h, w = gray.shape

    # 2.3 如果没有对应的 JSON，则把整图当作“主体区域”
    if not json_path.exists():
        body_pixels = gray.flatten()
    else:
        # 2.4 加载 JSON 标注
        with json_path.open('r', encoding='utf-8') as f:
            ann = json.load(f)

        # 2.5 根据 “main body” 生成掩膜 mask
        mask = np.zeros((h, w), dtype=np.uint8)
        for shape in ann.get("shapes", []):
            if shape.get("label", "").lower() == "main body":
                tmp = Image.new("L", (w, h), 0)
                draw = ImageDraw.Draw(tmp)
                pts = [tuple(p) for p in shape.get("points", [])]
                draw.polygon(pts, fill=1)
                mask[np.array(tmp, bool)] = 1

        # 2.6 提取主体区域像素
        body_pixels = gray[mask == 1]
        if body_pixels.size == 0:
            body_pixels = gray.flatten()

    # 3. 计算统计量：平均、最大、中值、70%分位、最小（新增）
    if body_pixels.size == 0:
        mean_val    = float('nan')
        max_val     = float('nan')
        min_val     = float('nan')    # 新增
        median_val  = float('nan')
        perc70_val  = float('nan')
    else:
        mean_val    = float(body_pixels.mean())
        max_val     = int(body_pixels.max())
        min_val     = int(body_pixels.min())         # 新增
        median_val  = float(np.median(body_pixels))
        perc70_val  = float(np.percentile(body_pixels, 70))

    # 4. 存入结果列表
    results.append({
        "image_name": image_path.name,
        "mean_brightness": mean_val,
        "max_brightness": max_val,
        "min_brightness": min_val,                 # 新增
        "median_brightness": median_val,
        "percentile_70_brightness": perc70_val
    })

# 5. 转换为 DataFrame，并打印/保存
df = pd.DataFrame(results)
print(df)

# 如果需要保存成 CSV：
output_csv = Path(r'E:\Project_VAE') / "brightness_stats2.csv"
df.to_csv(output_csv, index=False)
print(f"Saved brightness stats to: {output_csv}")