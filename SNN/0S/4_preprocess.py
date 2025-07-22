import os
import json
import base64
import csv
import numpy as np
from io import BytesIO
from PIL import Image
from skimage.transform import AffineTransform, warp
from shapely.geometry import Polygon

def compute_consistent_orientation(points):
    pts = np.array(points, dtype=float)
    poly = Polygon(pts)
    min_rect = poly.minimum_rotated_rectangle
    rect = np.array(min_rect.exterior.coords)[:-1]

    # 找到最长的边
    edges = [(rect[i], rect[(i+1)%4]) for i in range(4)]
    lengths = [np.linalg.norm(p2-p1) for p1,p2 in edges]
    idx = int(np.argmax(lengths))
    p1, p2 = edges[idx]

    # 计算边的朝向角度
    raw = np.arctan2(p2[1]-p1[1], p2[0]-p1[0])
    # 归一化到 (-pi/2, pi/2]
    if raw > np.pi/2:
        raw -= np.pi
    elif raw <= -np.pi/2:
        raw += np.pi
    # 如果更接近垂直，则再旋转 ±90°
    if raw > np.pi/4:
        angle = raw - np.pi/2
    elif raw < -np.pi/4:
        angle = raw + np.pi/2
    else:
        angle = raw

    center = np.array(min_rect.centroid.coords[0])
    return angle, center

def rotate_and_center(image, shapes, angle, center, output_size=400):
    cx, cy = center
    # 构建仿射变换：先平移到原点，再旋转，再移回
    T = (AffineTransform(translation=(-cx, -cy))
         + AffineTransform(rotation=-angle)
         + AffineTransform(translation=(cx, cy)))

    # 旋转图像
    if image.ndim == 2:
        img_rot = warp(image, T.inverse, order=1, preserve_range=True)
    else:
        chans = [
            warp(image[..., c], T.inverse, order=1, preserve_range=True)
            for c in range(image.shape[2])
        ]
        img_rot = np.stack(chans, axis=-1)
    img_rot = img_rot.astype(image.dtype)

    # 旋转并更新所有 shape 点
    shapes_rot = []
    for shp in shapes:
        pts = np.array(shp['points'], dtype=float)
        pts_h = np.hstack([pts, np.ones((len(pts),1))])  # 齐次坐标
        pts_new = (T.params @ pts_h.T).T[:, :2]
        s2 = shp.copy()
        s2['points'] = pts_new.tolist()
        shapes_rot.append(s2)

    # 在中心画布上居中放置
    canvas = np.zeros(
        (output_size, output_size) + (() if image.ndim==2 else (image.shape[2],)),
        dtype=image.dtype
    )
    mb = next(s for s in shapes_rot if s.get('label') == 'main body')
    mb_c = np.array(mb['points'], dtype=float).mean(axis=0)
    dx = int(round(output_size/2 - mb_c[0]))
    dy = int(round(output_size/2 - mb_c[1]))

    x0, y0 = dx, dy
    x1, y1 = x0 + img_rot.shape[1], y0 + img_rot.shape[0]

    xs0, ys0 = max(0, x0), max(0, y0)
    xs1, ys1 = min(output_size, x1), min(output_size, y1)
    sx0, sy0 = xs0 - x0, ys0 - y0
    sx1, sy1 = sx0 + (xs1 - xs0), sy0 + (ys1 - ys0)

    canvas[ys0:ys1, xs0:xs1] = img_rot[sy0:sy1, sx0:sx1]

    # 更新 shapes 的坐标到画布上
    final_shapes = []
    for shp in shapes_rot:
        pts = np.array(shp['points'], dtype=float)
        pts[:, 0] += x0
        pts[:, 1] += y0
        s3 = shp.copy()
        s3['points'] = pts.tolist()
        final_shapes.append(s3)

    return canvas, final_shapes

def encode_image_to_base64(arr, fmt="PNG"):
    buf = BytesIO()
    Image.fromarray(arr).save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode('utf-8')


if __name__ == "__main__":
    file_name = '13_14_15_16'
    N = 16
    img_dir    = fr"D:\Study\Postgraduate\S2\Project\Code\Resource\Original\{file_name}\{N}"
    json_dir   = fr"E:\Project_SNV\0S\2_MB\{N}"
    out_img_d  = fr"E:\Project_SNV\0S\3_full_img\{N}"
    out_json_d = fr"E:\Project_SNV\0S\4_moved_MB\{N}"

    os.makedirs(out_img_d,  exist_ok=True)
    os.makedirs(out_json_d, exist_ok=True)

    # 打开 CSV 并写表头
    csv_path = os.path.join(out_json_d, "rotation_angles.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as csvf:
        writer = csv.writer(csvf)
        writer.writerow(["index", "angle_degrees"])

        idx = 0
        for fn in sorted(os.listdir(img_dir)):
            if not fn.lower().endswith(".tif"):
                continue

            base = os.path.splitext(fn)[0]
            img_p  = os.path.join(img_dir,  fn)
            json_p = os.path.join(json_dir, base + ".json")
            if not os.path.exists(json_p):
                print(f"跳过，无 JSON: {base}")
                continue

            # 读取图像和 JSON
            img = np.array(Image.open(img_p))
            with open(json_p, "r", encoding="utf-8") as f:
                data = json.load(f)

            shapes = data.get("shapes", [])
            mb = next((s for s in shapes if s.get("label")=="main body"), None)
            if mb is None:
                print(f"跳过，无 main body: {base}")
                continue

            # 计算旋转角度和中心
            angle, center = compute_consistent_orientation(mb["points"])
            deg = np.degrees(angle)
            writer.writerow([idx, f"{deg:.2f}"])

            # 旋转并居中
            img_out, shapes_out = rotate_and_center(img, shapes, angle, center, output_size=400)

            # 保存对齐后的图像 —— 用 idx 作为文件名
            out_img = os.path.join(out_img_d, f"{idx}.png")
            Image.fromarray(img_out).save(out_img)

            # 更新 JSON 并保存 —— 文件名也用 idx
            data["shapes"]     = shapes_out
            data["imageData"]  = encode_image_to_base64(img_out, fmt="PNG")
            data["imagePath"]  = os.path.basename(out_img)
            data["imageHeight"]= img_out.shape[0]
            data["imageWidth"] = img_out.shape[1]

            out_json = os.path.join(out_json_d, f"{idx}.json")
            with open(out_json, "w", encoding="utf-8") as fw:
                json.dump(data, fw, ensure_ascii=False, indent=2)

            print(f"[{idx}] 已处理：{base} → 输出 {idx}.png / {idx}.json ，旋转 {deg:.2f}°")
            idx += 1

    print(f"处理完毕，共 {idx} 张，旋转角度已记录到：{csv_path}")
