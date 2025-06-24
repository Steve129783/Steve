import os
import json
import base64
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection, box

def embed_existing_patches_in_json(
    labelme_json_path: str,
    patch_img_folder:    str,
    output_folder:       str,
    patch_size:          int = 50
):
    """
    对应 patch_img_folder 里切好的 <base>_row_col.png，
    将其 Base64 嵌入 JSON，同时根据原始标注裁剪多边形，
    并写入 patch_row/patch_col、instance_id。
    """
    with open(labelme_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 取原始文件 base name，并去掉可能的 "_defect"
    base = os.path.splitext(os.path.basename(labelme_json_path))[0]
    if base.endswith("_defect"):
        base = base[:-len("_defect")]

    # 给每个 shape 分配 instance_id
    for iid, shape in enumerate(data['shapes']):
        shape['instance_id'] = iid

    orig_w = data['imageWidth']
    orig_h = data['imageHeight']
    shapes = data['shapes']
    os.makedirs(output_folder, exist_ok=True)

    # 计算与图像切片脚本一致的中心填充量
    pad_h = ((orig_h + patch_size - 1) // patch_size) * patch_size - orig_h
    pad_w = ((orig_w + patch_size - 1) // patch_size) * patch_size - orig_w
    new_H = orig_h + pad_h
    new_W = orig_w + pad_w

    # 遍历每一个 patch 网格
    for y in range(0, new_H, patch_size):
        for x in range(0, new_W, patch_size):
            pr = y // patch_size  # patch row
            pc = x // patch_size  # patch col

            # patch 在原图上的实际位置
            minx = x - pad_w // 2
            miny = y - pad_h // 2
            patch_box = box(minx, miny, minx + patch_size, miny + patch_size)

            # 裁剪所有跨此 patch 的 polygon
            new_shapes = []
            for shape in shapes:
                pts = shape['points']
                if len(pts) < 3:
                    continue
                try:
                    poly = Polygon(pts)
                    if not poly.is_valid:
                        poly = poly.buffer(0)
                except:
                    continue

                try:
                    inter = poly.intersection(patch_box)
                except:
                    try:
                        inter = poly.buffer(0).intersection(patch_box)
                    except:
                        continue

                if inter.is_empty:
                    continue

                # 提取所有子 Polygon
                polys = []
                if isinstance(inter, Polygon):
                    polys = [inter]
                elif isinstance(inter, MultiPolygon):
                    polys = list(inter.geoms)
                elif isinstance(inter, GeometryCollection):
                    for g in inter.geoms:
                        if isinstance(g, Polygon):
                            polys.append(g)

                for p in polys:
                    coords = list(p.exterior.coords)
                    if len(coords) < 4:
                        continue
                    # 转到局部坐标
                    local_pts = [[float(px - minx), float(py - miny)]
                                 for px, py in coords]
                    new_shapes.append({
                        'label':       shape['label'],
                        'instance_id': shape['instance_id'],
                        'points':      local_pts,
                        'group_id':    shape.get('group_id'),
                        'shape_type':  'polygon',
                        'flags':       shape.get('flags', {}),
                    })

            # 如果这个 patch 上有标注，则嵌入 imageData 并写 JSON
            if not new_shapes:
                continue

            # 对应的切片图名必须和你的 cv2 切图脚本一致
            patch_name = f"{base}_{pr}_{pc}.png"
            patch_path = os.path.join(patch_img_folder, patch_name)
            if not os.path.exists(patch_path):
                print(f"[WARN] missing patch image: {patch_path}")
                continue

            with open(patch_path, 'rb') as pf:
                b64_str = base64.b64encode(pf.read()).decode('utf-8')

            out = {
                'version':      data.get('version', '5.0.1'),
                'flags':        data.get('flags', {}),
                'patch_row':    pr,
                'patch_col':    pc,
                'imagePath':    patch_name,
                'imageData':    b64_str,
                'imageHeight':  patch_size,
                'imageWidth':   patch_size,
                'shapes':       new_shapes,
            }

            out_name = f"{base}_{pr}_{pc}.json"
            with open(os.path.join(output_folder, out_name),
                      'w', encoding='utf-8') as fw:
                json.dump(out, fw, ensure_ascii=False, indent=2)


def main():
    # 修改为你实际的路径
    input_json_folder  = r"D:\Study\Postgraduate\S2\Project\Code\Resource\Middle Stage D\1_2_3_4\1"
    patch_img_folder   = r"E:\Project_VAE\V2\Sliced_png\1_2_3_4\1"
    output_json_folder = r"E:\Project_VAE\V2\Sliced_json\1_2_3_4\1"

    for fn in os.listdir(input_json_folder):
        if not fn.lower().endswith('.json'):
            continue
        embed_existing_patches_in_json(
            os.path.join(input_json_folder, fn),
            patch_img_folder,
            output_json_folder,
            patch_size=50
        )

if __name__ == '__main__':
    main()
