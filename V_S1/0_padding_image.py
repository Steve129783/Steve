#!/usr/bin/env python3
# pad_images_and_embed_json.py

import os, math, json, base64, cv2
from copy import deepcopy
from io import BytesIO

def compute_padding(orig_w, orig_h, patch_size=50):
    new_w = math.ceil(orig_w / patch_size) * patch_size
    new_h = math.ceil(orig_h / patch_size) * patch_size
    pad_left  = (new_w - orig_w) // 2
    pad_top   = (new_h - orig_h) // 2
    return pad_left, pad_top, new_w, new_h

def pad_and_save_image(img_path, out_path, patch_size=50):
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"无法读取图像：{img_path}")
    h, w = img.shape[:2]
    pad_left, pad_top, new_w, new_h = compute_padding(w, h, patch_size)
    pad_right = new_w - w - pad_left
    pad_bottom= new_h - h - pad_top

    padded = cv2.copyMakeBorder(
        img,
        top=pad_top, bottom=pad_bottom,
        left=pad_left, right=pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=0
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, padded)
    return pad_left, pad_top, new_w, new_h

def find_json_for_base(base, json_folder):
    candidates = [f"{base}.json"]
    if not base.endswith("_defect"):
        candidates.append(f"{base}_defect.json")
    for name in candidates:
        path = os.path.join(json_folder, name)
        if os.path.exists(path):
            return path
    return None

def pad_and_embed_json(
    json_src, json_dst,
    pad_left, pad_top, new_w, new_h,
    padded_img_path
):
    # 1) 读原始 JSON
    with open(json_src, 'r', encoding='utf-8') as f:
        data = json.load(f)
    data = deepcopy(data)

    # 2) 平移所有标注点
    for shape in data.get("shapes", []):
        shape["points"] = [
            [x + pad_left, y + pad_top]
            for x,y in shape.get("points",[])
        ]

    # 3) 更新宽高
    data["imageWidth"]  = new_w
    data["imageHeight"] = new_h

    # 4) 嵌入填充后的图像到 imageData
    with open(padded_img_path, 'rb') as pf:
        b64 = base64.b64encode(pf.read()).decode('utf-8')
    data["imageData"] = b64
    # 可以把 imagePath 置为新的文件名或保留原名
    data["imagePath"] = os.path.basename(padded_img_path)

    # 5) 写 JSON
    os.makedirs(os.path.dirname(json_dst), exist_ok=True)
    with open(json_dst, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def main():
    # ——— 修改成你自己的目录 ———
    input_img_folder   = r"D:\Study\Postgraduate\S2\Project\Code\Resource\Original\1_2_3_4\1"
    input_json_folder  = r"D:\Study\Postgraduate\S2\Project\Code\Resource\Middle Stage D\1_2_3_4\1"
    output_img_folder  = r"E:\Project_VAE\V2\padded_image\1_2_3_4\1"
    output_json_folder = r"E:\Project_VAE\V2\moved_json\1_2_3_4\1"
    patch_size = 50
    # ————————————————————

    for fname in os.listdir(input_img_folder):
        if not fname.lower().endswith(('.png','.jpg','.jpeg','.tif','.tiff')):
            continue
        base = os.path.splitext(fname)[0]
        src_img = os.path.join(input_img_folder, fname)
        dst_img = os.path.join(output_img_folder, fname)

        # — pad 图像 — 
        try:
            pad_left, pad_top, new_w, new_h = pad_and_save_image(
                src_img, dst_img, patch_size
            )
        except Exception as e:
            print(f"[ERROR] 无法 pad 图像 {src_img}: {e}")
            continue

        # — 找 JSON — 
        js_src = find_json_for_base(base, input_json_folder)
        if not js_src:
            print(f"[WARN] 没有找到对应 JSON: {base}")
            continue
        js_dst = os.path.join(output_json_folder, os.path.basename(js_src))

        # — pad & embed JSON — 
        try:
            pad_and_embed_json(
                js_src, js_dst,
                pad_left, pad_top, new_w, new_h,
                dst_img
            )
        except Exception as e:
            print(f"[ERROR] 无法处理 JSON {js_src}: {e}")

    print("全部处理完毕！")

if __name__=="__main__":
    main()
