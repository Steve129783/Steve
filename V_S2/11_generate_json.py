#!/usr/bin/env python3
# generate_labelme_jsons.py

import os
import json
import pickle
import numpy as np
import base64
from collections import defaultdict
from PIL import Image

# ─────────── 配置区 ───────────
INST_META_PKL = r"E:\Project_VAE\V2\Clusters\inst_meta.pkl"
LABELS_NPY    = r"E:\Project_VAE\V2\Clusters\reassembled_cluster_labels.npy"
IMAGE_DIR     = r"E:\Project_VAE\V2\padded_image\1_2_3_4\1"       # 原图所在文件夹
OUTPUT_DIR    = r"E:\Project_VAE\Output\1_2_3_4\1"  # JSON 输出目录


def load_data(meta_pkl_path, labels_npy_path):
    """
    从 pickle 和 npy 文件中加载元数据列表和标签数组
    """
    with open(meta_pkl_path, 'rb') as f:
        meta_list = pickle.load(f)
    labels = np.load(labels_npy_path)
    return meta_list, labels


def encode_image_to_base64(image_path):
    """
    读取图像文件并返回纯 Base64 编码字符串（不含 data URI 前缀）
    """
    with open(image_path, 'rb') as img_f:
        data = img_f.read()
    b64 = base64.b64encode(data).decode('ascii')
    return b64


def save_labelme_json(meta_list, labels, img_dir, out_dir):
    """
    根据 meta_list 和 labels 生成 LabelMe 格式的 JSON 文件
    每个原图对应一个 .json，包含所有缺陷实例的矩形框和标签
    imageData 内嵌纯 Base64，LabelMe 可直接打开
    """
    os.makedirs(out_dir, exist_ok=True)

    # 按原图文件名分组
    grouped = defaultdict(list)
    for md, lbl in zip(meta_list, labels):
        fname = os.path.basename(md.get('source_tif') or md['base'])
        grouped[fname].append({
            'bbox': md['padded_bbox'],  # [r0, c0, r1, c1]
            'cluster': int(lbl) + 1
        })

    count = 0
    for fname, items in grouped.items():
        img_path = os.path.join(img_dir, fname)
        if not os.path.isfile(img_path):
            print(f"⚠️ 未找到图片：{img_path}，跳过")
            continue

        # 读取图片尺寸
        with Image.open(img_path) as img:
            width, height = img.size

        # 编码整图为纯 Base64
        img_b64 = encode_image_to_base64(img_path)

        # 构造 shapes 列表
        shapes = []
        for it in items:
            r0, c0, r1, c1 = it['bbox']
            shapes.append({
                "label": str(it['cluster']),
                "points": [[c0, r0], [c1, r1]],
                "group_id": None,
                "shape_type": "rectangle",
                "flags": {}
            })

        # LabelMe JSON 结构
        labelme_json = {
            "version": "5.0.1",
            "flags": {},
            "shapes": shapes,
            "imagePath": fname,
            "imageData": img_b64,
            "imageHeight": height,
            "imageWidth": width
        }

        # 保存 JSON
        json_path = os.path.join(out_dir, os.path.splitext(fname)[0] + ".json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(labelme_json, f, ensure_ascii=False, indent=2)
        count += 1

    print(f"✅ 已生成 {count} 个 LabelMe JSON 文件，保存在: {out_dir}")


def main():
    meta_list, labels = load_data(INST_META_PKL, LABELS_NPY)
    save_labelme_json(meta_list, labels, IMAGE_DIR, OUTPUT_DIR)


if __name__ == "__main__":
    main()