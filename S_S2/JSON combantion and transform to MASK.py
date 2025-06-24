import os
import json
import glob
import numpy as np
import cv2

def merge_and_generate_masks(mainbody_dir, defect_dir, output_dir, save_npy=True):
    os.makedirs(output_dir, exist_ok=True)
    mainbody_jsons = sorted(glob.glob(os.path.join(mainbody_dir, "*_mainbody.json")))

    label_map = {
        "main body": 1,
        "defect": 2
    }

    for mb_json in mainbody_jsons:
        base_name = os.path.basename(mb_json).replace("_mainbody.json", "")
        defect_json_path = os.path.join(defect_dir, base_name + "_defect.json")
        merged_json_path = os.path.join(output_dir, base_name + "_merged.json")
        npy_path = os.path.join(output_dir, base_name + ".npy")

        # 读取 main body JSON
        with open(mb_json, 'r') as f:
            mb_data = json.load(f)

        # 读取 defect JSON（如果存在）
        if os.path.exists(defect_json_path):
            with open(defect_json_path, 'r') as f:
                defect_data = json.load(f)
            defect_shapes = defect_data.get("shapes", [])
        else:
            defect_shapes = []

        # ----- 新增：过滤掉不“完全在 MB 内且不接触边界”的 defect -----
        # 1. 把 main body 的多边形都取出来
        mb_polygons = [
            np.array(s["points"], dtype=np.int32)
            for s in mb_data["shapes"]
            if s["label"].lower() == "main body"
        ]

        filtered_defects = []
        for shape in defect_shapes:
            pts = np.array(shape["points"], dtype=np.float32)
            shape_type = shape.get("shape_type", "").lower()

            # --- 如果是圆形（LabelMe 的 circle，points = [[cx, cy], [cx+r, cy]]） ---
            if shape_type == "circle" or len(pts) == 2:
                # 提取圆心与半径
                cx, cy = pts[0]
                # 假设第二个点是 (cx+r, cy)，所以半径约等于两点 x 差值
                r = abs(pts[1][0] - cx)

                # 检查在所有 MB 多边形中，是否有一个能让圆“完全在内部且不碰边界”
                fully_inside = False
                for poly in mb_polygons:
                    # 使用 pointPolygonTest(..., True) 获取到中心点到 poly 边界的**最短有符号距离**
                    d = cv2.pointPolygonTest(poly, (cx, cy), True)
                    # 如果 d > r，则圆心距离边界足够远，圆不会接触边界
                    if d > r:
                        fully_inside = True
                        break

                if fully_inside:
                    filtered_defects.append(shape)

            # --- 否则当做一般多边形来处理 ---
            else:
                # 先判断所有顶点是否都严格在该 MB 多边形内部（pointPolygonTest > 0）
                # 只要某一个顶点不满足 “>0” 则说明不在严格内部，直接丢弃
                all_vertices_strictly_inside = False

                for poly in mb_polygons:
                    inside_all = True
                    for (x, y) in pts:
                        if cv2.pointPolygonTest(poly, (x, y), False) <= 0:
                            # 如果 ≤ 0：要么在边界(=0)，要么在外部(<0)，都视为“不满足严格内部”
                            inside_all = False
                            break
                    if inside_all:
                        # 如果遇到某个 MB 多边形，对当前 defect 所有顶点都能满足“>0”，
                        # 说明这个 defect 多边形整体都在这个 poly 内部（且不与边界相交）
                        all_vertices_strictly_inside = True
                        break

                if all_vertices_strictly_inside:
                    filtered_defects.append(shape)

        # -------------------------------------------

        # 合并 shapes：只把“严格在 MB 内且不接触边界”的 defect 加进去
        merged_shapes = mb_data["shapes"] + filtered_defects
        mb_data["shapes"] = merged_shapes

        # 保存 merged JSON
        with open(merged_json_path, 'w') as f:
            json.dump(mb_data, f, indent=4)
        print(f"[✅] Merged JSON saved: {merged_json_path}")

        # 如果需要也生成 npy mask
        if save_npy:
            h, w = mb_data["imageHeight"], mb_data["imageWidth"]
            mask = np.zeros((h, w), dtype=np.uint8)
            for shape in merged_shapes:
                label = shape["label"]
                pts = np.array(shape["points"], dtype=np.int32)
                value = label_map.get(label, 0)
                cv2.fillPoly(mask, [pts], value)
            np.save(npy_path, mask)
            print(f"[📁] NPY mask saved: {npy_path}")


merge_and_generate_masks(
    mainbody_dir=r"D:\Study\Postgraduate\S2\Project\Code\Resource\Middle Stage B\1_2_3_4\1",
    defect_dir=r"D:\Study\Postgraduate\S2\Project\Code\Resource\Middle Stage D\1_2_3_4\1",
    output_dir=r"D:\Study\Postgraduate\S2\Project\Code\Resource\Middle Stage\1_2_3_4\1\First"
)
