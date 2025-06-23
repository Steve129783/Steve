import os
import json
import torch
import random
from PIL import Image, ImageDraw
from torchvision import transforms

# ———— 配置 ————
images_folder     = r"E:\Project_VAE\V2\Sliced_png\1_2_3_4\1"
patch_json_folder = r"E:\Project_VAE\V2\Sliced_json\1_2_3_4\1"
save_path         = r"E:\Project_VAE\V2\Pack_json_png\data_cache.pt"
patch_size        = 50
# ——————————————————

to_tensor = transforms.ToTensor()

# —— 交互询问 1: 是否开启增强 ——#
choice_aug = input("是否对 patch 进行数据增强？(y=Aug / n=Ori): ").strip().lower()
do_aug = (choice_aug == 'y')

# —— 交互询问 2: 是否丢弃无 JSON 的 patch ——#
choice_drop = input("是否丢弃没有对应 JSON 的 PNG？(y=drop / n=save): ").strip().lower()
drop_unmatched = (choice_drop == 'y')

images, masks, metas = [], [], []

for fn in sorted(os.listdir(images_folder)):
    if not fn.lower().endswith(".png"):
        continue

    base = os.path.splitext(fn)[0]
    parts = base.rsplit("_", 2)
    if len(parts) != 3:
        continue
    prefix, row_s, col_s = parts
    try:
        row, col = int(row_s), int(col_s)
    except ValueError:
        continue

    js_path = os.path.join(patch_json_folder, f"{base}.json")
    has_json = os.path.exists(js_path)
    if not has_json and drop_unmatched:
        continue

    # 1) 读图和 mask
    img_orig = Image.open(os.path.join(images_folder, fn)).convert("L")
    mask_orig = Image.new("L", img_orig.size, 0)
    iid = -1
    if has_json:
        with open(js_path, 'r', encoding='utf-8') as f:
            js = json.load(f)
        draw = ImageDraw.Draw(mask_orig)
        for shape in js.get("shapes", []):
            pts = [tuple(pt) for pt in shape.get("points", [])]
            if len(pts) >= 3:
                draw.polygon(pts, outline=255, fill=255)
        iid = js["shapes"][0].get("instance_id", -1)

    # 封装一个函数，把 (PIL img, mask, meta) 推入列表
    def append_example(img_pil, mask_pil, meta_suffix=""):
        images.append(to_tensor(img_pil))
        masks.append(to_tensor(mask_pil))
        metas.append({
            "base":        prefix + meta_suffix,
            "patch_row":   row,
            "patch_col":   col,
            "instance_id": iid
        })

    # —— 总是先保存原版 ——#
    append_example(img_orig, mask_orig, meta_suffix="")

    # —— 若开启增强，则额外生成一次随机翻转并保存 ——#
    if do_aug:
        img_aug, mask_aug = img_orig, mask_orig
        if random.random() < 0.5:
            img_aug = img_aug.transpose(Image.FLIP_LEFT_RIGHT)
            mask_aug = mask_aug.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            img_aug = img_aug.transpose(Image.FLIP_TOP_BOTTOM)
            mask_aug = mask_aug.transpose(Image.FLIP_TOP_BOTTOM)
        # 如果需要更多增强策略，在这里继续添加…
        append_example(img_aug, mask_aug, meta_suffix="_aug")

# —— 自检并保存 ——#
num_packed = len(images)
print(f"最终打包 {num_packed} 个 patch，增强={'开启' if do_aug else '关闭'}，"
      f"{'丢弃' if drop_unmatched else '保留'} 无 JSON 的 patch。")

cache = {
    "images": torch.stack(images, 0),  # (N,1,50,50)
    "masks":  torch.stack(masks, 0),   # (N,1,50,50)
    "meta":   metas                    # List[dict]
}
os.makedirs(os.path.dirname(save_path), exist_ok=True)
torch.save(cache, save_path)
print(f"Saved data cache to {save_path}")
