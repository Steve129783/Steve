import torch
from pathlib import Path

CACHE_FILE = Path(r"E:\Project_VAE\V2\Pack_json_png\Ori_drop\data_cache.pt")
cache = torch.load(CACHE_FILE, map_location="cpu")

# 打印 images 前 5 条
print("=== images 前 5 条 ===")
images = cache["images"]
for i in range(5):
    img = images[i]
    print(f"{i}: shape={tuple(img.shape)}, dtype={img.dtype}, 前 5 像素值={img.view(-1)[:5].tolist()}")

# 打印 masks 前 5 条
print("\n=== masks 前 5 条 ===")
masks = cache["masks"]
for i in range(5):
    m = masks[i]
    print(f"{i}: shape={tuple(m.shape)}, dtype={m.dtype}, 前 5 像素值={m.view(-1)[:5].tolist()}")

# 打印 meta 前 5 条
print("\n=== meta 前 5 条 ===")
meta = cache["meta"]
for i in range(5):
    print(f"{i}: {meta[i]}")
