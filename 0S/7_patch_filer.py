#!/usr/bin/env python3
import os
import re

def prune_patches(folder,
                  keep_min_row=1, keep_max_row=6,
                  keep_min_col=1, keep_max_col=6,
                  exts=None,
                  extra_remove=None,
                  dry_run=True):
    """
    删除 folder 下所有不在 [keep_min_row..keep_max_row]×[keep_min_col..keep_max_col] 范围内的 patch 文件，
    并额外删除 extra_remove 列表中的 (row, col) 后缀文件。
    文件名格式假定为：<prefix>_<row>_<col>.<ext>，如 patch_3_5.png
    - exts: 要处理的文件后缀列表，默认 ['.png','.jpg','.jpeg','.tif','.tiff']
    - extra_remove: 要额外删除的 (row, col) 对列表，如 [(5,6),(6,6)]
    - dry_run: True 时只打印将要删除的文件，False 时真正删除
    """
    exts = exts or ['.png', '.jpg', '.jpeg', '.tif', '.tiff']
    extra_remove = set(extra_remove or [])

    # 构造正则：匹配 _<row>_<col>.<ext>
    pat = re.compile(r'_(\d+)_(\d+)\.(' + '|'.join([e.lstrip('.') for e in exts]) + r')$', re.IGNORECASE)

    to_delete = []
    for fname in os.listdir(folder):
        m = pat.search(fname)
        if not m:
            # 文件名不匹配格式，跳过
            continue
        row, col = int(m.group(1)), int(m.group(2))
        # 超出保留范围，或在额外删除列表中
        if not (keep_min_row <= row <= keep_max_row and keep_min_col <= col <= keep_max_col) \
           or (row, col) in extra_remove:
            to_delete.append(fname)

    if not to_delete:
        print("✅ 当前无需删除任何文件。")
        return

    print(f"将删除 {len(to_delete)} 个文件（保留行 {keep_min_row}–{keep_max_row}，列 {keep_min_col}–{keep_max_col}，额外删除 {sorted(extra_remove)}）：")
    for fn in to_delete:
        print("  删除 ->", fn)
        if not dry_run:
            try:
                os.remove(os.path.join(folder, fn))
            except Exception as e:
                print(f"⚠️ 删除失败 {fn}: {e}")

    if dry_run:
        print("\n提示：当前为 dry run 模式，文件尚未被删除。")
        print("如确认无误，请将 dry_run=False 并重试。")


if __name__ == "__main__":
    # —— 在这里设置你的目录和参数 —— #
    patch_folder    = r"E:\Project_SNV\0S\6_patch\19"
    keep_min_row    = 2
    keep_max_row    = 6
    keep_min_col    = 2
    keep_max_col    = 6
    # 指定额外要删除的 (row, col)
    extra_remove    = [(6,6)] # [(5, 6), (6, 6)]
    dry_run         = False   # False 则真正执行删除
    # —— 结束设置 —— #

    prune_patches(
        folder=patch_folder,
        keep_min_row=keep_min_row,
        keep_max_row=keep_max_row,
        keep_min_col=keep_min_col,
        keep_max_col=keep_max_col,
        extra_remove=extra_remove,
        dry_run=dry_run
    )
