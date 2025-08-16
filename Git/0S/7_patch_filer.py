#!/usr/bin/env python3
import os
import re

def prune_patches(
    folder,
    keep_min_row=1, keep_max_row=6,
    keep_min_col=1, keep_max_col=6,
    exts=None,
    extra_remove=None,
    dry_run=True
):
    """
    Delete patch files in `folder` whose row/column indices fall outside the specified
    range [keep_min_row..keep_max_row]×[keep_min_col..keep_max_col],
    as well as any additional (row, col) pairs in `extra_remove`.

    Assumes filenames have the format: <prefix>_<row>_<col>.<ext>, e.g. patch_3_5.png.

    Parameters:
      - exts: list of file extensions to consider (default: ['.png', '.jpg', '.jpeg', '.tif', '.tiff']).
      - extra_remove: list of (row, col) tuples to always remove.
      - dry_run: if True, only print files that would be deleted; if False, delete them.
    """
    # Default extensions and extra removal set
    exts = exts or ['.png', '.jpg', '.jpeg', '.tif', '.tiff']
    extra_remove = set(extra_remove or [])

    # Compile regex to match suffix _<row>_<col>.<ext>
    pat = re.compile(r'_(\d+)_(\d+)\.(' + '|'.join([e.lstrip('.') for e in exts]) + r')$', re.IGNORECASE)

    to_delete = []
    for fname in os.listdir(folder):
        m = pat.search(fname)
        if not m:
            # Skip filenames that don't match the expected pattern
            continue
        row, col = int(m.group(1)), int(m.group(2))
        # Mark for deletion if outside keep range or in extra_remove
        if not (keep_min_row <= row <= keep_max_row and keep_min_col <= col <= keep_max_col) \
           or (row, col) in extra_remove:
            to_delete.append(fname)

    if not to_delete:
        print("✅ No files need to be deleted.")
        return

    print(f"Deleting {len(to_delete)} files (keeping rows {keep_min_row}–{keep_max_row}, "
          f"cols {keep_min_col}–{keep_max_col}, extra remove {sorted(extra_remove)}):")
    for fn in to_delete:
        print("  Removing ->", fn)
        if not dry_run:
            try:
                os.remove(os.path.join(folder, fn))
            except Exception as e:
                print(f"⚠️ Failed to remove {fn}: {e}")

    if dry_run:
        print("\nNote: dry_run=True, so no files were actually deleted.")
        print("Set dry_run=False to perform actual deletion.")


if __name__ == "__main__":
    # —— Configure your folder and parameters here —— #
    patch_folder = r"E:\Project_SNV\0S\6_patch\19"
    keep_min_row = 2
    keep_max_row = 5
    keep_min_col = 2
    keep_max_col = 6
    extra_remove = []  # Additional (row, col) pairs to delete
    dry_run = False          # Set to False to actually delete files
    # —— End configuration —— #

    prune_patches(
        folder=patch_folder,
        keep_min_row=keep_min_row,
        keep_max_row=keep_max_row,
        keep_min_col=keep_min_col,
        keep_max_col=keep_max_col,
        extra_remove=extra_remove,
        dry_run=dry_run
    )
