#!/usr/bin/env python3
# view_clusters_interactive.py

import os
import numpy as np
import pickle
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

# —— 配置 ——
BASE_DIR   = r"E:\Project_VAE\V2\Clusters"
PAD_TIF    = r"E:\Project_VAE\V2\padded_image\1_2_3_4\1"
CACHE_META = os.path.join(BASE_DIR, 'inst_meta.pkl')
CACHE_IMGS = os.path.join(BASE_DIR, 'inst_images.pkl')

# 文件映射
FILES = {
    'pca': {
        'coords': os.path.join(BASE_DIR, 'reassembled_pca_2d.npy'),
        'labels': os.path.join(BASE_DIR, 'reassembled_cluster_labels.npy'),
        'title' : 'PCA 2D Cluster'
    },
    'umap': {
        'coords': os.path.join(BASE_DIR, 'reassembled_umap_2d.npy'),
        'labels': os.path.join(BASE_DIR, 'reassembled_cluster_labels.npy'),
        'title' : 'UMAP 2D Cluster'
    }
}


def main():
    choice = input("请选择可视化方式 (pca/umap): ").strip().lower()
    if choice not in FILES:
        print("无效选择，请输入 'pca' 或 'umap'.")
        return

    coord_path = FILES[choice]['coords']
    label_path = FILES[choice]['labels']
    title_main = FILES[choice]['title']

    # 1) 加载坐标、标签与实例数据
    coords = np.load(coord_path)             # (N,2)
    labels = np.load(label_path).astype(int) # (N,)
    with open(CACHE_IMGS, 'rb') as f:
        rois = pickle.load(f)
    with open(CACHE_META, 'rb') as f:
        metas = pickle.load(f)

    # 2) 绘制散点图
    fig, ax = plt.subplots(figsize=(8, 8))
    sc = ax.scatter(
        coords[:, 0], coords[:, 1],
        c=labels, cmap='tab10',
        s=30, alpha=0.6,
        edgecolor='white', picker=5
    )
    plt.colorbar(sc, ax=ax, label="Cluster")
    ax.set_title(f"{title_main} (click point to view)")
    ax.set_xlabel('Dim 1')
    ax.set_ylabel('Dim 2')
    ax.grid(alpha=0.3)

    highlights = {}

    def on_pick(evt):
        idx = evt.ind[0]
        if idx in highlights:
            return
        # 高亮点
        x, y = coords[idx]
        hl = ax.scatter(x, y,
                        s=200, facecolor='none',
                        edgecolor='black', linewidth=2,
                        zorder=10)
        highlights[idx] = hl
        fig.canvas.draw_idle()

        # 显示 ROI
        roi = rois[idx]
        fig_roi, ax_roi = plt.subplots(figsize=(3,3))
        ax_roi.imshow(roi, cmap='gray')
        ax_roi.set_title(f"ROI #{idx} Cluster {labels[idx]}")
        ax_roi.axis('off')

        # 显示上下文
        meta = metas[idx]
        base = meta['base']
        r0, c0, r1, c1 = meta['padded_bbox']
        tif_path = meta.get('source_tif') or os.path.join(PAD_TIF, f"{base}.tif")
        full = Image.open(tif_path)
        fig_full, ax_full = plt.subplots(figsize=(6,6))
        ax_full.imshow(full, cmap='gray')
        rect = Rectangle((c0, r0), c1-c0, r1-r0,
                         linewidth=2, edgecolor='red', facecolor='none')
        ax_full.add_patch(rect)
        ax_full.set_title(os.path.basename(tif_path))
        ax_full.axis('off')

        def on_close(evt):
            # 关闭时移除高亮
            highlights[idx].remove()
            del highlights[idx]
            fig.canvas.draw_idle()
            plt.close(fig_roi)
            plt.close(fig_full)

        fig_roi.canvas.mpl_connect('close_event', on_close)
        plt.show()

    fig.canvas.mpl_connect('pick_event', on_pick)
    plt.show()


if __name__ == '__main__':
    main()
