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
PCA_2D     = r"E:\Project_VAE\V2\Clusters\reassembled_pca_2d.npy"
LABELS_NPY = r"E:\Project_VAE\V2\Clusters\reassembled_cluster_labels.npy"
IMAGES_PKL = r"E:\Project_VAE\V2\Clusters\inst_images.pkl"
META_PKL   = r"E:\Project_VAE\V2\Clusters\inst_meta.pkl"
PAD_TIF    = r"E:\Project_VAE\V2\padded_image\1_2_3_4\1"  # ROI 上下文大图存放目录

def main():
    # 1) 加载数据
    pca2d   = np.load(PCA_2D)           # (N,2)
    labels  = np.load(LABELS_NPY).astype(int)
    with open(IMAGES_PKL, 'rb') as f:
        rois = pickle.load(f)          # list of small PIL or ndarray
    with open(META_PKL, 'rb') as f:
        metas = pickle.load(f)         # list of dicts with base & padded_bbox

    # 2) 散点图
    fig, ax = plt.subplots(figsize=(8,8))
    sc = ax.scatter(pca2d[:,0], pca2d[:,1],
                    c=labels, cmap="tab10",
                    s=30, alpha=0.6,
                    edgecolor="white", picker=5)
    plt.colorbar(sc, ax=ax, label="Cluster")
    ax.set_title("Cluster PCA (click a point)")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.grid(alpha=0.3)

    highlights = {}

    def on_pick(evt):
        idx = evt.ind[0]
        if idx in highlights: 
            return
        # 高亮散点
        x,y = pca2d[idx]
        hl = ax.scatter(x,y,
                        s=200, facecolor='none',
                        edgecolor='black', linewidth=2,
                        zorder=10)
        highlights[idx] = hl
        fig.canvas.draw_idle()

        # 弹出 ROI
        roi = rois[idx]
        fig_roi, ax_roi = plt.subplots(figsize=(3,3))
        ax_roi.imshow(roi, cmap='gray')
        ax_roi.set_title(f"ROI #{idx}  Cluster {labels[idx]}")
        ax_roi.axis('off')

        # 弹出上下文
        meta = metas[idx]
        base = meta['base']
        r0,c0,r1,c1 = meta['padded_bbox']
        tif = meta.get('source_tif') or os.path.join(PAD_TIF, f"{base}.tif")
        full = Image.open(tif)
        fig_full, ax_full = plt.subplots(figsize=(6,6))
        ax_full.imshow(full, cmap='gray')
        rect = Rectangle((c0,r0), c1-c0, r1-r0,
                         linewidth=2, edgecolor='red', facecolor='none')
        ax_full.add_patch(rect)
        ax_full.set_title(os.path.basename(tif))
        ax_full.axis('off')

        def on_close(evt):
            # 关闭子窗口时清除高亮
            highlights[idx].remove()
            del highlights[idx]
            fig.canvas.draw_idle()
            plt.close(fig_roi)
            plt.close(fig_full)

        fig_roi.canvas.mpl_connect("close_event", on_close)
        plt.show()

    fig.canvas.mpl_connect("pick_event", on_pick)
    plt.show()

if __name__=='__main__':
    main()
