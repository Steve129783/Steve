# interactive_cluster_viz_with_context.py

import os
import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import pickle
from matplotlib.patches import Rectangle
from PIL import Image    # 确保可以使用 Image.open

# 抽样参数
MAX_PER_CLUSTER = 1000
# 原始 TIFF 所在目录（根据你的环境调整）
PADDED_TIF_DIR = r"E:\Project_VAE\V1\padded_image\1_2_3_4\1"


def interactive_cluster_viz_with_context(
    pca_coord_file: str,
    labels_file:    str,
    images_pkl:     str,
    meta_pkl:       str
):
    """
    抽样 + 点击后在原图上画框高亮＋显示 ROI。
    """
    # 1) 加载 PCA 坐标、聚类标签、ROI 图像和对应 meta
    X2         = np.load(pca_coord_file)           # (M,2)
    labels     = np.load(labels_file).astype(int)  # (M,)
    with open(images_pkl, "rb") as f:
        inst_images = pickle.load(f)               # list of PIL.Image
    with open(meta_pkl, "rb") as f:
        inst_meta   = pickle.load(f)               # list of dict with padded_bbox and base

    M = len(labels)
    # 2) 每簇抽样
    display_mask = np.zeros(M, dtype=bool)
    rng = np.random.default_rng(0)
    for c in np.unique(labels):
        idxs = np.where(labels == c)[0]
        chosen = rng.choice(idxs, min(len(idxs), MAX_PER_CLUSTER), replace=False)
        display_mask[chosen] = True

    # 3) 绘制散点图（只展示抽样后的点）
    fig, ax = plt.subplots(figsize=(8,8))
    sc = ax.scatter(
        X2[display_mask,0], X2[display_mask,1],
        c=labels[display_mask], cmap="tab10",
        s=20, alpha=0.6,
        edgecolor="white", linewidth=0.5,
        picker=5
    )
    plt.colorbar(sc, ax=ax, pad=0.02, label="Cluster")
    ax.set_title(f"PCA (≤{MAX_PER_CLUSTER}/cluster sampled)")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.grid(alpha=0.3)

    highlights = {}

    def on_pick(event):
        disp_idxs = np.where(display_mask)[0]
        disp_idx  = event.ind[0]
        idx       = disp_idxs[disp_idx]
        if idx in highlights:
            return

        # 高亮主图点
        x, y = X2[idx]
        hl = ax.scatter(x, y, s=120, facecolor="none",
                        edgecolor="black", linewidth=1.5, zorder=10)
        highlights[idx] = hl
        fig.canvas.draw_idle()

        # 弹出 ROI 小窗
        roi = inst_images[idx]
        fig2, ax2 = plt.subplots(figsize=(3,3))
        ax2.imshow(roi, cmap="gray")
        ax2.set_title(f"Idx:{idx} Cluster:{labels[idx]}", fontsize=9)
        ax2.axis("off")
        plt.tight_layout()

        # 弹出上下文大图并框选
        meta = inst_meta[idx]
        base = meta.get('base')
        # 如果 meta 中已有 source_tif 字段优先使用
        tif = meta.get('source_tif') or os.path.join(PADDED_TIF_DIR, f"{base}.tif")
        (r0, c0, r1, c1) = meta['padded_bbox']
        img_full = Image.open(tif)
        fig3, ax3 = plt.subplots(figsize=(6,6))
        ax3.imshow(img_full, cmap="gray")
        # 画个红色框
        rect = Rectangle((c0, r0), c1 - c0, r1 - r0,
                         linewidth=2, edgecolor='red', facecolor='none')
        ax3.add_patch(rect)
        ax3.set_title(f"{os.path.basename(tif)}\nBBox:({r0},{c0})–({r1},{c1})")
        ax3.axis("off")
        plt.tight_layout()

        def on_close(evt):
            # 关闭时清除高亮并关闭子窗口
            highlights[idx].remove()
            del highlights[idx]
            fig.canvas.draw_idle()
            plt.close(fig2)
            plt.close(fig3)

        fig2.canvas.mpl_connect("close_event", on_close)
        plt.show()

    fig.canvas.mpl_connect("pick_event", on_pick)
    plt.show()


if __name__ == "__main__":
    PCA_NPY    = r"E:\Project_VAE\V1\Clusters\reassembled_pca_2d.npy"
    LAB_NPY    = r"E:\Project_VAE\V1\Clusters\reassembled_cluster_labels.npy"
    IMAGES_PKL = r"E:\Project_VAE\V1\Clusters\inst_images.pkl"
    META_PKL   = r"E:\Project_VAE\V1\Clusters\inst_meta.pkl"

    interactive_cluster_viz_with_context(
        pca_coord_file=PCA_NPY,
        labels_file=LAB_NPY,
        images_pkl=IMAGES_PKL,
        meta_pkl=META_PKL
    )