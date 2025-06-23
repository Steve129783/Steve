# interactive_pca_cvae.py

import os
import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt

import torch
from c5_CVAE_model import CachedImageDataset

def interactive_pca_visualization(pca_coord_file, outlier_file, cache_file):
    # 1) 读 PCA 坐标 & 异常标记
    pca_coords = np.load(pca_coord_file)      # [N,2]
    outlier_idx = np.load(outlier_file)       # [N,], bool

    # 2) 载入 cache 生成 label_list，再实例化 Dataset
    cache = torch.load(cache_file, map_location="cpu")
    masks = cache["masks"]                    # Tensor[N,1,50,50]
    label_list = (masks.view(len(masks), -1).sum(1) > 0) \
                    .long().tolist()         # [0,1,...]

    dataset = CachedImageDataset(cache_file=cache_file,
                                 label_list=label_list)

    # 3) 绘制 PCA 散点图
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = np.where(outlier_idx, "red", "gray")
    pts = ax.scatter(
        pca_coords[:, 0], pca_coords[:, 1],
        s=30, c=colors, alpha=0.6,
        edgecolor="white", linewidth=0.5,
        picker=5  # 允许点击拾取
    )
    ax.set_title("CVAE Latent Space PCA (click to view image)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(alpha=0.3)

    # 用来存放当前高亮的点
    highlights = {}

    def on_pick(event):
        ind = event.ind[0]                   # 拾取到的第一个点索引
        if ind in highlights:
            return  # 已高亮过就不再重复

        # 在主图上高亮
        x, y = pca_coords[ind]
        hl = ax.scatter(
            [x], [y],
            s=120, facecolor="gold",
            edgecolor="black", linewidth=1, zorder=10
        )
        highlights[ind] = hl
        fig.canvas.draw_idle()

        # 弹出一个新窗口显示原图
        img = dataset.images[ind].squeeze().numpy()
        img_fig, img_ax = plt.subplots(figsize=(4, 4))
        img_ax.imshow(img, cmap="gray")
        img_ax.set_title(f"Sample #{ind}")
        img_ax.axis("off")
        plt.show()

        # 当新图窗口关闭时，清除高亮
        def on_close(evt):
            if ind in highlights:
                highlights[ind].remove()
                del highlights[ind]
                fig.canvas.draw_idle()
        img_fig.canvas.mpl_connect("close_event", on_close)

    # 绑定拾取事件
    fig.canvas.mpl_connect("pick_event", on_pick)
    plt.show()


if __name__ == "__main__":
    cache_path = r"E:\Project_VAE\V2\Pack_json_png\Ori_drop\data_cache.pt"
    viz_dir    = r"E:\Project_VAE\V2\visual_latent"

    interactive_pca_visualization(
        pca_coord_file = os.path.join(viz_dir, "pca_2d.npy"),
        outlier_file   = os.path.join(viz_dir, "outliers.npy"),
        cache_file     = cache_path
    )
