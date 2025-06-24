import os
import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import torch
from c5_CVAE_model import CachedImageDataset


def interactive_visualization(method, coord_file, flag_file, cache_file):
    # 1) 读取坐标和标记
    coords = np.load(coord_file)       # [N,2]
    flags = np.load(flag_file)         # PCA: bool array; UMAP: int labels

    # 2) 准备数据集
    cache = torch.load(cache_file, map_location="cpu")
    masks = cache["masks"]            # Tensor[N,1,50,50]
    label_list = (masks.view(len(masks), -1).sum(1) > 0).long().tolist()
    dataset = CachedImageDataset(cache_file=cache_file, label_list=label_list)

    # 3) 绘图
    fig, ax = plt.subplots(figsize=(10, 8))
    if method == 'pca':
        colors = np.where(flags, 'red', 'gray')
        title = 'CVAE PCA Latent Space (click to view image)'
        legend_elems = [('Outlier', 'red'), ('Inlier', 'gray')]
    else:
        unique = np.unique(flags)
        cmap = plt.get_cmap('tab20', len(unique))
        colors = [cmap(i) if lab != -1 else 'black' for i, lab in enumerate(flags)]
        title = 'CVAE UMAP Latent Space (click to view image)'
        legend_elems = [(f'Cluster {lab}', cmap(i)) for i, lab in enumerate(unique) if lab != -1] + [('Noise', 'black')]

    pts = ax.scatter(coords[:,0], coords[:,1], s=30, c=colors,
                     edgecolor='white', linewidth=0.5, alpha=0.6, picker=5)
    ax.set_title(title)
    ax.set_xlabel('Dim 1')
    ax.set_ylabel('Dim 2')
    ax.grid(alpha=0.3)

    # 添加图例
    handles = [plt.Line2D([0],[0], marker='o', color='w', markerfacecolor=col, markersize=8)
               for _, col in legend_elems]
    labels = [lab for lab, _ in legend_elems]
    ax.legend(handles, labels, loc='best')

    highlights = {}

    def on_pick(event):
        ind = event.ind[0]
        if ind in highlights:
            return
        x, y = coords[ind]
        hl = ax.scatter([x], [y], s=120, facecolor='gold', edgecolor='black', linewidth=1, zorder=10)
        highlights[ind] = hl
        fig.canvas.draw_idle()

        img = dataset.images[ind].squeeze().numpy()
        img_fig, img_ax = plt.subplots(figsize=(4,4))
        img_ax.imshow(img, cmap='gray')
        img_ax.set_title(f'Sample #{ind}')
        img_ax.axis('off')
        plt.show()

        def on_close(evt):
            if ind in highlights:
                highlights[ind].remove()
                del highlights[ind]
                fig.canvas.draw_idle()
        img_fig.canvas.mpl_connect('close_event', on_close)

    fig.canvas.mpl_connect('pick_event', on_pick)
    plt.show()


if __name__ == '__main__':
    cache_path = r"E:\Project_VAE\V2\Pack_json_png\Ori_drop\data_cache.pt"
    base_dir   = r"E:\Project_VAE\V2\visual_latent"

    choice = input("请选择可视化方式 (pca/umap): ").strip().lower()
    if choice == 'pca':
        coord = os.path.join(base_dir, 'pca', 'pca_2d.npy')
        flag  = os.path.join(base_dir, 'pca', 'pca_outliers.npy')
    elif choice == 'umap':
        coord = os.path.join(base_dir, 'umap', 'umap_2d.npy')
        flag  = os.path.join(base_dir, 'umap', 'umap_labels.npy')
    else:
        print("无效选择，请输入 'pca' 或 'umap'！")
        exit(1)

    interactive_visualization(choice, coord, flag, cache_path)