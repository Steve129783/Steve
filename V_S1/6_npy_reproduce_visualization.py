import os
import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from o3_VAE_model import CachedImageDataset

def interactive_pca_visualization(pca_coord_file, outlier_file, cache_file):
    """基础版多点高亮交互可视化"""
    # 1) 加载数据
    pca_coords = np.load(pca_coord_file)
    outlier_idx = np.load(outlier_file)
    dataset = CachedImageDataset(cache_file=cache_file)

    # 2) 创建主图
    fig, ax = plt.subplots(figsize=(10, 8))
    original_colors = np.where(outlier_idx, 'red', 'gray')
    ax.scatter(
        pca_coords[:, 0], pca_coords[:, 1],
        s=30, c=original_colors, alpha=0.6,
        edgecolor='white', linewidth=0.5, picker=5
    )
    ax.set_title("PCA Projection (Click points)")
    ax.grid(True, alpha=0.3)

    # 3) 交互功能
    highlights = {}  # 存储高亮点 {index: highlight_object}

    def on_pick(event):
        ind = event.ind[0]
        x, y = pca_coords[ind]
        
        # 如果已高亮则跳过
        if ind in highlights:
            return
            
        # 创建高亮点
        hl = ax.scatter(
            x, y, s=120, facecolor='gold',
            edgecolor='black', linewidth=1, zorder=10
        )
        highlights[ind] = hl
        fig.canvas.draw_idle()
        
        # 创建图像窗口
        img_fig = plt.figure(figsize=(5, 5))
        plt.imshow(dataset.images[ind].squeeze(), cmap='gray')
        plt.title(f"Index: {ind}")
        plt.axis('off')
        
        # 关闭窗口时移除对应高亮
        def on_close(_):
            if ind in highlights:
                highlights[ind].remove()
                del highlights[ind]
                fig.canvas.draw_idle()
            plt.close(img_fig)
            
        img_fig.canvas.mpl_connect('close_event', on_close)
        plt.show()

    fig.canvas.mpl_connect('pick_event', on_pick)
    plt.show()

if __name__ == "__main__":
    cache_path = r"E:\Project_VAE\V1\Ori_Pack\1_2_3_4\data_cache.pt"
    viz_dir = r"E:\Project_VAE\V1\Visualization\BGMBDF"
    
    interactive_pca_visualization(
        pca_coord_file=os.path.join(viz_dir, "pca_2d.npy"),
        outlier_file=os.path.join(viz_dir, "outliers.npy"),
        cache_file=cache_path
    )
