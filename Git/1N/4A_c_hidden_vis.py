#!/usr/bin/env python3
import os
import re
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torchvision import transforms
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from PIL import Image
from c2_CNN_model import CachedImageDataset, CNN

# =========================
# 1. Configuration
# =========================
file_name     = '1_2_3_4'
cache_file    = rf'E:\Project_SNV\1N\1_Pack\{file_name}\data_cache.pt'
model_path    = rf'E:\Project_SNV\1N\c2_cache\{file_name}\best_model.pth'
splits        = (0.7, 0.15, 0.15)
seed          = 42
batch_size    = 32
num_workers   = 0   # ensure determinism
in_shape      = (50, 50)
device        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
max_per_group = 1500  # maximum number of samples to show per group

# Read group-id → name mapping
def load_group_names(cache_path):
    info_path = os.path.join(os.path.dirname(cache_path), 'info.txt')
    pattern   = re.compile(r'^\s*(\d+)\s*:\s*([^\s(]+)')
    mapping   = {}
    with open(info_path, 'r', encoding='utf-8') as f:
        for line in f:
            m = pattern.match(line)
            if m:
                mapping[int(m.group(1))] = m.group(2)
    return mapping

group_names = load_group_names(cache_file)

# =========================
# 2. Main routine
# =========================
def extract_and_plot_all():
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Dataset loading and splitting
    ds = CachedImageDataset(cache_file, transform=None, return_path=True)
    N  = len(ds)
    n1 = int(splits[0] * N)
    n2 = int((splits[0] + splits[1]) * N)
    train_ds, val_ds, test_ds = random_split(
        ds, [n1, n2-n1, N-n2],
        generator=torch.Generator().manual_seed(seed)
    )

    # Load model
    state = torch.load(model_path, map_location=device)
    linear_keys = [k for k,v in state.items() if k.endswith('weight') and v.dim()==2]
    last_key = sorted(linear_keys)[-1]
    n_classes = state[last_key].shape[0]
    model = CNN(in_shape, n_classes).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()

    # Merge all subsets
    combined = ConcatDataset([train_ds, val_ds, test_ds])

    # Extract hidden features
    loader = DataLoader(combined, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    all_h, all_g, all_paths = [], [], []
    with torch.no_grad():
        for x,y,p in loader:
            _, h = model(x.to(device))
            all_h.append(h.cpu().numpy())
            all_g.extend(y.numpy())
            all_paths.extend(p)
    H = np.vstack(all_h)
    G = np.array(all_g)

    # PCA: compute 5 components, print explained variance ratios of the first 5,
    # then keep the first 2 components for visualization
    pca_full = PCA(n_components=5, random_state=seed).fit(H)
    ratios = pca_full.explained_variance_ratio_
    print("Explained variance ratio of first 5 PCs:", [round(r,4) for r in ratios])
    H_pca = pca_full.transform(H)[:, :2]

    # Downsample if a group has more than max_per_group samples
    idxs = []
    for grp in np.unique(G):
        grp_idx = np.where(G==grp)[0]
        if len(grp_idx) > max_per_group:
            grp_idx = np.random.RandomState(seed).choice(grp_idx, max_per_group, replace=False)
        idxs.extend(grp_idx)
    idxs = np.array(idxs)

    # Scatter plot
    fig, ax = plt.subplots(figsize=(8,7))
    sc = ax.scatter(H_pca[idxs,0], H_pca[idxs,1],
                    c=G[idxs], cmap='tab10',
                    s=30, picker=True, alpha=0.8)
    ax.set_title("PCA of Hidden Features")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    cbar = fig.colorbar(sc, ax=ax, label="Group")
    ticks = sorted(group_names.keys())
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([group_names[k] for k in ticks])

    highlights = {}
    def on_pick(event):
        ind0 = event.ind[0]
        ind  = idxs[ind0]
        x0,y0 = H_pca[ind]
        hl, = ax.plot(x0, y0,
                      marker='o', markersize=12,
                      markerfacecolor='none',
                      markeredgecolor='red',
                      linewidth=2)
        fig.canvas.draw_idle()

        # Popup: original image vs processed cached version
        img_tensor, gid, path = combined[ind]
        proc_img = img_tensor.squeeze().numpy()
        orig_img = np.array(Image.open(path))

        pf, (ax1,ax2) = plt.subplots(1,2, figsize=(8,4))
        ax1.imshow(orig_img, cmap='gray',
           vmin=0, vmax=np.iinfo(orig_img.dtype).max)
        ax1.set_title('Original (u16)')
        ax1.axis('off')
        ax2.imshow(proc_img, cmap='gray',vmin=0,vmax=1)
        ax2.set_title('Processed')
        ax2.axis('off')
        plt.tight_layout()
        plt.show()

        def on_close(evt):
            art = highlights.pop(pf, None)
            if art:
                art.remove()
                fig.canvas.draw_idle()
        highlights[pf] = hl
        pf.canvas.mpl_connect('close_event', on_close)

    fig.canvas.mpl_connect('pick_event', on_pick)
    plt.tight_layout()
    plt.show()

if __name__=='__main__':
    extract_and_plot_all()
