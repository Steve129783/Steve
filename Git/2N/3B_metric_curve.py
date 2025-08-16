#!/usr/bin/env python3
import os
import re
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from c1_frozen import CachedImageDataset, CNN

# =========================
# 1. Configuration
# =========================
file_name      = 'h_c_l'
cha            = 11
cache_file     = rf'E:\Project_SNV\1N\1_Pack\{file_name}\data_cache.pt'
model_path     = rf'E:\Project_SNV\2N\c2_cache\{file_name}\{cha}\best_model_{cha}.pth'
splits         = (0.7, 0.15, 0.15)
seed           = 42
batch_size     = 32
num_workers    = 0
in_shape       = (50, 50)
device         = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
max_per_group  = 1500    # maximum number of samples to show per group
keep_channels  = [cha]   # channel index to keep in the frozen model

# —— Load gid → name mapping —— #
def load_group_names(cache_path):
    info = {}
    p = re.compile(r'^\s*(\d+)\s*:\s*([^\s(]+)')
    path = os.path.join(os.path.dirname(cache_path), 'info.txt')
    with open(path, encoding='utf-8') as f:
        for line in f:
            m = p.match(line)
            if m:
                info[int(m.group(1))] = m.group(2)
    return info

group_names = load_group_names(cache_file)
n_classes   = len(group_names)

# =========================
# 2. Main pipeline: Extract and plot PC1 vs Contrast
# =========================
def extract_and_plot_contrast():
    # Fix random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # 1) Load dataset and model
    ds = CachedImageDataset(cache_file)  # returns (x, y)
    model = CNN(in_shape=in_shape,
                n_classes=n_classes,
                keep_channels=keep_channels) 
    # Load checkpoint and filter out mismatched weights
    ckpt = torch.load(model_path, map_location=device)
    own_state = model.state_dict()
    filtered = {k: v for k, v in ckpt.items() if k in own_state and v.size() == own_state[k].size()}
    own_state.update(filtered)
    model.load_state_dict(own_state)
    model.to(device).eval()

    # 2) Prepare DataLoader
    loader = DataLoader(ds, batch_size=batch_size,
                        shuffle=False, num_workers=num_workers)

    # 3) Register hook on penultimate layer (bn2) to extract hidden features
    feats = []
    def hook_fn(module, inp, out):
        feats.append(out.detach().cpu().numpy())
    handle = model.bn2.register_forward_hook(hook_fn)

    all_contrast = []
    all_labels   = []

    # 4) Forward pass to trigger hook and compute contrast
    with torch.no_grad():
        for x, y in loader:
            x_dev = x.to(device)
            _ = model(x_dev)  # trigger hook

            # Contrast = (mean-min) / (max-min)
            arr = (x.cpu().numpy() * 255).astype(np.uint8)  # [B,1,H,W]
            for im in arr:
                gray = im.squeeze()
                mn, mx = gray.min(), gray.max()
                contrast = ((gray.mean() - mn) / (mx - mn)
                            if mx != mn else 0.0)
                all_contrast.append(contrast)

            all_labels.extend(y.numpy())

    handle.remove()

    # 5) Collect features and labels
    H        = np.vstack(feats)              # [N, hidden_dim]
    contrast = np.array(all_contrast)        # [N]
    G        = np.array(all_labels)          # [N]

    # 6) PCA → extract PC1
    pcs = PCA(n_components=2, random_state=seed).fit_transform(H)
    pc1 = pcs[:,0]

    # 7) Limit sample size to max_per_group per group
    idxs = []
    for g in np.unique(G):
        gi = np.where(G == g)[0]
        if len(gi) > max_per_group:
            gi = np.random.RandomState(seed).choice(
                gi, max_per_group, replace=False)
        idxs.extend(gi.tolist())
    idxs = np.array(idxs)

    # 8) Compute Pearson correlation coefficient
    r, _ = pearsonr(pc1[idxs], contrast[idxs])
    print(f"Pearson r (PC1 vs Contrast): {r:.3f}")

    # 9) Plot PC1 vs Contrast
    fig, ax = plt.subplots(figsize=(7,7))
    sc = ax.scatter(
        pc1[idxs], contrast[idxs],
        c=G[idxs], cmap='tab10',
        s=25, alpha=0.7, picker=5
    )
    ax.set_title("PC1 vs NMB")
    ax.set_xlabel("PC1")
    ax.set_ylabel("NMB")

    cbar = fig.colorbar(sc, ax=ax, label="Group")
    cbar.set_ticks(sorted(group_names.keys()))
    cbar.set_ticklabels([group_names[int(t)] for t in cbar.get_ticks()])

    # —— Interactive pick callback —— #
    highlights = {}
    def on_pick(event):
        ind0 = event.ind[0]
        sel  = idxs[ind0]
        x0, y0 = pc1[sel], contrast[sel]

        # Highlight point
        hl, = ax.plot(x0, y0, marker='o', markersize=12,
                      markerfacecolor='none',
                      markeredgecolor='red', linewidth=2)
        highlights[sel] = hl
        fig.canvas.draw_idle()

        # Popup window to show original patch
        img_t, gid = ds[sel]
        fig2, ax2 = plt.subplots(figsize=(4,4))
        ax2.imshow(img_t.squeeze().numpy(), cmap='gray',
                   vmin=0, vmax=1)
        ax2.axis('off')
        ax2.set_title(f"{group_names[int(gid)]} — idx {sel}")

        # Remove highlight when window closes
        def on_close(evt):
            art = highlights.pop(sel, None)
            if art:
                art.remove()
                fig.canvas.draw_idle()
        fig2.canvas.mpl_connect('close_event', on_close)
        plt.show()

    fig.canvas.mpl_connect('pick_event', on_pick)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    extract_and_plot_contrast()
