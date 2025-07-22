#!/usr/bin/env python3
import os
import re
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# =========================
# 1. Configuration
# =========================
file_name    = '1_2_3_4'
cache_file   = rf'E:\Project_SNV\1N\1_Pack\{file_name}\data_cache.pt'
model_path   = rf'E:\Project_SNV\1N\c1_cache\{file_name}\best_model.pth'
seed         = 42
batch_size   = 32    # reduce if memory is tight
num_workers  = 0
in_shape     = (50, 50)
device       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
max_per_group = 1500  # max points per group in visualization

# —— load group id → name mapping ——#
def load_group_names(cache_path):
    info_path = os.path.join(os.path.dirname(cache_path), 'info.txt')
    pattern   = re.compile(r'^\s*(\d+)\s*:\s*([^\s(]+)')
    mapping = {}
    with open(info_path, 'r', encoding='utf-8') as f:
        for line in f:
            m = pattern.match(line)
            if m:
                mapping[int(m.group(1))] = m.group(2)
    return mapping

group_names = load_group_names(cache_file)

# =========================
# 2. Dataset
# =========================
class CachedImageDataset(Dataset):
    def __init__(self, cache_path, transform=None):
        data = torch.load(cache_path, map_location='cpu')
        self.images    = data['images']           # [N,1,H,W]
        self.group_ids = torch.tensor(data['group_ids'], dtype=torch.long)
        self.paths     = data.get('paths', [None]*len(self.images))
        self.transform = transform

    def __len__(self):
        return len(self.group_ids)

    def __getitem__(self, idx):
        x = self.images[idx]
        if self.transform:
            x = self.transform(x)
        y = self.group_ids[idx]
        p = self.paths[idx]
        return x, y, p

# =========================
# 3. CNN with hidden features
# =========================
class CNNWithHidden(nn.Module):
    def __init__(self, in_shape, n_classes):
        super().__init__()
        c_list = [1,16,32,64]
        layers = []
        for i in range(2):
            layers += [
                nn.Conv2d(c_list[i], c_list[i+1], 3, padding='same'),
                nn.BatchNorm2d(c_list[i+1]),
                nn.ELU(),
                nn.Dropout2d(0.2),
                nn.MaxPool2d(2)
            ]
        layers.append(nn.Flatten())
        with torch.no_grad():
            dummy = torch.zeros(1,1,*in_shape)
            flat_dim = nn.Sequential(*layers)(dummy).shape[1]
        layers += [
            nn.Linear(flat_dim, 128),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(128, n_classes)
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        h = self.model[:-1](x)
        logits = self.model[-1](h)
        return logits, h

# =========================
# 4. Extract hidden features and contrast
# =========================
def extract_features_and_contrast():
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    ds = CachedImageDataset(cache_file)
    model = CNNWithHidden(in_shape, len(group_names)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model.eval()

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers)
    all_h, all_contrast, all_groups, all_paths = [], [], [], []
    with torch.no_grad():
        for x, y, p in loader:
            x = x.to(device)
            _, h = model(x)
            all_h.append(h.cpu().numpy())

            flat = x.view(x.size(0), -1)
            mins  = flat.min(dim=1).values.cpu().numpy()
            maxs  = flat.max(dim=1).values.cpu().numpy()
            means = flat.mean(dim=1).cpu().numpy()
            contrast = (means - mins) / (maxs - mins + 1e-6)
            all_contrast.extend(contrast)
            all_groups.extend(y.cpu().numpy())
            all_paths.extend(p)

    H = np.vstack(all_h)
    contrast = np.array(all_contrast)
    groups   = np.array(all_groups)
    return H, contrast, groups, all_paths

# =========================
# 5. Find most correlated hidden dimension
# =========================
def find_most_correlated_dim(H, contrast):
    D = H.shape[1]
    corrs = np.zeros(D)
    for i in range(D):
        corrs[i], _ = pearsonr(H[:, i], contrast)
    best_dim = np.argmax(np.abs(corrs))
    best_r   = corrs[best_dim]
    return best_dim, best_r, corrs

# =========================
# 6. Compute PC-correlations
# =========================
def find_pc_correlation(H, contrast, comp=0, seed=42):
    """
    Compute the Pearson correlation between the (comp+1)-th principal component of H and contrast.

    Parameters:
      H        np.ndarray, shape (N, D)   – matrix of hidden vectors
      contrast np.ndarray, shape (N,)     – array of contrast values
      comp     int                        – index of the principal component to use (0 for PC1, 1 for PC2, etc.)
      seed     int                        – random seed for PCA (optional)

    Returns:
      r        float   – Pearson correlation coefficient
      pval     float   – two-tailed p-value
    """
    pca = PCA(n_components=comp+1, random_state=seed).fit(H)
    pcs = pca.transform(H)       # shape (N, comp+1)
    pc  = pcs[:, comp]           # select the specified component

    r, pval = pearsonr(pc, contrast)
    return r, pval

# =========================
# 7. Main: export CSVs and plot
# =========================
if __name__ == "__main__":
    H, contrast, groups, paths = extract_features_and_contrast()

    # 7.1 export variance ranking
    variances = H.var(axis=0, ddof=0)
    D = variances.shape[0]
    sorted_idx = np.argsort(-variances)
    ranks = np.empty(D, dtype=int)
    for rank, dim in enumerate(sorted_idx, start=1):
        ranks[dim] = rank
    df_var = pd.DataFrame({
        'rank':     ranks,
        'dim':      np.arange(D),
        'variance': variances
    })
    var_csv = rf'E:\Project_SNV\1N\c1_cache\{file_name}\variance_rank.csv'
    df_var.to_csv(var_csv, index=False)
    print(f"Saved variance ranking to {var_csv}")

    # 7.2 export hidden-dim correlation ranking
    best_dim, best_r, all_corrs = find_most_correlated_dim(H, contrast)
    sorted_idx = np.argsort(-np.abs(all_corrs))
    ranks_corr = np.empty(D, dtype=int)
    for rank, dim in enumerate(sorted_idx, start=1):
        ranks_corr[dim] = rank
    df_corr = pd.DataFrame({
        'rank':     ranks_corr,
        'dim':      np.arange(D),
        'pearson_r': all_corrs
    })
    corr_csv = rf'E:\Project_SNV\1N\c1_cache\{file_name}\hidden_dim_correlation_rank.csv'
    df_corr.to_csv(corr_csv, index=False)
    print(f"Saved correlation ranking to {corr_csv}")

    # 7.3 compute PC1 correlation
    r_pc1, p_pc1 = find_pc_correlation(H, contrast, comp=0, seed=seed)
    print(f"PC1 vs contrast: r = {r_pc1:.3f}, p = {p_pc1:.3e}")
    print(f"Most correlated dim: H[:,{best_dim}] vs contrast: r = {best_r:.3f}")

    # 7.4 limit per-group and plot H[:,best_dim] vs contrast
    idxs = []
    for g in np.unique(groups):
        gi = np.where(groups == g)[0]
        if len(gi) > max_per_group:
            gi = np.random.RandomState(seed).choice(gi, max_per_group, replace=False)
        idxs.extend(gi.tolist())
    idxs = np.array(idxs)

    fig, ax = plt.subplots(figsize=(7,6))
    sc = ax.scatter(
        H[idxs, best_dim], contrast[idxs],
        c=groups[idxs], cmap='tab10', s=30, picker=True, alpha=0.7
    )
    # linear fit
    a, b = np.polyfit(H[idxs, best_dim], contrast[idxs], 1)
    xs = np.linspace(H[idxs, best_dim].min(), H[idxs, best_dim].max(), 200)
    ax.plot(xs, a*xs + b, color='black', lw=2, label=f"y={a:.3f}x+{b:.3f}")
    ax.legend()
    ax.set_title(f"H[:,{best_dim}] vs Contrast (r={best_r:.3f})")
    ax.set_xlabel(f"H[:, {best_dim}]")
    ax.set_ylabel("Contrast")
    cbar = fig.colorbar(sc, ax=ax, label="Group")
    ticks = sorted(group_names.keys())
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([group_names[k] for k in ticks])

    # interactive highlight & pop-up
    highlights = {}
    ds = CachedImageDataset(cache_file)
    def on_pick(event):
        ind0 = event.ind[0]
        idx = idxs[ind0]
        x0, y0 = H[idx, best_dim], contrast[idx]

        # red circle highlight
        hl, = ax.plot(x0, y0,
                      marker='o', markersize=12,
                      markerfacecolor='none',
                      markeredgecolor='red', linewidth=2)
        highlights[idx] = hl
        fig.canvas.draw_idle()

        # popup original patch
        img_tensor, gid, _ = ds[idx]
        img = img_tensor.squeeze().numpy()
        pf, pa = plt.subplots(figsize=(4,4))
        pa.imshow(img, cmap='gray', vmin=0, vmax=1)
        pa.axis('off')
        pa.set_title(f"{os.path.basename(paths[idx])}\nGroup {group_names[int(gid)]}")

        def on_close(ev):
            art = highlights.pop(idx, None)
            if art:
                art.remove()
                fig.canvas.draw_idle()
        pf.canvas.mpl_connect('close_event', on_close)
        plt.show()

    fig.canvas.mpl_connect('pick_event', on_pick)
    plt.tight_layout()
    plt.show()
