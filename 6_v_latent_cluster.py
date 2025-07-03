import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch
import numpy as np
from torch.utils.data import DataLoader
from N_S1.v5_CVAE_model import CachedImageDataset, CVAE_ResSimple_50

# Paths
CACHE_FILE = r"E:\Project_CNN\2_Pack\data_cache.pt"
CHECKPOINT = r"E:\Project_CNN\v_cache\best_model.pth"

# Load cache
cache = torch.load(CACHE_FILE, map_location="cpu")
groups = cache["group_ids"]
labels = (cache["masks"].view(len(cache["masks"]), -1).sum(1) > 0).long().tolist()
group_names = {0: "correct", 1: "high", 2: "low"}

# Dataset & DataLoader
ds = CachedImageDataset(CACHE_FILE, labels, groups)
loader = DataLoader(ds, batch_size=128, shuffle=False)

# Load model
device = torch.device("cpu")
latent_dim = 128  # your trained latent_dim
model = CVAE_ResSimple_50(1, latent_dim, 1, len(group_names)).to(device)
model.load_state_dict(torch.load(CHECKPOINT, map_location="cpu"))
model.eval()

# Extract all μ
all_mu, all_g = [], []
with torch.no_grad():
    for img, lbl, gid in loader:
        mu, _ = model.encode(img.to(device), lbl.to(device))
        all_mu.append(mu.cpu().numpy())
        all_g.extend(gid.numpy())
all_mu = np.concatenate(all_mu, axis=0)
all_g  = np.array(all_g)

# PCA → 2D
pca = PCA(n_components=2)
mu_pca = pca.fit_transform(all_mu)

# Plot PCA only
fig, ax = plt.subplots(figsize=(8,6))
scatter = ax.scatter(
    mu_pca[:,0], mu_pca[:,1],
    c=all_g, cmap='tab10', s=20, picker=True
)
ax.set_title("PCA of Latent Means")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
plt.colorbar(scatter, label="Group ID")

def on_pick(event):
    ind = event.ind[0]
    img_tensor, _, gid = ds[ind]
    img = img_tensor.squeeze().numpy()
    name = group_names.get(all_g[ind], str(all_g[ind]))
    fig2, ax2 = plt.subplots(figsize=(4,4))
    ax2.imshow(img, cmap='gray', vmin=0, vmax=1)
    ax2.set_title(f"{name} — Index {ind}")
    ax2.axis('off')
    plt.show()

fig.canvas.mpl_connect('pick_event', on_pick)
plt.tight_layout()
plt.show()
