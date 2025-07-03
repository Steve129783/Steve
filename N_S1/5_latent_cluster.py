import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch
import numpy as np
from torch.utils.data import DataLoader
from c4_CVAE_model import CachedImageDataset, CVAE_ResSimple_50

# Paths
CACHE_FILE = r"E:\Project_CNN\3_Pack\data_cache.pt"
CHECKPOINT = r"E:\Project_CNN\v_cache\best_model.pth"

# Load cache
cache = torch.load(CACHE_FILE, map_location="cpu")
groups = cache["group_ids"]
labels = (cache["masks"].view(len(cache["masks"]), -1).sum(1) > 0).long().tolist()
group_names = {
    0: "correct",
    1: "high",
    2: "low"
}

# Dataset & DataLoader
ds = CachedImageDataset(CACHE_FILE, labels, groups)
loader = DataLoader(ds, batch_size=128, shuffle=False)

# Load model
device = torch.device("cpu")
latent_dim = 32  # adjust to your trained latent_dim
model = CVAE_ResSimple_50(1, latent_dim, 1, len(group_names)).to(device)
model.load_state_dict(torch.load(CHECKPOINT, map_location="cpu"))
model.eval()

# Extract latent means
all_mu = []
all_g = []
with torch.no_grad():
    for img, lbl, gid in loader:
        mu, _ = model.encode(img.to(device), lbl.to(device))
        all_mu.append(mu.cpu().numpy())
        all_g.extend(gid.numpy())
all_mu = np.concatenate(all_mu, axis=0)
all_g = np.array(all_g)

# PCA to 2D
pca = PCA(n_components=2)
mu_2d = pca.fit_transform(all_mu)

# Interactive plot
fig, ax = plt.subplots(figsize=(8,6))
scatter = ax.scatter(mu_2d[:,0], mu_2d[:,1], c=all_g, cmap='tab10', s=20, picker=True)
ax.set_title("Interactive PCA of Latent Means")
ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
plt.colorbar(scatter, label="Group ID")

def on_pick(event):
    ind = event.ind[0]
    img_tensor, _, gid = ds[ind]
    img = img_tensor.squeeze().numpy()
    cluster = all_g[ind]
    name = group_names.get(cluster, str(cluster))
    # Display image in new window
    fig2, ax2 = plt.subplots(figsize=(4,4))
    ax2.imshow(img, cmap='gray', vmin=0, vmax=1)
    ax2.set_title(f"{name} (Cluster {cluster}) — Index {ind}")
    ax2.axis('off')
    plt.show()

fig.canvas.mpl_connect('pick_event', on_pick)
plt.show()
