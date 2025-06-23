import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from o3_VAE_model import CachedImageDataset, VAE_ResSkip_50  # ensure correct path

def compute_reconstructions(model, img, axis_dim, span, steps):
    with torch.no_grad():
        mu, logvar, skips = model.encode(img)
        mu = mu.squeeze(0)
        f2, f3 = skips
        f2 = f2.squeeze(0)
        f3 = f3.squeeze(0)

        alphas = torch.linspace(
            mu[axis_dim] - span,
            mu[axis_dim] + span,
            steps,
            device=mu.device
        )
        z0 = mu.unsqueeze(0).repeat(steps, 1)
        z0[:, axis_dim] = alphas

        f2s = f2.unsqueeze(0).repeat(steps, 1, 1, 1)
        f3s = f3.unsqueeze(0).repeat(steps, 1, 1, 1)

        recons = model.decode(z0, (f2s, f3s))
    return recons

def plot_combined(recons, axis_dim, span, out_path, cmap_diff="seismic"):
    steps, _, H, W = recons.shape
    # Combine reconstructions in a row
    grid = make_grid(recons.cpu(), nrow=steps, padding=2)
    trav = grid[0].numpy()

    center = steps // 2
    base = recons[center]
    diffs = (recons - base).cpu().numpy()
    vmax = np.percentile(np.abs(diffs).ravel(), 99)

    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(steps * 1.2, 4),
        gridspec_kw={"height_ratios": [1, 1]},
        constrained_layout=True
    )
    ax1.imshow(trav, cmap="gray", vmin=0, vmax=1)
    ax1.axis("off")
    ax1.set_title(f"Traversal on z-dim {axis_dim} ±{span:.2f}")

    diff_strip = np.concatenate([diffs[i, 0] for i in range(steps)], axis=1)
    im = ax2.imshow(diff_strip, cmap=cmap_diff, vmin=-vmax, vmax=vmax)
    ax2.axis("off")
    ax2.set_title("Difference to center (hot=+)")

    cbar = fig.colorbar(
        im,
        ax=[ax2],
        orientation="horizontal",
        fraction=0.05,
        pad=0.02
    )
    cbar.set_label("Δ Reconstruction")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved traversal for dim {axis_dim} to {out_path}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_path = r"E:\Project_VAE\V1\Ori_Pack\1_2_3_4\data_cache.pt"
    model_path = r"E:\Project_VAE\V1\oVAE_cache\1_2_3_4\best_model.pth"
    output_dir = r"E:\Project_VAE\V1\Interpolation\A"
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset and model
    ds = CachedImageDataset(cache_file=cache_path)
    loader = DataLoader(
        ds,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    latent_dim = 24
    model = VAE_ResSkip_50(img_channels=1, latent_dim=latent_dim).to(device)
    ckpt = torch.load(model_path, map_location=device)
    state_dict = ckpt.get('model', ckpt.get('model_state_dict', ckpt))
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # Compute stds for each latent dimension
    zs = []
    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device)
            mu, logvar, _ = model.encode(imgs)
            z = model.reparameterize(mu, logvar)
            zs.append(z.cpu().numpy())
    zs = np.concatenate(zs, axis=0)
    stds = zs.std(axis=0)

    # Select a single example for traversal
    img, _ = ds[25659]
    img = img.unsqueeze(0).to(device)
    steps = 11

    # Loop over all latent dimensions
    for dim in range(latent_dim):
        span = stds[dim] * 3.0  # ±3σ range
        recons = compute_reconstructions(model, img, axis_dim=dim, span=span, steps=steps)
        out_png = os.path.join(output_dir, f"traversal_zdim{dim}.png")
        plot_combined(recons, axis_dim=dim, span=span, out_path=out_png)
