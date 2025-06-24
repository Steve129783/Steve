import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.covariance import MinCovDet
import umap
from sklearn.cluster import DBSCAN
import joblib
import matplotlib.pyplot as plt

from c5_CVAE_model import CachedImageDataset, CVAE_ResSkip_50


def extract_latents(cache_path, model_path, batch_size=32, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = torch.load(cache_path, map_location="cpu")
    masks = cache["masks"]
    labels = (masks.view(len(masks), -1).sum(1) > 0).long().tolist()

    ds = CachedImageDataset(cache_file=cache_path, label_list=labels)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    model = CVAE_ResSkip_50(img_channels=1, latent_dim=24, label_dim=1).to(device)
    ckpt = torch.load(model_path, map_location=device)
    state_dict = ckpt if not isinstance(ckpt, dict) or 'model' not in ckpt else ckpt['model']
    new_state = {k.replace('skip_conv_mid','skip_mid').replace('skip_conv_low','skip_low'): v
                 for k, v in state_dict.items()}
    model.load_state_dict(new_state)
    model.eval()

    mus = []
    with torch.no_grad():
        for imgs, labs in loader:
            imgs, labs = imgs.to(device), labs.to(device)
            mu, _, _ = model.encode(imgs, labs)
            mus.append(mu.cpu().numpy())
    return np.concatenate(mus, axis=0)


def run_pca(latents, out_dir, variance=0.99):
    os.makedirs(out_dir, exist_ok=True)
    scaler = StandardScaler()
    lat_scaled = scaler.fit_transform(latents)

    pca = PCA(n_components=variance)
    feats = pca.fit_transform(lat_scaled)

    cov = MinCovDet().fit(feats)
    md = cov.mahalanobis(feats)
    thr = np.percentile(md, 95)
    outliers = md > thr

    # Save PCA outputs
    np.save(os.path.join(out_dir, 'pca_features.npy'), feats)
    np.save(os.path.join(out_dir, 'pca_outliers.npy'), outliers)
    np.save(os.path.join(out_dir, 'pca_2d.npy'), feats[:, :2])
    np.save(os.path.join(out_dir, 'explained_variance_ratio.npy'), pca.explained_variance_ratio_)
    joblib.dump(pca, os.path.join(out_dir, 'pca_model.joblib'))
    joblib.dump(scaler, os.path.join(out_dir, 'pca_scaler.joblib'))

    print(f"PCA: {pca.n_components_} components, explained {pca.explained_variance_ratio_.sum():.2%} variance")
    print(f"PCA outliers: {int(outliers.sum())}/{len(outliers)}")

    plt.figure(figsize=(6,6))
    plt.scatter(feats[:,0], feats[:,1], s=5, alpha=0.6)
    plt.title('PCA 2D projection')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'pca_2d.png'), dpi=300)
    plt.close()


def run_umap(latents, out_dir, n_neighbors=15, min_dist=0.1, eps=0.5, min_samples=10):
    os.makedirs(out_dir, exist_ok=True)
    scaler = StandardScaler()
    lat_scaled = scaler.fit_transform(latents)

    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                        n_components=2, random_state=42)
    proj = reducer.fit_transform(lat_scaled)

    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(proj)
    labels = clustering.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    print(f"UMAP: detected {n_clusters} clusters, {n_noise} noise points.")

    # Save UMAP outputs
    np.save(os.path.join(out_dir, 'umap_2d.npy'), proj)
    np.save(os.path.join(out_dir, 'umap_labels.npy'), labels)
    joblib.dump(reducer, os.path.join(out_dir, 'umap_model.joblib'))
    joblib.dump(scaler, os.path.join(out_dir, 'umap_scaler.joblib'))

    plt.figure(figsize=(6,6))
    unique = set(labels)
    for lab in unique:
        mask = labels == lab
        color = 'black' if lab == -1 else None
        plt.scatter(proj[mask,0], proj[mask,1], s=5, alpha=0.6, label=('noise' if lab==-1 else f'cluster {lab}'), c=color)
    plt.legend(markerscale=3)
    plt.title('UMAP projection with DBSCAN')
    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'umap_2d.png'), dpi=300)
    plt.close()


def main():
    cache_path = r"E:\Project_VAE\V2\Pack_json_png\Ori_drop\data_cache.pt"
    model_path = r"E:\Project_VAE\V2\cvae_cache\best_model.pth"
    base_out = r"E:\Project_VAE\V2\visual_latent"

    latents = extract_latents(cache_path, model_path)

    run_pca(latents, os.path.join(base_out, 'pca'))
    run_umap(latents, os.path.join(base_out, 'umap'))

    print("All analyses completed. Results in:", base_out)

if __name__ == '__main__':
    main()
