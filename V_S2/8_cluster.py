#!/usr/bin/env python3
# 8_cluster_with_umap_auto.py

import os
import torch
import numpy as np
from PIL import Image
from skimage.measure import label, regionprops
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import umap
import csv
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import pickle

# ─────────── 配置区 ───────────
REASSEMBLED_DATA = r"E:\Project_VAE\V2\Reorganization\1_2_3_4\1\reassembled_defects.pt"
CVAE_CKPT        = r"E:\Project_VAE\V2\cvae_cache\best_model.pth"
OUT_CSV          = r"E:\Project_VAE\V2\Clusters\reassembled_cluster_results.csv"
PCA_OUT_2D       = r"E:\Project_VAE\V2\Clusters\reassembled_pca_2d.npy"
UMAP_OUT_2D      = r"E:\Project_VAE\V2\Clusters\reassembled_umap_2d.npy"
LABELS_OUT       = r"E:\Project_VAE\V2\Clusters\reassembled_cluster_labels.npy"
INST_IMAGES_PKL  = r"E:\Project_VAE\V2\Clusters\inst_images.pkl"
INST_META_PKL    = r"E:\Project_VAE\V2\Clusters\inst_meta.pkl"

LATENT_DIM       = 24
BATCH_SIZE       = 64
NUM_WORKERS      = 3

# ── 辅助：PIL→Tensor ──
_to_tensor = transforms.ToTensor()
def pil_to_float_tensor(img: Image.Image) -> torch.Tensor:
    return _to_tensor(img).float()

# ── Dataset & collate ──
class DefectDataset(Dataset):
    def __init__(self, defect_data, transform=None):
        self.data = defect_data
        self.transform = transform
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        img = item['image']
        if self.transform:
            img = self.transform(img)
        meta = {
            'base': item['base'],
            'instance_id': item['instance_id'],
            'padded_bbox': item['padded_bbox'],
            'contributing_patches': item['contributing_patches'],
            'source_tif': item.get('source_tif', None)
        }
        return img, meta, item['image']

def defect_collate_fn(batch):
    imgs  = torch.stack([b[0] for b in batch], dim=0)
    metas = [b[1] for b in batch]
    pimgs = [b[2] for b in batch]
    return imgs, metas, pimgs

# ── 保存结果 ──
def save_results(meta_list, label_list, features, images):
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    # 1) 写 CSV
    with open(OUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'base','instance_id',
            'padded_r0','padded_c0','padded_r1','padded_c1',
            'contributing_patches','cluster'
        ])
        writer.writeheader()
        for md, lbl in zip(meta_list, label_list):
            r0, c0, r1, c1 = md['padded_bbox']
            writer.writerow({
                'base': md['base'],
                'instance_id': md['instance_id'],
                'padded_r0': r0, 'padded_c0': c0,
                'padded_r1': r1, 'padded_c1': c1,
                'contributing_patches': ';'.join(md['contributing_patches']),
                'cluster': int(lbl)
            })

    # 2) 保存标签数组
    np.save(LABELS_OUT, label_list)

    # 3) PCA 2D
    pca2d = PCA(n_components=2).fit_transform(features)
    np.save(PCA_OUT_2D, pca2d)

    # 4) UMAP 2D
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    umap2d  = reducer.fit_transform(features)
    np.save(UMAP_OUT_2D, umap2d)

    # 5) Pickle images & meta
    with open(INST_IMAGES_PKL, 'wb') as f:
        pickle.dump(images, f)
    with open(INST_META_PKL, 'wb') as f:
        pickle.dump(meta_list, f)

    print(f"\n结果已保存：")
    print(f"- CSV            : {OUT_CSV}")
    print(f"- Labels         : {LABELS_OUT}")
    print(f"- PCA 2D         : {PCA_OUT_2D}")
    print(f"- UMAP 2D        : {UMAP_OUT_2D}")
    print(f"- Images pickle  : {INST_IMAGES_PKL}")

# ── 主流程 ──
def main():
    # 1. load defect instances
    defect_data = torch.load(REASSEMBLED_DATA, map_location='cpu', weights_only=False)
    print(f"Loaded {len(defect_data)} defect instances.")

    # 2. build CVAE and load checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from c5_CVAE_model import CVAE_ResSkip_50

    model = CVAE_ResSkip_50(img_channels=1, latent_dim=LATENT_DIM, label_dim=1).to(device)
    ckpt  = torch.load(CVAE_CKPT, map_location=device)
    state = ckpt.get('model', ckpt)
    new_state = {k.replace('skip_conv_mid','skip_mid')
                   .replace('skip_conv_low','skip_low'): v
                 for k, v in state.items()}
    model.load_state_dict(new_state)
    model.eval()

    # 3. DataLoader
    ds     = DefectDataset(defect_data, transform=pil_to_float_tensor)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True,
                        collate_fn=defect_collate_fn)

    inst_vecs, inst_meta, inst_imgs = [], [], []

    # 4. encode（全标为有缺陷）
    with torch.no_grad():
        for imgs, metas, pimgs in loader:
            imgs = imgs.to(device)
            labs = torch.ones(imgs.size(0), 1, device=device)
            mu, _, _ = model.encode(imgs, labs)
            inst_vecs.append(mu.cpu().numpy())
            inst_meta.extend(metas)
            inst_imgs.extend(pimgs)
    X = np.vstack(inst_vecs)

    # 5. 标准化
    Xs = StandardScaler().fit_transform(X)

    # 6. 自动选择最佳簇数（基于轮廓系数）
    best_n = 2
    best_score = -1
    max_clusters = 10

    print("\nSearching optimal number of clusters...")
    for n in range(2, max_clusters + 1):
        km = KMeans(n_clusters=n, random_state=42)
        lbls = km.fit_predict(Xs)
        score = silhouette_score(Xs, lbls)
        print(f"n_clusters={n:2d}  silhouette={score:.4f}")
        if score > best_score:
            best_score = score
            best_n = n

    print(f"\n最佳簇数: {best_n}，对应 silhouette={best_score:.4f}")

    # 最优簇数下的最终聚类
    km = KMeans(n_clusters=best_n, random_state=42)
    labels = km.fit_predict(Xs)

    print("\n最终聚类报告:")
    print(f"- 簇数        : {best_n}")
    print(f"- Inertia    : {km.inertia_:.1f}")
    print(f"- Silhouette : {silhouette_score(Xs, labels):.3f}")
    print(f"- 各簇样本量  : {np.bincount(labels)}")

    # 7. 保存所有结果（包含 PCA & UMAP & 标签）
    save_results(inst_meta, labels, Xs, inst_imgs)

if __name__ == "__main__":
    main()
