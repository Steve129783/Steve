#!/usr/bin/env python3
# 8_cluster.py

import os
import torch
import numpy as np
from PIL import Image
from skimage.measure import label, regionprops
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import csv
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import pickle

# ─────────── 配置区 ───────────
REASSEMBLED_DATA = r"E:\Project_VAE\V2\Reorganization\1_2_3_4\1\reassembled_defects.pt"
CVAE_CKPT        = r"E:\Project_VAE\V2\cvae_cache\best_model.pth"
OUT_CSV          = r"E:\Project_VAE\V2\Clusters\reassembled_cluster_results.csv"
PCA_OUT          = r"E:\Project_VAE\V2\Clusters\reassembled_pca_2d.npy"
LABELS_OUT       = r"E:\Project_VAE\V2\Clusters\reassembled_cluster_labels.npy"
INST_IMAGES_PKL  = r"E:\Project_VAE\V2\Clusters\inst_images.pkl"
INST_META_PKL    = r"E:\Project_VAE\V2\Clusters\inst_meta.pkl"

LATENT_DIM       = 24
N_CLUSTERS       = 6
BATCH_SIZE       = 64
NUM_WORKERS      = 3

# ── 辅助：PIL→Tensor ──
_to_tensor = transforms.ToTensor()
def pil_to_float_tensor(img: Image.Image) -> torch.Tensor:
    return _to_tensor(img).float()

# ── DefectDataset & collate ──
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
    # CSV
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
    # PCA 2D + labels
    pca2d = PCA(n_components=2).fit_transform(features)
    np.save(PCA_OUT, pca2d)
    np.save(LABELS_OUT, label_list)
    # pickle images/meta
    with open(INST_IMAGES_PKL, 'wb') as f:
        pickle.dump(images, f)
    with open(INST_META_PKL, 'wb') as f:
        pickle.dump(meta_list, f)
    print(f"\n结果已保存：\n- CSV: {OUT_CSV}\n- PCA: {PCA_OUT}\n- Labels: {LABELS_OUT}")

# ── 主流程 ──
def main():
    # 1. load defect instances (包含 PIL.Image.Image)
    defect_data = torch.load(
        REASSEMBLED_DATA,
        map_location='cpu',
        weights_only=False    # 允许加载 image 对象
    )
    print(f"Loaded {len(defect_data)} defect instances")

    # 2. build CVAE and load checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from c5_CVAE_model import CVAE_ResSkip_50

    model = CVAE_ResSkip_50(img_channels=1, latent_dim=LATENT_DIM, label_dim=1)\
                .to(device)

    ckpt = torch.load(CVAE_CKPT, map_location=device)
    state = ckpt.get('model', ckpt)

    # rename skip_conv_* → skip_*
    new_state = {}
    for k, v in state.items():
        nk = k.replace('skip_conv_mid', 'skip_mid')\
              .replace('skip_conv_low', 'skip_low')
        new_state[nk] = v

    model.load_state_dict(new_state)
    model.eval()

    # 3. DataLoader
    ds = DefectDataset(defect_data, transform=pil_to_float_tensor)
    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=defect_collate_fn
    )

    inst_vecs, inst_meta, inst_imgs = [], [], []

    # 4. 批量 encode（全部标为有缺陷 => label=1）
    with torch.no_grad():
        for imgs, metas, pimgs in loader:
            imgs = imgs.to(device)
            labs = torch.ones(imgs.size(0), 1, device=device)
            mu, _, _ = model.encode(imgs, labs)
            inst_vecs.append(mu.cpu().numpy())
            inst_meta.extend(metas)
            inst_imgs.extend(pimgs)

    X = np.vstack(inst_vecs)

    # 5. 标准化 & 聚类
    Xs = StandardScaler().fit_transform(X)
    km = KMeans(n_clusters=N_CLUSTERS, random_state=42)
    labels = km.fit_predict(Xs)

    print("\nClustering report:")
    print(f"- Inertia: {km.inertia_:.1f}")
    print(f"- Silhouette: {silhouette_score(Xs, labels):.3f}")
    print(f"- Counts per cluster: {np.bincount(labels)}")

    # 6. 保存所有结果
    save_results(inst_meta, labels, Xs, inst_imgs)

if __name__ == "__main__":
    main()
