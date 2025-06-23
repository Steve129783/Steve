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
from typing import List, Tuple, Dict

# ─────────── 配置区 ───────────
REASSEMBLED_DATA = r"E:\Project_VAE\V1\Reorganization\1_2_3_4\1\reassembled_defects.pt"
VAE_CKPT         = r"E:\Project_VAE\V1\oVAE_cache\1_2_3_4\best_model.pth"
OUT_CSV          = r"E:\Project_VAE\V1\Clusters\reassembled_cluster_results.csv"
PCA_OUT          = r"E:\Project_VAE\V1\Clusters\reassembled_pca_2d.npy"
LABELS_OUT       = r"E:\Project_VAE\V1\Clusters\reassembled_cluster_labels.npy"
INST_IMAGES_PKL  = r"E:\Project_VAE\V1\Clusters\inst_images.pkl"
INST_META_PKL    = r"E:\Project_VAE\V1\Clusters\inst_meta.pkl"
LATENT_DIM       = 24
N_CLUSTERS       = 6
BATCH_SIZE       = 64  # 并行批量大小
NUM_WORKERS      = 4   # DataLoader 并行数
# ─────────────────────────────────

# 将 PIL 图像转换为 FloatTensor 的顶层函数
_to_tensor = transforms.ToTensor()
def pil_to_float_tensor(img: Image.Image) -> torch.Tensor:
    return _to_tensor(img).float()

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
        return img, meta, item['image']  # return original PIL too

# 自定义 collate_fn
def defect_collate_fn(batch: List[Tuple[torch.Tensor, Dict, Image.Image]]) -> Tuple[torch.Tensor, List[Dict], List[Image.Image]]:
    imgs = torch.stack([item[0] for item in batch], dim=0)
    metas = [item[1] for item in batch]
    pil_imgs = [item[2] for item in batch]
    return imgs, metas, pil_imgs


def save_results(meta_list: List[Dict], label_list: np.ndarray, features: np.ndarray, images: List[Image.Image]):
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    # 保存 CSV
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
                'padded_r0': r0,
                'padded_c0': c0,
                'padded_r1': r1,
                'padded_c1': c1,
                'contributing_patches': ';'.join(md['contributing_patches']),
                'cluster': int(lbl)
            })
    # 保存 PCA 坐标和标签
    pca2d = PCA(n_components=2).fit_transform(features)
    np.save(PCA_OUT, pca2d)
    np.save(LABELS_OUT, label_list)
    # 保存 images 和 meta
    with open(INST_IMAGES_PKL, 'wb') as f:
        pickle.dump(images, f)
    with open(INST_META_PKL, 'wb') as f:
        pickle.dump(meta_list, f)
    # 打印结果路径
    print(f"\n结果已保存:")
    print(f"- 聚类CSV: {OUT_CSV}")
    print(f"- PCA坐标: {PCA_OUT}")
    print(f"- 聚类标签: {LABELS_OUT}")
    print(f"- 实例图列表: {INST_IMAGES_PKL}")
    print(f"- 实例元数据: {INST_META_PKL}")


def main():
    # 1. 加载数据
    defect_data = torch.load(REASSEMBLED_DATA, map_location='cpu', weights_only=False)
    print(f"成功加载 {len(defect_data)} 个重组缺陷实例")
    # 2. 加载 VAE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from o3_VAE_model import VAE_ResSkip_50
    vae = VAE_ResSkip_50(img_channels=1, latent_dim=LATENT_DIM).to(device)
    checkpoint = torch.load(VAE_CKPT, map_location=device)
    vae.load_state_dict(checkpoint.get('model', checkpoint))
    vae.eval()
    # 3. DataLoader
    dataset = DefectDataset(defect_data, transform=pil_to_float_tensor)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True,
                        collate_fn=defect_collate_fn)
    inst_vectors, inst_meta, inst_images = [], [], []
    # 4. 批量编码
    with torch.no_grad():
        for imgs, metas, pil_imgs in loader:
            mu, _, _ = vae.encode(imgs.to(device))
            inst_vectors.append(mu.cpu().numpy())
            inst_meta.extend(metas)
            inst_images.extend(pil_imgs)
    # 5. 标准化 & 聚类
    X = np.vstack(inst_vectors)
    Xs = StandardScaler().fit_transform(X)
    km = KMeans(n_clusters=N_CLUSTERS, init="k-means++",
                n_init=10, max_iter=300, tol=1e-4,
                algorithm="elkan", random_state=42)
    labels = km.fit_predict(Xs)
    print("\n聚类质量评估:")
    print(f"- 惯性 (Inertia): {km.inertia_:.1f}")
    print(f"- 轮廓系数 (Silhouette): {silhouette_score(Xs, labels):.3f}")
    print(f"- 各类样本数: {np.bincount(labels)}")
    # 6. 保存结果
    save_results(inst_meta, labels, Xs, inst_images)

if __name__ == "__main__":
    main()
