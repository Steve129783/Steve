# pca_analysis.py
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.covariance import MinCovDet  # 鲁棒性异常检测
import joblib

from o3_VAE_model import CachedImageDataset, VAE_ResSkip_50

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_path = r"E:\Project_VAE\V1\Ori_Pack\1_2_3_4\data_cache.pt"
    model_path = r"E:\Project_VAE\V1\oVAE_cache\1_2_3_4\best_model.pth"
    out_dir = r"E:\Project_VAE\V1\Visualization"
    os.makedirs(out_dir, exist_ok=True)

    # 1. 加载数据
    dataset = CachedImageDataset(cache_file=cache_path)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    # 2. 加载模型
    model = VAE_ResSkip_50(img_channels=1, latent_dim=24).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device)['model'])
    model.eval()

    # 3. 提取潜在向量
    mus = []
    with torch.no_grad():
        for imgs, _ in loader:
            mu, _, _ = model.encode(imgs.to(device))
            mus.append(mu.cpu())
    mus = np.concatenate(mus, axis=0)

    # 4. 标准化
    scaler = StandardScaler()
    mus_scaled = scaler.fit_transform(mus)

    # 5. PCA分析
    pca = PCA(n_components=0.99)  # 保留95%方差
    pca_features = pca.fit_transform(mus_scaled)
    
    # 6. 异常检测 (使用鲁棒性方法)
    robust_cov = MinCovDet().fit(pca_features)
    mahalanobis_dist = robust_cov.mahalanobis(pca_features)
    threshold = np.percentile(mahalanobis_dist, 95)
    outliers = mahalanobis_dist > threshold

    # 7. 保存结果
    np.save(os.path.join(out_dir, 'pca_features.npy'), pca_features)
    np.save(os.path.join(out_dir, 'outliers.npy'), outliers)
    np.save(os.path.join(out_dir, 'pca_2d.npy'), pca_features[:, :2])  # 前两个主成分
    np.save(os.path.join(out_dir, 'explained_variance.npy'), pca.explained_variance_ratio_)
    joblib.dump(pca, os.path.join(out_dir, 'pca_model.joblib'))
    joblib.dump(scaler, os.path.join(out_dir, 'scaler.joblib'))

    print(f"异常点数量: {outliers.sum()}/{len(outliers)}")
    print(f"主成分数量: {pca.n_components_} (解释方差: {pca.explained_variance_ratio_.sum():.2%})")

if __name__ == '__main__':
    main()