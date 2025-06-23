# pca_analysis.py

import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.covariance import MinCovDet  # 鲁棒性异常检测
import joblib

from c5_CVAE_model import CachedImageDataset, CVAE_ResSkip_50

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_path = r"E:\Project_VAE\V2\Pack_json_png\Ori_drop\data_cache.pt"
    model_path = r"E:\Project_VAE\V2\cvae_cache\best_model.pth"
    out_dir    = r"E:\Project_VAE\V2\visual_latent"
    os.makedirs(out_dir, exist_ok=True)

    # 1. 从 cache 中加载 masks，生成 label_list
    cache      = torch.load(cache_path, map_location="cpu")
    masks      = cache["masks"]  # Tensor[N,1,50,50]
    label_list = (masks.view(len(masks), -1).sum(1) > 0).long().tolist()

    # 2. 实例化 Dataset（需要 cache_file 和 label_list）
    dataset = CachedImageDataset(cache_file=cache_path,
                                 label_list=label_list)
    loader  = DataLoader(dataset, batch_size=32, shuffle=False)

    # 3. 构建模型
    model = CVAE_ResSkip_50(img_channels=1,
                            latent_dim=24,
                            label_dim=1).to(device)

    # 4. 加载 checkpoint 并重命名 key
    ckpt = torch.load(model_path, map_location=device)
    # 如果你保存时用了 `torch.save({'model': model.state_dict()}, ...)`，则：
    #    state_dict = ckpt['model']
    # 否则直接：
    state_dict = ckpt

    new_state = {}
    for k, v in state_dict.items():
        # 把旧命名 skip_conv_mid → skip_mid、skip_conv_low → skip_low
        nk = k.replace('skip_conv_mid', 'skip_mid') \
              .replace('skip_conv_low', 'skip_low')
        new_state[nk] = v

    model.load_state_dict(new_state)
    model.eval()

    # 5. 提取潜在向量（带条件 label）
    mus = []
    with torch.no_grad():
        for imgs, labs in loader:
            imgs, labs = imgs.to(device), labs.to(device)
            mu, _, _   = model.encode(imgs, labs)
            mus.append(mu.cpu().numpy())
    mus = np.concatenate(mus, axis=0)

    # 6. 标准化
    scaler     = StandardScaler()
    mus_scaled = scaler.fit_transform(mus)

    # 7. PCA 分析（保留 99% 方差）
    pca = PCA(n_components=0.99)
    pca_features = pca.fit_transform(mus_scaled)

    # 8. 异常检测
    robust_cov = MinCovDet().fit(pca_features)
    mdist      = robust_cov.mahalanobis(pca_features)
    threshold  = np.percentile(mdist, 95)
    outliers   = mdist > threshold

    # 9. 保存结果
    np.save(os.path.join(out_dir, 'pca_features.npy'),      pca_features)
    np.save(os.path.join(out_dir, 'outliers.npy'),          outliers)
    np.save(os.path.join(out_dir, 'pca_2d.npy'),            pca_features[:, :2])
    np.save(os.path.join(out_dir, 'explained_variance.npy'),
            pca.explained_variance_ratio_)
    joblib.dump(pca,    os.path.join(out_dir, 'pca_model.joblib'))
    joblib.dump(scaler, os.path.join(out_dir, 'scaler.joblib'))

    print(f"异常点数量: {int(outliers.sum())}/{len(outliers)}")
    print(f"主成分数量: {pca.n_components_} "
          f"(解释方差: {pca.explained_variance_ratio_.sum():.2%})")

if __name__ == '__main__':
    main()
