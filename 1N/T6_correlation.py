#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import numpy as np
import cv2
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from c2_CNN_model import CNN

# ----------- 配置区 ------------
file_name   = 'h_c_l'
cache_file  = rf'E:\Project_SNV\1N\1_Pack\{file_name}\data_cache.pt'
model_path  = rf'E:\Project_SNV\1N\c1_cache\{file_name}\best_model.pth'
ratio_csv   = rf'E:\Project_SNV\1N\c1_cache\{file_name}\defect_ratios.csv'
out_dir     = rf'E:\Project_SNV\1N\c1_cache\{file_name}'
os.makedirs(out_dir, exist_ok=True)

seed        = 42
batch_size  = 32
num_workers = 0
in_shape    = (50, 50)
device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_pcs       = 5   # 保留前 5 个主成分

class CachedImageDataset(Dataset):
    def __init__(self, cache_path):
        cache = torch.load(cache_path, map_location='cpu')
        self.images = cache['images']
        self.paths  = cache.get(
            'paths',
            [f"idx_{i}" for i in range(len(self.images))]
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img  = self.images[idx].squeeze().numpy()
        gray = (img * 255).astype(np.uint8)
        path = str(self.paths[idx])   # <-- 确保这里返回字符串
        return gray, path

# —— 读取 gid→name ——#
def load_group_names(cache_path):
    info = {}
    p = re.compile(r'^\s*(\d+)\s*:\s*([^\s(]+)')
    path = os.path.join(os.path.dirname(cache_path), 'info.txt')
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            m = p.match(line)
            if m:
                info[int(m.group(1))] = m.group(2)
    return info

group_names = load_group_names(cache_file)

# --- 特征度量函数 ---
def laplacian_var(gray):
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def glcm_energy(gray):
    glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256,
                        symmetric=True, normed=True)
    return graycoprops(glcm, 'energy')[0,0]

def edge_density(gray):
    edges = cv2.Canny(gray, 100, 200)
    return edges.astype(bool).sum() / gray.size

def lbp_entropy(gray, P=8, R=1):
    lbp = local_binary_pattern(gray, P, R, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, P+3), density=True)
    p = hist[hist>0]
    return -np.sum(p * np.log2(p))

def sobel_mean(gray):
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    mag = np.hypot(gx, gy)
    return mag.mean()

def gabor_energy(gray, ksize=31, sigma=4.0, theta=0, lambd=10.0, gamma=0.5):
    kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma)
    resp   = cv2.filter2D(gray, cv2.CV_64F, kernel)
    return np.mean(resp**2)

def gray_variance(gray):
    return gray.var()

def hist_skew(gray):
    hist = np.bincount(gray.flatten(), minlength=256).astype(float)
    p    = hist / hist.sum()
    vals = np.arange(256)
    mu    = np.sum(vals * p)
    sigma = np.sqrt(np.sum((vals - mu)**2 * p))
    m3    = np.sum((vals - mu)**3 * p)
    return m3 / (sigma**3 + 1e-12)

def hist_kurtosis(gray):
    hist = np.bincount(gray.flatten(), minlength=256).astype(float)
    p    = hist / hist.sum()
    vals = np.arange(256)
    mu     = np.sum(vals * p)
    sigma2 = np.sum((vals - mu)**2 * p)
    m4     = np.sum((vals - mu)**4 * p)
    return m4 / (sigma2**2 + 1e-12) - 3

def contrast(gray):
    gray = gray.astype(np.float32)
    mn, mx, mean = gray.min(), gray.max(), gray.mean()
    return (mean - mn) / ((mx - mn) + 1e-6)

metrics = {
    'lap_var':      laplacian_var,
    'glcm_energy':  glcm_energy,
    'edge_den':     edge_density,
    'lbp_ent':      lbp_entropy,
    'sobel_mean':   sobel_mean,
    'gabor_energy': gabor_energy,
    'gray_var':     gray_variance,
    'hist_skew':    hist_skew,
    'hist_kurt':    hist_kurtosis,
    'contrast':     contrast
}

# —— 读取已有 defect_ratios.csv，做成 path→ratio 的映射 ——#
df_ratio   = pd.read_csv(ratio_csv, dtype={'path': str, 'defect_ratio': float})
ratio_dict = dict(zip(df_ratio['path'], df_ratio['defect_ratio']))

# —— 加载模型与数据集 ——#
ds    = CachedImageDataset(cache_file)
model = CNN(in_shape, len(group_names)).to(device)
model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
model.eval()

loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    num_workers=num_workers, collate_fn=lambda x: x)

# —— 提取深度特征 H 和各度量，包括 defect_ratio ——#
all_h = []
feature_vals = {name: [] for name in metrics}
feature_vals['defect_ratio'] = []

with torch.no_grad():
    for batch in loader:
        imgs, paths = zip(*batch)

        # 构建模型输入 [B,1,H,W]
        tensors = []
        for im in imgs:
            arr = im.squeeze() if isinstance(im, np.ndarray) else im.squeeze().cpu().numpy()
            t   = torch.from_numpy(arr).unsqueeze(0).float() / 255.0
            tensors.append(t)
        x = torch.stack(tensors, dim=0).to(device)

        # 前向提取 H
        _, h = model(x)
        all_h.append(h.cpu().numpy())

        # 按 path 精确映射 defect_ratio
        for p in paths:
            key = p  # p is already string
            feature_vals['defect_ratio'].append(ratio_dict[key])

        # 其它度量
        for im in imgs:
            arr = im.squeeze() if isinstance(im, np.ndarray) else im.squeeze().cpu().numpy()
            # 保证 uint8
            if arr.dtype != np.uint8:
                if np.issubdtype(arr.dtype, np.floating) and arr.max() <= 1.0:
                    arr = (arr * 255).astype(np.uint8)
                else:
                    arr = arr.astype(np.uint8)
            for name, fn in metrics.items():
                feature_vals[name].append(fn(arr))

H = np.vstack(all_h)

# —— 计算 PCA & Pearson 相关 ——#
pca = PCA(n_components=n_pcs, random_state=seed).fit(H)
pcs = pca.transform(H)  # shape [N, n_pcs]

records = []
for i in range(n_pcs):
    pc_label = f'PC{i+1}'
    for metric_name, vals in feature_vals.items():
        r, _ = pearsonr(pcs[:, i], np.array(vals))
        records.append({
            'PC':        pc_label,
            'metric':    metric_name,
            'pearson_r': r
        })

df = pd.DataFrame(records)
df_pivot = df.pivot(index='metric', columns='PC', values='pearson_r')
df_pivot.columns.name = None
df_pivot.index.name   = 'metric'

out_csv = os.path.join(out_dir, 'pc_metric_correlations_with_ratio.csv')
df_pivot.to_csv(out_csv)
print(f'Correlation table saved to: {out_csv}')
