#!/usr/bin/env python3
import os
import argparse
import logging

# 必须在导入 plt 之前设置后端
import matplotlib
matplotlib.use("Qt5Agg")

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch
from torch.utils.data import DataLoader

from c4_CVAE_model import CachedImageDataset, CVAE_ResSimple_50

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)

def extract_mu_and_ds(model, cache_file, batch_size, device):
    """同时返回 mu, gids, 还有对应的 Dataset 实例 (用于交互时取图片)"""
    cache = torch.load(cache_file, map_location="cpu")
    masks  = cache["masks"]
    labels = (masks.view(len(masks), -1).sum(1) > 0).long().tolist()
    groups = cache["group_ids"]
    ds     = CachedImageDataset(cache_file, labels, groups)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    mus, gids = [], []
    model.eval()
    with torch.no_grad():
        for imgs, labs, gid_batch in loader:
            imgs, labs = imgs.to(device), labs.to(device)
            mu, _ = model.encode(imgs, labs)
            mus.append(mu.cpu().numpy())
            gids.extend(gid_batch.numpy())
    return np.vstack(mus), np.array(gids), ds

def interactive_plot(mu_train, gid_train, ds_train,
                     mu_new, gid_new, ds_new):
    """绘制交互式散点图，点击点会弹出对应 patch"""
    pca = PCA(n_components=2).fit(mu_train)
    pts_train = pca.transform(mu_train)
    pts_new   = pca.transform(mu_new)

    fig, ax = plt.subplots(figsize=(8,6))
    scatter_train = ax.scatter(
        pts_train[:,0], pts_train[:,1],
        c=gid_train, cmap='tab10',
        s=20, alpha=0.6, picker=5,
        label="train")
    scatter_new = ax.scatter(
        pts_new[:,0], pts_new[:,1],
        c='red', marker='x',
        s=50, picker=5,
        label="new_group")

    ax.set_title("Interactive PCA of Latent μ")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)

    def on_pick(event):
        # 不管点击的是哪一个 scatter，都能拿到点的 index
        ind = event.ind[0]
        # 区分 train vs new by查看 event.artist
        if event.artist is scatter_train:
            img_tensor, _, gid = ds_train[ind]
        else:
            img_tensor, _, gid = ds_new[ind]
        img = img_tensor.squeeze().numpy()
        # 弹出图片
        fig2, ax2 = plt.subplots(figsize=(4,4))
        ax2.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax2.set_title(f"Group {gid}")
        ax2.axis('off')
        fig2.show()

    fig.canvas.mpl_connect('pick_event', on_pick)
    plt.show()

def main(args):
    device = torch.device(args.device)

    # Load pretrained 3-class model
    model = CVAE_ResSimple_50(1, args.latent_dim, 1, num_groups=3).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    logging.info(f"Loaded model from {args.checkpoint}")

    # Extract train mus + dataset
    mu_train, gid_train, ds_train = extract_mu_and_ds(
        model, args.cache_train, args.batch_size, device)
    logging.info(f"Extracted train μ: {mu_train.shape[0]} samples")

    # Extract new-group mus + dataset
    mu_new, gid_new, ds_new = extract_mu_and_ds(
        model, args.cache_new, args.batch_size, device)
    logging.info(f"Extracted new μ: {mu_new.shape[0]} samples")

    # Launch interactive plot
    interactive_plot(mu_train, gid_train, ds_train,
                     mu_new, gid_new, ds_new)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cache-train", default=r"E:\Project_CNN\3_Pack\data_cache.pt")
    p.add_argument("--cache-new",   default=r"E:\Project_CNN\4_new\data_cache.pt")
    p.add_argument("--checkpoint",  default=r"E:\Project_CNN\v_cache\best_model.pth")
    p.add_argument("--latent-dim",  type=int, default=32)
    p.add_argument("--batch-size",  type=int, default=128)
    p.add_argument("--device",      choices=["cpu","cuda"], default="cpu")
    args = p.parse_args()
    main(args)
