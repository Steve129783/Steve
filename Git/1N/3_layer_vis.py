#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from PIL import Image
import numpy as np
from c2_CNN_model import CNN

# --- Visualization function (supports ndarray, image files, or .pt cache + original path) ---
def visualize_feature_and_heatmaps(
    model: nn.Module,
    img_input,                        # Can be np.ndarray, image path (.png/.jpg…), or .pt cache path
    layer_idx: int = 0,
    in_shape: tuple = (50,50),
    device: torch.device = torch.device('cpu'),
    n_cols: int = 4,
    cache_pt: str = None,            # If provided, the function will look up the patch from .pt using original path
):
    # 1) Load or retrieve arr_norm, arr.dtype, info based on input type
    if isinstance(img_input, np.ndarray):
        arr = img_input
        if np.issubdtype(arr.dtype, np.integer):
            info = np.iinfo(arr.dtype)
            arr_norm = (arr.astype(np.float32) - info.min) / (info.max - info.min)
        else:
            info = type('i', (), {'min':0, 'max':1})
            arr_norm = arr.astype(np.float32)

    elif isinstance(img_input, str) and cache_pt and img_input in torch.load(cache_pt)['paths']:
        # From .pt cache, retrieve corresponding patch using the original path
        cache = torch.load(cache_pt, map_location='cpu')
        idx = cache['paths'].index(img_input)
        tensor = cache['images'][idx]         # [1,H,W]
        arr = tensor.squeeze().numpy()
        # Already normalized float32
        info = type('i', (), {'min':0, 'max':1})
        arr_norm = arr.astype(np.float32)

    else:
        # Directly open as image file
        pil = Image.open(img_input)
        arr = np.array(pil)
        info = np.iinfo(arr.dtype)
        arr_norm = (arr.astype(np.float32) - info.min) / (info.max - info.min)

    # 2) Convert to Tensor and resize to in_shape
    x = torch.from_numpy(arr_norm).unsqueeze(0).unsqueeze(0).to(device)
    x = F.interpolate(x, size=in_shape, mode='bilinear', align_corners=False)

    # 3) Hook to extract output of the specified conv layer
    convs = [m for m in model.model if isinstance(m, nn.Conv2d)]
    conv = convs[layer_idx]
    feats = {}
    def hook(m, i, o):
        feats['maps'] = o.detach().cpu()[0]
    h = conv.register_forward_hook(hook)
    model.eval()
    with torch.no_grad():
        _ = model(x)
    h.remove()

    fmap = feats['maps']           # (C, Hf, Wf)
    n_maps, Hf, Wf = fmap.shape
    n_rows = (n_maps + n_cols - 1) // n_cols

    # 4) Original image
    plt.figure(figsize=(2,2))
    plt.imshow(arr_norm, cmap='gray', vmin=0, vmax=1)
    plt.title(f'Original (dtype={arr.dtype}, range=[{info.min},{info.max}])')
    plt.axis('off'); plt.show(block=False)

    # 5) Feature maps in grayscale
    plt.figure(figsize=(n_cols,n_rows))
    for i in range(n_maps):
        ax = plt.subplot(n_rows, n_cols, i+1)
        mi, ma = float(fmap[i].min()), float(fmap[i].max())
        ax.imshow(fmap[i], cmap='gray', vmin=mi, vmax=ma)
        ax.axis('off')
        ax.text(0.95,0.05,str(i),color='white',fontsize=8,
                ha='right',va='bottom',transform=ax.transAxes,
                bbox=dict(fc='black',alpha=0.5))
    plt.suptitle(f'Layer {layer_idx} Feature Maps (Gray)')
    plt.tight_layout(); plt.show(block=False)

    # 6) Heatmaps with colorbar
    fig, axes = plt.subplots(n_rows,n_cols,figsize=(n_cols,n_rows))
    axes = axes.flatten()
    norm = plt.Normalize(vmin=fmap.min(), vmax=fmap.max())
    for i in range(n_maps):
        axes[i].imshow(fmap[i], cmap='jet', vmin=float(fmap[i].min()), vmax=float(fmap[i].max()))
        axes[i].axis('off')
        axes[i].text(0.95,0.05,str(i),color='white',fontsize=8,
                     ha='right',va='bottom',transform=axes[i].transAxes,
                     bbox=dict(fc='black',alpha=0.5))
    for j in range(n_maps, len(axes)):
        axes[j].axis('off')
    cax = fig.add_axes([0.92,0.15,0.02,0.7])
    mappable = plt.cm.ScalarMappable(cmap='jet', norm=norm)
    mappable.set_array(fmap)
    fig.colorbar(mappable, cax=cax)
    plt.suptitle(f'Layer {layer_idx} Feature Maps (Heat)')
    plt.tight_layout(rect=[0,0,0.9,1]); plt.show(block=False)

    # 7) Absolute activation strength heatmaps
    strength = np.abs(fmap)
    fig2, axes2 = plt.subplots(n_rows,n_cols,figsize=(n_cols,n_rows))
    axes2 = axes2.flatten()
    norm2 = plt.Normalize(vmin=strength.min(), vmax=strength.max())
    for i in range(n_maps):
        axes2[i].imshow(strength[i], cmap='hot', norm=norm2)
        axes2[i].axis('off')
        axes2[i].text(0.95,0.05,str(i),color='white',fontsize=8,
                      ha='right',va='bottom',transform=axes2[i].transAxes,
                      bbox=dict(fc='black',alpha=0.5))
    for j in range(n_maps, len(axes2)):
        axes2[j].axis('off')
    cax2 = fig2.add_axes([0.92,0.15,0.02,0.7])
    m2 = plt.cm.ScalarMappable(cmap='hot', norm=norm2)
    m2.set_array(strength)
    fig2.colorbar(m2, cax=cax2)
    plt.suptitle(f'Layer {layer_idx} Feature Strength (|act|)')
    plt.tight_layout(rect=[0,0,0.9,1]); plt.show(block=False)

    # 8) Mean activation strength bar plot
    mean_str = strength.reshape(n_maps, -1).mean(axis=1)
    fig3, ax3 = plt.subplots(figsize=(10,3))
    bars = ax3.bar(range(n_maps), mean_str,
                   color=plt.cm.hot((mean_str-mean_str.min())/(mean_str.max()-mean_str.min()+1e-8)),
                   edgecolor='black', linewidth=0.5)
    ax3.set_title('Mean |activation| per channel')
    ax3.set_xlabel('Channel'); ax3.set_ylabel('Mean |act|')
    ax3.set_ylim(0, mean_str.max()*1.1)
    for b in bars:
        h = b.get_height()
        ax3.text(b.get_x()+b.get_width()/2, h+0.005, f'{h:.3f}',
                 ha='center',va='bottom',fontsize=6)
    plt.tight_layout(); plt.show(block=False)

    plt.pause(0.1)
    input("Press Enter to close all windows…")


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1) Initialize model
    n_group = '1_2_3_4'
    ckpt = rf'E:\Project_SNV\1N\c2_cache\{n_group}\best_model.pth'
    sd = torch.load(ckpt, map_location='cpu')
    lin = [k for k,v in sd.items() if k.endswith('weight') and v.dim()==2]
    n_cls = sd[sorted(lin)[-1]].shape[0]
    model = CNN(in_shape=(50,50), n_classes=n_cls)
    model.load_state_dict(sd); model.to(device)

    # 2) Example call: directly pass .pt cache + original image path
    cache_file = rf"E:\Project_SNV\1N\1_Pack\{n_group}\data_cache.pt"
    orig_path = rf"E:\Project_SNV\0S\6_patch\1\0_6_4.png"
    visualize_feature_and_heatmaps(
        model,
        img_input=orig_path,
        layer_idx=0,
        device=device,
        cache_pt=cache_file
    )
