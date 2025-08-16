#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from c1_frozen import CNN

# --------------------------
# Visualization utility (supports PNG paths or .pt cache lookup)
# --------------------------
def visualize_feature_and_heatmaps(
    model: nn.Module,
    img_path: str,
    layer_idx: int = 0,
    in_shape: tuple = (50,50),
    device: torch.device = torch.device('cpu'),
    n_cols: int = 4,
    apply_mask: bool = False,
    cache_pt: str = None,          # .pt cache path to load image tensor
):
    # 1) Load image either from .pt or from file
    if cache_pt and img_path in torch.load(cache_pt, map_location='cpu')['paths']:
        data = torch.load(cache_pt, map_location='cpu')
        idx = data['paths'].index(img_path)
        tensor = data['images'][idx]  # shape [1,H,W]
        arr_f = tensor.squeeze().numpy()
        # assume arr_f already in [0,1]
        info = type('i', (), {'min':0, 'max':1})
    else:
        pil_img = Image.open(img_path)
        arr = np.array(pil_img)
        if np.issubdtype(arr.dtype, np.integer):
            info = np.iinfo(arr.dtype)
            arr_f = (arr.astype(np.float32) - info.min) / (info.max - info.min)
        else:
            arr_f = arr.astype(np.float32)
            arr_f = (arr_f - arr_f.min()) / (arr_f.max() - arr_f.min() + 1e-8)

    # 2) Prepare tensor and resize
    x = torch.from_numpy(arr_f).unsqueeze(0).unsqueeze(0).to(device)
    x = F.interpolate(x, size=in_shape, mode='bilinear', align_corners=False)

    # 3) Hook target conv
    conv_layers = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    assert 0 <= layer_idx < len(conv_layers), "layer_idx out of range"
    target_conv = conv_layers[layer_idx]
    feats = {}
    def hook_fn(module, input, output):
        fmap = output.detach().cpu()[0]
        if apply_mask and layer_idx == 0 and hasattr(model, 'chan_mask'):
            mask = model.chan_mask.cpu().view(-1)
            fmap = fmap * mask.unsqueeze(-1).unsqueeze(-1)
        feats['maps'] = fmap
    handle = target_conv.register_forward_hook(hook_fn)

    # 4) Forward pass
    model.eval()
    with torch.no_grad():
        _ = model(x)
    handle.remove()

    # 5) Plotting
    fmap = feats['maps']  # (C, Hf, Wf)
    C, Hf, Wf = fmap.shape
    n_rows = math.ceil(C / n_cols)

    # Original patch
    plt.figure(figsize=(3,3))
    plt.imshow(arr_f, cmap='gray', vmin=0, vmax=1)
    plt.title('Original Patch')
    plt.axis('off'); plt.show(block=False)

    # Grayscale maps
    fig1, axes1 = plt.subplots(n_rows, n_cols, figsize=(n_cols, n_rows))
    axes1 = axes1.flatten()
    for i in range(C):
        mi, ma = float(fmap[i].min()), float(fmap[i].max())
        axes1[i].imshow(fmap[i], cmap='gray', vmin=mi, vmax=ma)
        axes1[i].axis('off')
        axes1[i].text(0.95, 0.05, str(i), color='white', fontsize=8,
                      va='bottom', ha='right', transform=axes1[i].transAxes,
                      bbox=dict(fc='black', alpha=0.5))
    for j in range(C, len(axes1)):
        axes1[j].axis('off')
    plt.suptitle(f"Layer {layer_idx} Feature Maps (Gray){' with Mask' if layer_idx==0 and apply_mask else ''}")
    plt.tight_layout(rect=[0,0,1,0.95]); plt.show(block=False)

    # Heatmaps
    fig2, axes2 = plt.subplots(n_rows, n_cols, figsize=(n_cols, n_rows))
    axes2 = axes2.flatten()
    for i in range(C):
        mi, ma = float(fmap[i].min()), float(fmap[i].max())
        axes2[i].imshow(fmap[i], cmap='jet', vmin=mi, vmax=ma)
        axes2[i].axis('off')
        axes2[i].text(0.95, 0.05, str(i), color='white', fontsize=8,
                      va='bottom', ha='right', transform=axes2[i].transAxes,
                      bbox=dict(fc='black', alpha=0.5))
    for j in range(C, len(axes2)):
        axes2[j].axis('off')
    cbar_ax = fig2.add_axes([0.92, 0.15, 0.02, 0.7])
    mappable = plt.cm.ScalarMappable(cmap='jet')
    mappable.set_array(fmap.numpy())
    fig2.colorbar(mappable, cax=cbar_ax)
    plt.suptitle(f"Layer {layer_idx} Feature Maps (Heat){' with Mask' if layer_idx==0 and apply_mask else ''}")
    plt.tight_layout(rect=[0,0,0.9,0.95]); plt.show(block=False)

    # Strength maps
    strength = np.abs(fmap)
    fig3, axes3 = plt.subplots(n_rows, n_cols, figsize=(n_cols, n_rows))
    axes3 = axes3.flatten()
    norm3 = plt.Normalize(vmin=strength.min(), vmax=strength.max())
    for i in range(C):
        axes3[i].imshow(strength[i], cmap='hot', norm=norm3)
        axes3[i].axis('off')
        axes3[i].text(0.95, 0.05, str(i), color='white', fontsize=8,
                      va='bottom', ha='right', transform=axes3[i].transAxes,
                      bbox=dict(fc='black', alpha=0.5))
    for j in range(C, len(axes3)):
        axes3[j].axis('off')
    cbar_ax3 = fig3.add_axes([0.92, 0.15, 0.02, 0.7])
    m3 = plt.cm.ScalarMappable(cmap='hot', norm=norm3)
    m3.set_array(strength)
    fig3.colorbar(m3, cax=cbar_ax3)
    plt.suptitle(f"Layer {layer_idx} Feature Strength (|activation|){' with Mask' if layer_idx==0 and apply_mask else ''}")
    plt.tight_layout(rect=[0,0,0.9,0.95]); plt.show(block=False)

    # Mean activation bar
    mean_strength = strength.reshape(C, -1).mean(axis=1)
    fig4, ax4 = plt.subplots(figsize=(10, 3))
    bars = ax4.bar(
        range(C), mean_strength,
        color=plt.cm.hot((mean_strength - mean_strength.min()) / (mean_strength.max() - mean_strength.min() + 1e-8)),
        edgecolor='black', linewidth=0.5
    )
    ax4.set_title('Mean |activation| per channel')
    ax4.set_xlabel('Channel Index'); ax4.set_ylabel('Mean Absolute Activation')
    ax4.set_ylim(0, mean_strength.max() * 1.1)
    for bar in bars:
        h = bar.get_height()
        ax4.text(
            bar.get_x() + bar.get_width()/2, h + 0.005,
            f'{h:.3f}', ha='center', va='bottom', fontsize=6
        )
    plt.tight_layout(); plt.show(block=False)

    input("Press Enter to close all windows...")

# --------------------------
# Usage example
# --------------------------
if __name__ == '__main__':
    n_group = 'h_c_l'
    cha = 11
    img_path  = rf"E:\Project_SNV\0S\6_patch\l\000_x100_y50.png"
    ckpt_path = rf'E:\Project_SNV\2N\c2_cache\{n_group}\{cha}\best_model_{cha}.pth'
    cache_pt  = rf'E:\Project_SNV\1N\1_Pack\{n_group}\data_cache.pt'
    in_shape  = (50,50)
    layer_idx = 1
    n_cols    = 4
    apply_mask= False

    # Load model
    state = torch.load(ckpt_path, map_location='cpu')
    linear_keys = [k for k,v in state.items() if k.endswith('weight') and v.dim()==2]
    last_key = sorted(linear_keys)[-1]
    n_classes = state[last_key].shape[0]

    model = CNN(in_shape=in_shape, n_classes=n_classes, keep_channels=[15])
    model.load_state_dict(state)
    model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    # Visualize with cache lookup
    visualize_feature_and_heatmaps(
        model,
        img_path,
        layer_idx=layer_idx,
        in_shape=in_shape,
        device=model.conv0.weight.device,
        n_cols=n_cols,
        apply_mask=apply_mask,
        cache_pt=cache_pt
    )
