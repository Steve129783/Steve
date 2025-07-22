import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import os
import json
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import glob
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt
from pathlib import Path

# ---- Helper: Load LabelMe JSON mask for main body ----
def load_labelme_mask(json_path, height, width, label_name='main body'):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    mask = np.zeros((height, width), dtype=np.uint8)
    for shape in data.get('shapes', []):
        if shape['label'] != label_name:
            continue
        pts = np.array(shape['points'], dtype=np.int32)
        cv2.fillPoly(mask, [pts], color=1)
    return mask

# ---- Dataset: Binary main body vs background ----
class MainBodyDataset(Dataset):
    def __init__(self, image_paths, json_paths, transform=None):
        self.image_paths = image_paths
        self.json_paths = json_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_bgr = cv2.imread(self.image_paths[idx])
        image = img_bgr[:, :, ::-1]
        h, w = image.shape[:2]
        mask_int = load_labelme_mask(self.json_paths[idx], h, w)
        mask = mask_int.astype(np.float32)
        if self.transform:
            aug = self.transform(image=image, mask=mask)
            image, mask = aug['image'], aug['mask']
        return image, mask.unsqueeze(0)

# ---- Data augmentations ----
def get_transforms():
    return A.Compose([
        A.HorizontalFlip(p=0.4),
        A.RandomRotate90(p=0.3),
        A.Affine(translate_percent=0.0625, scale=(0.9,1.1), rotate=15,
                 interpolation=0, border_mode=0, fill_value=0, p=0.7),
        A.RandomBrightnessContrast(p=0.5),
        A.GaussNoise(p=0.2),
        A.Resize(height=364, width=388),
        A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ToTensorV2()
    ])

# ---- Prepare DataLoader ----
image_paths = sorted(glob.glob(r'D:\Study\Postgraduate\S2\Project\Code\Resource\Original\13_14_15_16\14\*.tif'))
json_paths  = sorted(glob.glob(r'E:\Project_SNV\0S\1_Full_Raw_MB\14\*.json'))
train_imgs, val_imgs, train_jsons, val_jsons = train_test_split(
    image_paths, json_paths, test_size=0.2, random_state=42
)
train_dataset = MainBodyDataset(train_imgs, train_jsons, transform=get_transforms())
val_dataset   = MainBodyDataset(val_imgs, val_jsons, transform=get_transforms())
train_loader  = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader    = DataLoader(val_dataset, batch_size=4, shuffle=False)

# ---- Model: single-channel output ----
model = smp.Unet(encoder_name='resnet34', encoder_weights='imagenet',
                 in_channels=3, classes=1, activation=None).cuda()

# ---- Loss & Metric ----
def dice_loss(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum(dim=(2,3))
    union = pred.sum(dim=(2,3)) + target.sum(dim=(2,3))
    dice = (2.*intersection + smooth)/(union + smooth)
    return 1 - dice.mean()

total_loss_fn = lambda pred, target: dice_loss(pred, target) + F.binary_cross_entropy_with_logits(pred, target)

def dice_coef(pred, target, smooth=1e-6):
    pred_bin = (torch.sigmoid(pred) > 0.5)
    target_bin = (target > 0.5)
    intersection = (pred_bin & target_bin).float().sum(dim=(2,3))
    union = pred_bin.float().sum(dim=(2,3)) + target_bin.float().sum(dim=(2,3))
    dice = (2.*intersection + smooth)/(union + smooth)
    return dice.mean().item()

# ---- Optimizer & Scheduler ----
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

# ---- Training & Validation functions ----
def train_one_epoch():
    model.train()
    running_loss, running_dice = 0.0, 0.0
    for imgs, masks in train_loader:
        imgs, masks = imgs.cuda(), masks.cuda()
        preds = model(imgs)
        loss = total_loss_fn(preds, masks)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        running_loss += loss.item()
        running_dice += dice_coef(preds, masks)
    return running_loss/len(train_loader), running_dice/len(train_loader)

@torch.no_grad()
def validate():
    model.eval()
    val_loss, val_dice = 0.0, 0.0
    for imgs, masks in val_loader:
        imgs, masks = imgs.cuda(), masks.cuda()
        preds = model(imgs)
        val_loss += total_loss_fn(preds, masks).item()
        val_dice += dice_coef(preds, masks)
    return val_loss/len(val_loader), val_dice/len(val_loader)

# ---- Main Loop with Early Stopping ----
best_dice = 0.0
epochs_no_improve = 0
patience = 2
history = {'epoch': [], 'train_loss': [], 'val_loss': [], 'val_dice': []}
for epoch in range(1, 21):
    train_loss, train_dice = train_one_epoch()
    val_loss, val_dice = validate()
    scheduler.step(val_loss)
    # record
    history['epoch'].append(epoch)
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['val_dice'].append(val_dice)
    # print summary
    print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val Dice={val_dice:.4f}")
    # check improvement
    if val_dice > best_dice:
        best_dice = val_dice
        epochs_no_improve = 0
        os.makedirs(r'E:\Project_SNV\0S\s1_cache', exist_ok=True)
        torch.save(model.state_dict(), r'E:\Project_SNV\0S\s1_cache\best_model.pth')
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping after {epoch} epochs without improvement.")
            break

# ---- Plot Curves ----
plt.figure(figsize=(8,5))
plt.plot(history['epoch'], history['train_loss'], label='Train Loss')
plt.plot(history['epoch'], history['val_loss'], label='Val Loss')
plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)
plt.savefig(r'E:\Project_SNV\0S\s1_cache\loss_curve.png', dpi=300); plt.show()

plt.figure(figsize=(8,5))
plt.plot(history['epoch'], history['val_dice'], label='Val Dice')
plt.xlabel('Epoch'); plt.ylabel('Dice'); plt.legend(); plt.grid(True)
plt.savefig(r'E:\Project_SNV\0S\s1_cache\dice_curve.png', dpi=300); plt.show()
