import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import os
import albumentations as A # Data Argumentation
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import glob
from tqdm import tqdm
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt

# Dataset Definition
class DualChannelSegDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])[:, :, ::-1]  # BGR->RGB
        mask = np.load(self.mask_paths[idx])                   # Mask[idx].npy

        body = ((mask == 1) | (mask == 2)).astype(np.float32)  # 只要不是背景都为 mainbody
        defect = (mask == 2).astype(np.float32) * 0.9         # soft defect label
        multi_mask = np.stack([body, defect], axis=-1)

        if self.transform:
            augmented = self.transform(image=image, mask=multi_mask)
            image = augmented['image']
            multi_mask = augmented['mask']

        return image, multi_mask.permute(2, 0, 1)  # (C, H, W)

# 数据增强
def get_transforms():
    return A.Compose([
    A.HorizontalFlip(p=0.4),
    A.RandomRotate90(p=0.3),
    A.Affine(
        translate_percent=0.0625,
        scale=(0.9, 1.1),
        rotate=15,
        interpolation=0,      # cv2.INTER_NEAREST
        border_mode=0,        # cv2.BORDER_CONSTANT
        fill_value=0,
        mask_fill_value=0,
        p=0.7
    ),
    A.RandomBrightnessContrast(p=0.5),
    A.GaussNoise(p=0.2),
    A.Resize(height=364, width=388),
    A.Normalize(mean=(0.485,0.456,0.406),
                std=(0.229,0.224,0.225)),
    ToTensorV2()
])

# 加载路径，替换成你自己路径
image_paths = sorted(glob.glob(r'D:\Study\Postgraduate\S2\Project\Code\Resource\Original\1_2_3_4\1\*.tif')) # original image file
mask_paths = sorted(glob.glob(r'D:\Study\Postgraduate\S2\Project\Code\Resource\Semi labeled\1_2_3_4\1\*.npy')) # Semi label file/First mixed file

train_imgs, val_imgs, train_masks, val_masks = train_test_split(image_paths, mask_paths, test_size=0.2, random_state=42)

train_dataset = DualChannelSegDataset(train_imgs, train_masks, transform=get_transforms())
val_dataset = DualChannelSegDataset(val_imgs, val_masks, transform=get_transforms())

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# 模型初始化
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=2,
    activation=None
).cuda()

# 优化器和调度器
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25)
# scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

# ✅ 可选加载已有的上一阶段权重
checkpoint_path = r"D:\Study\Postgraduate\S2\Project\Code\Resource\semantic_cache\1_2_3_4\1\Binary\1checkpoint\2B_defect.pth"

load_weights = input(f"Would you want to keep going with the existed weight？[y/n]: ").strip().lower()

if load_weights == 'y':
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        best_val_iou = checkpoint.get("best_val_iou", 0)
        start_epoch = checkpoint.get("epoch", 24) + 1
        print(f"✅ 继续训练从 epoch {start_epoch} 开始")
    else:
        print("⚠️ 权重文件不存在，将从头开始")
        start_epoch = 1
else:
    print("🚀 从头开始训练")
    start_epoch = 1

# 损失函数定义
def dice_loss(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)
    dims = (2, 3) if pred.dim() == 4 else (1, 2)  # 根据维度调整
    intersection = (pred * target).sum(dim=dims)
    union = pred.sum(dim=dims) + target.sum(dim=dims)
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()

def total_loss_fn(preds, targets):
    body_pred = preds[:, 0, :, :]
    defect_pred = preds[:, 1, :, :]
    body_gt = targets[:, 0, :, :]
    defect_gt = targets[:, 1, :, :]

    # ✳️ 只使用 defect_gt 中置信度 >= 0.9 的像素
    defect_mask = (defect_gt >= 0).float()

    # 避免空 mask 导致 loss 为 NaN（全图都没 defect）
    if defect_mask.sum() < 1:
        defect_loss = torch.tensor(0.0, device=defect_gt.device)
    else:
        defect_pred_masked = defect_pred * defect_mask
        defect_gt_masked = defect_gt * defect_mask
        defect_loss = (
            dice_loss(defect_pred_masked, defect_gt_masked) +
            F.binary_cross_entropy_with_logits(defect_pred_masked, defect_gt_masked)
        )

    # ⚠️ body loss 不受影响
    body_loss = dice_loss(body_pred, body_gt) + F.binary_cross_entropy_with_logits(body_pred, body_gt)

    return 1.4 * defect_loss + 0.3 * body_loss

def dice_coef(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred) > 0.5
    target = target > 0.5
    intersection = (pred & target).float().sum(dim=(1, 2))
    union = pred.float().sum(dim=(1, 2)) + target.float().sum(dim=(1, 2))
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.mean().item()

def iou_coef(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred) > 0.5
    target = target > 0.5
    intersection = (pred & target).float().sum(dim=(1, 2))
    union = (pred | target).float().sum(dim=(1, 2))
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean().item()

# 训练与验证函数
def train_one_epoch(model, loader):
    model.train()
    running_loss = 0
    running_dice = 0
    running_iou = 0
    valid_batches = 0

    for img, mask in tqdm(loader):
        img, mask = img.cuda(), mask.cuda()
        
        # 只选取 defect 平均值大于 0.8 的样本
        if (mask[:, 1].sum() == 0):
            continue

        pred = model(img)
        loss = total_loss_fn(pred, mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        dice = dice_coef(pred[:, 1, :, :], mask[:, 1, :, :])
        iou = iou_coef(pred[:, 1, :, :], mask[:, 1, :, :])
        running_dice += dice
        running_iou += iou
        valid_batches += 1

    if valid_batches == 0:
        return 0.0

    print(f"Train - Loss: {running_loss/valid_batches:.4f}, Dice: {running_dice/valid_batches:.4f}, IoU: {running_iou/valid_batches:.4f}")
    return running_loss / valid_batches

@torch.no_grad()
def validate(model, loader):
    model.eval()
    total_loss, total_dice, total_iou = 0, 0, 0
    for img, mask in loader:
        img, mask = img.cuda(), mask.cuda()
        pred = model(img)
        loss = total_loss_fn(pred, mask)
        total_loss += loss.item()

        # ❗只计算 defect 通道的指标
        dice = dice_coef(pred[:, 1], mask[:, 1])
        iou = iou_coef(pred[:, 1], mask[:, 1])

        total_dice += dice
        total_iou += iou

    n = len(loader)
    return total_loss / n, total_dice / n, total_iou / n
    
best_val_dice = 0  # 初始化
history = {
    "epoch": [],
    "train_loss": [],
    "val_loss": [],
    "val_dice": [],
    "val_iou": []
}

for epoch in range(start_epoch, start_epoch + 40):
    train_loss = train_one_epoch(model, train_loader)
    val_loss, val_dice, val_iou = validate(model, val_loader)
    print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f}  Val Loss: {val_loss:.4f}  Dice: {val_dice:.4f}  IoU: {val_iou:.4f}")

    scheduler.step()

    # ✅ 记录历史
    history["epoch"].append(epoch)
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["val_dice"].append(val_dice)
    history["val_iou"].append(val_iou)

    # ✅ 保存模型
    if val_dice > best_val_dice:
        best_val_dice = val_dice
        save_path = r"D:\Study\Postgraduate\S2\Project\Code\Resource\semantic_cache\1_2_3_4\1\Binary\1checkpoint\2B_defect.pth"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_dice": best_val_dice,
        }, save_path)

        print(f"✅ Saved best model at epoch {epoch} with val dice {val_dice:.4f}")


plt.figure(figsize=(8, 5))
plt.plot(history["epoch"], history["train_loss"], label="Train Loss")
plt.plot(history["epoch"], history["val_loss"], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train & Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("dual_Binary_defect_1_loss.png", dpi=300)
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(history["epoch"], history["val_dice"], label="Val Dice")
plt.plot(history["epoch"], history["val_iou"], label="Val IoU")
plt.xlabel("Epoch")
plt.ylabel("Metric")
plt.title("Validation Dice & IoU")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("dual_Binary_defect_1_dice_iou.png", dpi=300)
plt.show()
