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

        body = (mask == 1).astype(np.float32) #
        defect = (mask == 2).astype(np.float32)

        multi_mask = np.stack([body, defect], axis=-1)          # (H, W, 2)

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
mask_paths = sorted(glob.glob(r'D:\Study\Postgraduate\S2\Project\Code\Resource\Semi labeled\1_2_3_4\1\*.npy')) # Semi label file

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

    body_loss = dice_loss(body_pred, body_gt) + F.binary_cross_entropy_with_logits(body_pred, body_gt)
    defect_loss = dice_loss(defect_pred, defect_gt) + F.binary_cross_entropy_with_logits(defect_pred, defect_gt)

    return 0.3 *defect_loss + body_loss

# 优化器和调度器
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

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
    for img, mask in tqdm(loader):
        img, mask = img.cuda(), mask.cuda()
        pred = model(img)
        loss = total_loss_fn(pred, mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # 计算整体两个通道的Dice和IoU平均（也可以单独算某个通道）
        dice = (dice_coef(pred[:, 0, :, :], mask[:, 0, :, :]) + dice_coef(pred[:, 1, :, :], mask[:, 1, :, :])) / 2
        iou = (iou_coef(pred[:, 0, :, :], mask[:, 0, :, :]) + iou_coef(pred[:, 1, :, :], mask[:, 1, :, :])) / 2
        running_dice += dice
        running_iou += iou

    n = len(loader)
    print(f"Train - Loss: {running_loss/n:.4f}, Dice: {running_dice/n:.4f}, IoU: {running_iou/n:.4f}")
    return running_loss / n

@torch.no_grad()
def validate(model, loader):
    model.eval()
    total_loss, total_dice, total_iou = 0, 0, 0
    for img, mask in loader:
        img, mask = img.cuda(), mask.cuda()
        pred = model(img)
        loss = total_loss_fn(pred, mask)
        total_loss += loss.item()

        dice = (dice_coef(pred[:, 0], mask[:, 0]) + dice_coef(pred[:, 1], mask[:, 1])) / 2
        iou = (iou_coef(pred[:, 0], mask[:, 0]) + iou_coef(pred[:, 1], mask[:, 1])) / 2

        total_dice += dice
        total_iou += iou

    n = len(loader)
    return total_loss / n, total_dice / n, total_iou / n

# 预测缺陷通道函数示例
def predict_mainbody_only(image_tensor):
    model.eval()
    with torch.no_grad():
        pred = model(image_tensor.unsqueeze(0).cuda())  # (1, 2, H, W)
        body_logits = pred[0, 0]
        body_prob = torch.sigmoid(body_logits)
        return (body_prob > 0.5).float().cpu().numpy()
    
best_val_dice = 0  # 初始化
history = {
    "epoch": [],
    "train_loss": [],
    "val_loss": [],
    "val_dice": [],
    "val_iou": []
}

for epoch in range(1, 20):
    train_loss = train_one_epoch(model, train_loader)
    val_loss, val_dice, val_iou = validate(model, val_loader)
    print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f}  Val Loss: {val_loss:.4f}  Dice: {val_dice:.4f}  IoU: {val_iou:.4f}")

    scheduler.step(val_loss)

    # ✅ 记录当前指标
    history["epoch"].append(epoch)
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["val_dice"].append(val_dice)
    history["val_iou"].append(val_iou)

    if val_dice > best_val_dice:
        best_val_dice = val_dice
        save_path = r"D:\Study\Postgraduate\S2\Project\Code\Resource\semantic_cache\1_2_3_4\1\Binary\mainbody\2B_mainbody.pth"
        torch.save(model.state_dict(), save_path)
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
plt.savefig("dual_Binary_main body_loss.png", dpi=300)
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
plt.savefig("dual_Binary_main body_dice_iou.png", dpi=300)
plt.show()