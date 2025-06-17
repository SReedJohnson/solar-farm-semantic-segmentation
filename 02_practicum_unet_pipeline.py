# Load Libraries
import os
import time
import numpy as np
import gc
import random
from PIL import Image
import rasterio
import geopandas as gpd

import torch
import torch.nn as nn
import torch.nn.functional as F
from rasterio.transform import from_origin
from rasterio.features import shapes
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from sklearn.metrics import jaccard_score
from tqdm import tqdm


#Setting seed

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


#Defining Mask Saving Functions

def save_mask_to_geotiff(pred_mask_tensor, reference_image_path, output_tif_path):
    pred_mask = pred_mask_tensor.squeeze().cpu().numpy().astype("uint8")

    with rasterio.open(reference_image_path) as src:
        profile = src.profile
        transform = src.transform
        crs = src.crs

    profile.update(dtype=rasterio.uint8, count=1)

    with rasterio.open(output_tif_path, 'w', **profile) as dst:
        dst.write(pred_mask, 1)

def mask_to_shapefile(tif_path, shp_path):
    with rasterio.open(tif_path) as src:
        mask = src.read(1)
        transform = src.transform
        results = (
            {"geometry": geom, "properties": {"value": val}}
            for geom, val in shapes(mask, mask=mask > 0, transform=transform)
        )

    gdf = gpd.GeoDataFrame.from_features(results, crs=src.crs)
    gdf.to_file(shp_path)

# Normalize Mask
def load_mask(path):
    mask = Image.open(path).convert("L")
    mask_array = np.array(mask)
    binary_mask = (mask_array == 255).astype("uint8")
    return binary_mask

# Convert to PyTorch Tensor
def load_mask_tensor(path):
    mask = Image.open(path).convert("L")
    mask_array = np.array(mask)
    binary_mask = (mask_array == 255).astype("float32")
    return torch.from_numpy(binary_mask).unsqueeze(0)

# Custom Dataset
class SolarFarmDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_ids = sorted([f for f in os.listdir(image_dir) if f.endswith(".png")])
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_path = os.path.join(self.image_dir, image_id)
        mask_path = os.path.join(self.mask_dir, image_id.replace("tile_", "mask_"))

        image = Image.open(image_path).convert("RGB")
        image = np.array(image).astype("float32") / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)

        mask = Image.open(mask_path).convert("L")
        mask = np.array(mask)
        mask = (mask == 255).astype("float32")
        mask = torch.from_numpy(mask).unsqueeze(0)

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask

# UNet Architecture
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.down1 = DoubleConv(in_channels, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)
        self.down4 = DoubleConv(256, 512)
        self.bottleneck = DoubleConv(512, 1024)
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(128, 64)
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(F.max_pool2d(d1, 2))
        d3 = self.down3(F.max_pool2d(d2, 2))
        d4 = self.down4(F.max_pool2d(d3, 2))
        bn = self.bottleneck(F.max_pool2d(d4, 2))
        u4 = self.up4(bn)
        u4 = self.conv4(torch.cat([u4, d4], dim=1))
        u3 = self.up3(u4)
        u3 = self.conv3(torch.cat([u3, d3], dim=1))
        u2 = self.up2(u3)
        u2 = self.conv2(torch.cat([u2, d2], dim=1))
        u1 = self.up1(u2)
        u1 = self.conv1(torch.cat([u1, d1], dim=1))
        return self.final_conv(u1)


# MAIN EXECUTION BLOCK

if __name__ == "__main__":
    # Use CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Paths in runpod
    image_dir = "/workspace/data_to_runpod/tiles_1024ft"
    mask_dir = "/workspace/data_to_runpod/masks_1024ft"

    dataset = SolarFarmDataset(image_dir=image_dir, mask_dir=mask_dir)

    # Filter balanced subset
    non_empty_indices = []
    empty_indices = []

    print("Filtering dataset for balance...")
    for i in range(len(dataset)):
        _, mask = dataset[i]
        if torch.any(mask > 0):
            non_empty_indices.append(i)
        else:
            empty_indices.append(i)

    desired_empty_count = min(len(empty_indices), 3 * len(non_empty_indices))
    balanced_indices = non_empty_indices + empty_indices[:desired_empty_count]
    balanced_dataset = Subset(dataset, balanced_indices)

    print(f"Balanced dataset size: {len(balanced_dataset)}")

    # Preview sample
    for i in range(3):
        image, mask = balanced_dataset[i]
        print(f"{i}: image {image.shape}, mask {mask.shape}, unique mask values: {torch.unique(mask)}")

    # Split dataset
    train_size = int(0.8 * len(balanced_dataset))
    val_size = int(0.1 * len(balanced_dataset))
    test_size = len(balanced_dataset) - train_size - val_size

    train_set, val_set, test_set = random_split(balanced_dataset, [train_size, val_size, test_size])
    train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=8)
    test_loader = DataLoader(test_set, batch_size=8)

    # Initialize model
    model = UNet(in_channels=3, out_channels=1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Training loop
    num_epochs = 25
    print("Starting training...")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
    
        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")
    
        for images, masks in train_loader_tqdm:  # <<-- fixed this line
            images = images.to(device)
            masks = masks.to(device)
    
            outputs = model(images)
            loss = criterion(outputs, masks)
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    
            train_loader_tqdm.set_postfix(loss=loss.item())
    
        avg_train_loss = total_loss / len(train_loader)
    
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{num_epochs} — Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

print(f"Training completed in {(time.time() - start_time)/60:.2f} minutes.")

#Model Evaluation

print("Evaluating on test set...")
model.eval()
ious, accs, dices = [], [], []

def compute_metrics(preds, targets):
    preds = preds.view(-1).cpu().numpy()
    targets = targets.view(-1).cpu().numpy()

    intersection = np.logical_and(preds, targets).sum()
    union = np.logical_or(preds, targets).sum()
    iou = intersection / union if union != 0 else 1.0

    accuracy = (preds == targets).mean()
    dice = (2 * intersection) / (preds.sum() + targets.sum()) if (preds.sum() + targets.sum()) != 0 else 1.0

    return iou, accuracy, dice


# Output Mask
os.makedirs("/workspace/outputs", exist_ok=True)

with torch.no_grad():
    for batch_idx, (images, masks) in enumerate(test_loader):
        images = images.to(device)
        masks = masks.to(device)
        outputs = model(images)
        preds = torch.sigmoid(outputs) > 0.5

        for i in range(images.size(0)):
            iou, acc, dice = compute_metrics(preds[i], masks[i])
            ious.append(iou)
            accs.append(acc)
            dices.append(dice)
        
            global_idx = test_set.indices[batch_idx * test_loader.batch_size + i]
            image_filename = dataset.image_ids[global_idx]
            ref_image_path = os.path.join(image_dir, image_filename)
        
            output_tif = f"/workspace/outputs/{image_filename.replace('.png', '.tif')}"
            output_shp = f"/workspace/outputs/{image_filename.replace('.png', '.shp')}"
        
            if torch.any(preds[i] > 0):
                save_mask_to_geotiff(preds[i], ref_image_path, output_tif)
                try:
                    mask_to_shapefile(output_tif, output_shp)
                except Exception as e:
                    print(f"Failed to write shapefile {output_shp}: {e}")
            else:
                print(f"Skipping empty prediction for {image_filename}")

print(f"Test IoU: {np.mean(ious):.4f}")
print(f"Test Accuracy: {np.mean(accs):.4f}")
print(f"Dice Coefficient: {np.mean(dices):.4f}")

# Save Model
torch.save(model.state_dict(), "/workspace/data_to_runpod/unet_solarfarm.pth")
