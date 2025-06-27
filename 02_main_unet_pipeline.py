#=============================================================================
# Solar Farm Semantic Segmentation with U-Net - Reed Johnson - UNET Pipeline
# -----------------------------------------------------------------------------
#Runpod Config: Runpod Pytorch 2.1
#20 GB Disk
#100 GB Network Volume
#Volume Path: /workspace
#Volume: RJ_Practicum
#1 x RTX A5000
#12 vCPU 25 GB RAM
#runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04
#
# -------------------- Load Libraries -------------------- 
import os
import time
import numpy as np
import random
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset 
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# -------------------- Set Random Seed --------------------
#Ensures reproducibility 
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# -------------------- Mask Processing --------------------
# Load binary mask and convert to torch tensor
# Opens mask, converts to grayscale (8-bit)
# Converts to NumPy array
# Converts 255 value pixels to 1.0, all others are 0.0 (floats)
# Convert to PyTorch tensor [1, H, W]
def load_mask_tensor(path):
    mask = Image.open(path).convert("L")
    mask_array = np.array(mask)
    binary_mask = (mask_array == 255).astype("float32")
    return torch.from_numpy(binary_mask).unsqueeze(0)

# -------------------- Dataset Definition --------------------
class SolarFarmDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
         # Paths to image and mask directories, creates a list of all image filenames, declares optional transform (future).  
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_ids = sorted([f for f in os.listdir(image_dir) if f.endswith(".png")])
        self.transform = transform

    def __len__(self):
        #Returns total number of samples
        return len(self.image_ids)

    def __getitem__(self, idx):
        #Pull image filename, build paths to images and masks
        image_id = self.image_ids[idx]
        image_path = os.path.join(self.image_dir, image_id)
        mask_path = os.path.join(self.mask_dir, image_id.replace("tile_", "mask_"))

        #Open RGB images, convert to numpy array and normalize pixel values, convert to pytorch tensor and reorder axes. 
        image = Image.open(image_path).convert("RGB")
        image = np.array(image).astype("float32") / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)

        #Open mask (8-bit), convert to numpy array, convert to binary (255 = 1), and convert to pytorch tensor. 
        mask = Image.open(mask_path).convert("L")
        mask = np.array(mask)
        mask = (mask == 255).astype("float32")
        mask = torch.from_numpy(mask).unsqueeze(0)

        #Place holder for possible future transforms.  
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        #Return image and corresponding mask. 
        return image, mask

# -------------------- U-Net Definition --------------------
# Double convolution block used repeatedly in U-Net
# Each block has 2 conv layers + batch norm + ReLU + dropout
# Added dropout = 0.1
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout =0.1):
        super().__init__()
        self.double_conv = nn.Sequential(
            # First conv layer
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            # Second conv layer
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)

# U-Net architecture
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        
        # Encoder (downsampling path)
        self.down1 = DoubleConv(in_channels, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)
        self.down4 = DoubleConv(256, 512)

        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)
        
        # Decoder (upsampling path)
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(128, 64)

        # Final 1x1 convolution to reduce channel dimension to 1
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        d1 = self.down1(x)
        d2 = self.down2(F.max_pool2d(d1, 2))
        d3 = self.down3(F.max_pool2d(d2, 2))
        d4 = self.down4(F.max_pool2d(d3, 2))
        
        # Bottleneck
        bn = self.bottleneck(F.max_pool2d(d4, 2))

        # Decoder with skip connections
        u4 = self.up4(bn)
        u4 = self.conv4(torch.cat([u4, d4], dim=1))
        u3 = self.up3(u4)
        u3 = self.conv3(torch.cat([u3, d3], dim=1))
        u2 = self.up2(u3)
        u2 = self.conv2(torch.cat([u2, d2], dim=1))
        u1 = self.up1(u2)
        u1 = self.conv1(torch.cat([u1, d1], dim=1))

        # Final output: raw logits
        return self.final_conv(u1)

# -------------------- Dice Loss Function --------------------
# Moved to diceloss from BCEWithLogitsLoss
# Dice is better for data imbalance 
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth #No dividing by zero

    #Applies sigmoid, flattens tensors
    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        #Calculate intersection and dice score, return dice loss.  
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        return 1 - dice

# ----------------------------------------------------------
# ------------------ Main Execution Block ------------------
# ----------------------------------------------------------

# Ensures this block runs only when the script is executed directly
if __name__ == "__main__":
    # Use CUDA/GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Paths in runpod
    image_dir = "/workspace/data_to_runpod/tiles_1024ft_v2"
    mask_dir = "/workspace/data_to_runpod/masks_1024ft_v2"

    #Initialize dataset
    dataset = SolarFarmDataset(image_dir=image_dir, mask_dir=mask_dir)

    
    # ---------------- Filter and Balance Dataset ----------------
    # Empty lists
    non_empty_indices = []
    empty_indices = []

    # Filter balanced subset
    print("Filtering dataset for balance...")
    #For loop searches for mask tiles that contain solar farms (mask > 0), append to lists
    for i in tqdm(range(len(dataset))):
        _, mask = dataset[i]
        if torch.any(mask > 0):
            non_empty_indices.append(i)
        else:
            empty_indices.append(i)

    # Limit the number of empty tiles to 3x the number of positive tiles
    desired_empty_count = min(len(empty_indices), 3 * len(non_empty_indices))
    balanced_indices = non_empty_indices + empty_indices[:desired_empty_count]
    balanced_dataset = Subset(dataset, balanced_indices)

    print(f"Balanced dataset size: {len(balanced_dataset)}")

    # Preview a few examples to confirm structure and mask labels
    num_preview = min(3, len(balanced_dataset))
    for i in range(num_preview):
        image, mask = balanced_dataset[i]
        print(f"{i}: image {image.shape}, mask {mask.shape}, unique mask values: {torch.unique(mask)}")

    # ---------------- Stratified Split into Train/Val/Test ----------------
    # Stratified split based on presence of solar farm pixels
    # Ensures train/val/test sets have similar class balance
    print("Performing stratified split...")
    
    # Assign binary labels for stratification
    balanced_indices = balanced_dataset.indices
    labels = [int(torch.any(dataset[i][1] > 0).item()) for i in balanced_indices]

    #Intial split, 80% train, 20% gets passed to next step. 
    train_idx, temp_idx, y_train, y_temp = train_test_split(
        balanced_indices, labels, test_size=0.2, stratify=labels, random_state=SEED
    )
    #Split remaining 20% into val and test
    val_idx, test_idx, _, _ = train_test_split(
        temp_idx, y_temp, test_size=0.5, stratify=y_temp, random_state=SEED
    )
    # Create final dataset subsets
    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)
    test_set = Subset(dataset, test_idx)

    #Defining process to count solar positive tiles
    def count_positives(subset):
        return sum(torch.any(dataset[idx][1] > 0).item() for idx in subset.indices)

    # Print split data size including positives 
    print(f"Train set size: {len(train_set)}, Positives: {count_positives(train_set)}")
    print(f"Val set size: {len(val_set)}, Positives: {count_positives(val_set)}")
    print(f"Test set size: {len(test_set)}, Positives: {count_positives(test_set)}")

    # ---------------- Create Data Loaders ----------------
    train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=8)
    test_loader = DataLoader(test_set, batch_size=8)


    # ------------------ Initialize Model ------------------
    model = UNet(in_channels=3, out_channels=1).to(device)
    criterion = DiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    best_val_loss = float('inf') # Keep track of best val loss
    early_stop_patience = 5 # number of epochs where training stops if no improvement
    early_stop_counter = 0

    # ------------------- Training Loop -------------------
    num_epochs = 25
    print("Starting training...")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        # Use tqdm trainaing bar
        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")
    
        for images, masks in train_loader_tqdm:
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
    
        # ---------------- Validation ----------------
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

        # Update learning rate scheduler
        scheduler.step(avg_val_loss)

        # -------------- Early Stopping Loop --------------
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), "/workspace/data_to_runpod/best_model.pth")
        else:
            early_stop_counter += 1
            print(f"Early Stop Counter: {early_stop_counter}/{early_stop_patience}")
            if early_stop_counter >= early_stop_patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

print(f"Training completed in {(time.time() - start_time)/60:.2f} minutes.")


# -------------------- Model Evaluation --------------------

print("Evaluating on test set...")
model.eval() # Enable evaluation mode
# Empty lists
ious, accs, dices = [], [], []

# Function to compute IOU, accuracy and Dice scores
def compute_metrics(preds, targets):
    # Flatten
    preds = preds.view(-1).cpu().numpy()
    targets = targets.view(-1).cpu().numpy()
    # IOU Calculation, avoids dividing by zero
    intersection = np.logical_and(preds, targets).sum()
    union = np.logical_or(preds, targets).sum()
    iou = intersection / union if union != 0 else 1.0

    #Accuracy calculation
    accuracy = (preds == targets).mean()
    
    #Dice calculation, also avoids dividing by zero
    dice = (2 * intersection) / (preds.sum() + targets.sum()) if (preds.sum() + targets.sum()) != 0 else 1.0

    return iou, accuracy, dice

# ---------------------- Save Outputs ----------------------
# Output Directory
os.makedirs("/workspace/outputs", exist_ok=True)

# Disable gradient computation for evaluation
with torch.no_grad():
    # Loop through the test dataset in batches
    for batch_idx, (images, masks) in enumerate(test_loader):
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward pass through model to get raw predictions
        outputs = model(images)
        preds = torch.sigmoid(outputs) > 0.8

        # Iterate over images and compute evaluation metrics
        for i in range(images.size(0)):
            iou, acc, dice = compute_metrics(preds[i], masks[i])
            ious.append(iou)
            accs.append(acc)
            dices.append(dice)

            # Find original index of this test image
            global_idx = test_set.indices[batch_idx * test_loader.batch_size + i]

            # Reconstruct original filename using global index
            image_filename = dataset.image_ids[global_idx]
            pred_filename = image_filename.replace(".png", "_pred.png")
            pred_path = os.path.join("/workspace/outputs", pred_filename)

            # Convert prediction mask to numpy format (0 or 255) and save as image
            pred_np = preds[i].squeeze().cpu().numpy().astype("uint8") * 255
            pred_img = Image.fromarray(pred_np)
            pred_img.save(pred_path)


# --------------------- Print Metrics ----------------------
print(f"Test IoU: {np.mean(ious):.4f}")
print(f"Test Accuracy: {np.mean(accs):.4f}")
print(f"Dice Coefficient: {np.mean(dices):.4f}")

# ----------------------- Save Model -----------------------
torch.save(model.state_dict(), "/workspace/data_to_runpod/unet_solarfarm.pth")

#Please see 03_save_doublets for visualization code and 04_acreage calculation for acreage calculation.  