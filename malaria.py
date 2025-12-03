import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import random

# ==========================================
# 1. NETWORK & ATTENTION MODULES
# ==========================================

class BayesianChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=4):
        super(BayesianChannelAttention, self).__init__()
        reduced_channels = max(in_channels // reduction_ratio, 4)
        
        self.fc_shared = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True)
        )
        self.fc_mu = nn.Conv2d(reduced_channels, in_channels, kernel_size=1, bias=False)
        self.fc_var = nn.Conv2d(reduced_channels, in_channels, kernel_size=1, bias=False)

    def forward(self, x):
        avg_pool = F.adaptive_avg_pool2d(x, 1)
        shared_out = self.fc_shared(avg_pool)
        
        mu = self.fc_mu(shared_out)
        log_var = self.fc_var(shared_out)
        std = torch.exp(0.5 * log_var)
        
        if self.training:
            epsilon = torch.randn_like(std)
            z = mu + std * epsilon
        else:
            z = mu 
            
        att_weights = torch.sigmoid(z)
        out_features = x + (x * att_weights)
        variance = torch.exp(log_var)
        
        return out_features, variance, mu, log_var

# class UncertaintyGuidedPixelAttention(nn.Module):
#     def __init__(self, in_channels, beta=10):
#         super(UncertaintyGuidedPixelAttention, self).__init__()
#         self.beta = beta
#         self.conv1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv2d(in_channels // 2, 1, kernel_size=1)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x, variance):
#         w_sigma = torch.exp(-self.beta * variance)
#         features_tilde = x * w_sigma
        
#         pa = self.conv1(features_tilde)
#         pa = self.relu(pa)
#         pa = self.conv2(pa)
#         att_map = self.sigmoid(pa)
        
#         output = features_tilde + (features_tilde * att_map)
#         return output

class UncertaintyGuidedPixelAttention(nn.Module):
    """
    Implements Section 3.6: Uncertainty-Guided Pixel Attention.
    Uses variance from BCA to filter features before spatial attention.
    Now returns the attention map for visualization.
    """
    def __init__(self, in_channels, beta=10):
        super(UncertaintyGuidedPixelAttention, self).__init__()
        self.beta = beta
        
        # Standard Pixel Attention layers (Conv 1x1 based)
        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels // 2, 1, kernel_size=1) # Output 1 spatial map
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, variance):
        # 1. Calculate Reliability Weights (Eq 13)
        # w_sigma = exp(-beta * sigma^2)
        w_sigma = torch.exp(-self.beta * variance)
        
        # 2. Filter unreliable channels (Eq 15 - first part)
        # Features with high variance get suppressed here
        features_tilde = x * w_sigma
        
        # 3. Compute Pixel Attention Map on reliable features (Eq 14)
        pa = self.conv1(features_tilde)
        pa = self.relu(pa)
        pa = self.conv2(pa)
        att_map = self.sigmoid(pa) # Shape: [B, 1, 5, 5]
        
        # 4. Apply Pixel Attention (Eq 15 - second part)
        output = features_tilde + (features_tilde * att_map)
        
        # Return both the processed features AND the heatmap
        return output, att_map

# class MalariaClassificationNet(nn.Module):
#     def __init__(self, num_classes=2):
#         super(MalariaClassificationNet, self).__init__()
        
#         # Feature Extraction
#         self.block1 = nn.Sequential(
#             nn.Conv2d(3, 16, kernel_size=3, padding=1),
#             nn.BatchNorm2d(16), nn.ReLU(inplace=True),
#             nn.Conv2d(16, 16, kernel_size=3, padding=1),
#             nn.BatchNorm2d(16), nn.ReLU(inplace=True),
#             nn.MaxPool2d(2) 
#         )
#         self.block2 = nn.Sequential(
#             nn.Conv2d(16, 32, kernel_size=3, padding=1),
#             nn.BatchNorm2d(32), nn.ReLU(inplace=True),
#             nn.Conv2d(32, 32, kernel_size=3, padding=1),
#             nn.BatchNorm2d(32), nn.ReLU(inplace=True),
#             nn.MaxPool2d(2)
#         )
#         self.block3 = nn.Sequential(
#             nn.Conv2d(32, 64, kernel_size=3, padding=1),
#             nn.BatchNorm2d(64), nn.ReLU(inplace=True),
#             nn.Conv2d(64, 64, kernel_size=3, padding=1),
#             nn.BatchNorm2d(64), nn.ReLU(inplace=True),
#             nn.Conv2d(64, 64, kernel_size=3, padding=1),
#             nn.BatchNorm2d(64), nn.ReLU(inplace=True),
#             nn.MaxPool2d(2)
#         )
        
#         self.bca = BayesianChannelAttention(in_channels=64)
#         self.ugpa = UncertaintyGuidedPixelAttention(in_channels=64, beta=10)
        
#         self.classifier = nn.Sequential(
#             nn.Linear(64 * 5 * 5, 512),
#             nn.ReLU(inplace=True),
#             nn.Dropout(p=0.5),
#             nn.Linear(512, 50),
#             nn.ReLU(inplace=True),
#             nn.Dropout(p=0.5),
#             nn.Linear(50, num_classes)
#         )

#     def forward(self, x):
#         f1 = self.block1(x)
#         f2 = self.block2(f1)
#         features = self.block3(f2)
        
#         bca_features, variance, mu, log_var = self.bca(features)
#         final_features = self.ugpa(bca_features, variance)
        
#         flat = final_features.view(final_features.size(0), -1)
#         logits = self.classifier(flat)
        
#         return logits, mu, log_var

class MalariaClassificationNet(nn.Module):
    def __init__(self, num_classes=2):
        super(MalariaClassificationNet, self).__init__()
        
        # --- Feature Extraction (Fig 3) ---
        # Block 1: 44x44 -> 22x22
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.MaxPool2d(2) 
        )
        
        # Block 2: 22x22 -> 11x11
        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        
        # Block 3: 11x11 -> 5x5
        self.block3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        
        # --- Attention Mechanisms ---
        self.bca = BayesianChannelAttention(in_channels=64)
        self.ugpa = UncertaintyGuidedPixelAttention(in_channels=64, beta=10)
        
        # --- Classification Head ---
        self.classifier = nn.Sequential(
            nn.Linear(64 * 5 * 5, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            
            nn.Linear(512, 50),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            
            nn.Linear(50, num_classes)
        )

    def forward(self, x):
        # 1. Feature Extraction
        f1 = self.block1(x)
        f2 = self.block2(f1)
        features = self.block3(f2) # Shape: [B, 64, 5, 5]
        
        # 2. Bayesian Channel Attention
        bca_features, variance, mu, log_var = self.bca(features)
        
        # 3. Uncertainty-Guided Pixel Attention
        # Now capturing the attention map 'att_map'
        final_features, att_map = self.ugpa(bca_features, variance)
        
        # 4. Classification
        flat = final_features.view(final_features.size(0), -1)
        logits = self.classifier(flat)
        
        # Return logits, uncertainty params, AND the attention map
        return logits, mu, log_var, att_map



# reference implementation of ELBO loss.
def elbo_loss(logits, targets, mu, log_var, kl_weight=0.0001):
    ce_loss = F.cross_entropy(logits, targets)
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    kl_loss = kl_loss / logits.size(0)
    return ce_loss + (kl_weight * kl_loss)

# Our "weighted" elbo.

def curriculum_elbo_loss(logits, targets, mu, log_var, current_epoch, max_epochs, kl_weight=0.0001):
    """
    NOVELTY: Dynamic Uncertainty-Weighted Loss.
    
    Logic:
    1. Calculate individual loss for every image in the batch.
    2. Calculate uncertainty (variance) for every image.
    3. Generate 'Reliability Weights': High Uncertainty -> Low Weight.
    4. Anneal: Start with strict weighting, slowly fade to standard uniform weighting.
    """
    # 1. Standard Losses (reduction='none' keeps the loss individual per image)
    ce_loss = F.cross_entropy(logits, targets, reduction='none')
    
    # KL Divergence per image
    # Sum over channel dimensions [1,2,3] to get one scalar per image
    kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=[1, 2, 3])
    
    # Total raw loss per image
    raw_loss = ce_loss + (kl_weight * kl_div)
    
    # 2. Calculate Uncertainty Score per image
    # We use the mean variance across channels as the uncertainty proxy
    # Shape: [Batch_Size]
    variance = torch.exp(log_var).mean(dim=[1, 2, 3])
    
    # 3. Calculate Annealing Factor (Lambda)
    # Starts at 1.0 (Strict Curriculum), drops linearly to 0.0 (Standard Training)
    # We stop the curriculum halfway through (e.g., epoch 25 of 50) to let it converge normally
    lambda_factor = max(0, 1.0 - (current_epoch / (max_epochs * 0.5)))
    
    # 4. Calculate Dynamic Weights
    # weight = (1 / variance)^lambda
    # If variance is high, weight is low.
    # We detach() variance because we don't want to backpropagate through the weighting mechanism itself
    inverse_uncertainty = 1.0 / (variance.detach() + 1e-6) # Add epsilon for stability
    dynamic_weights = torch.pow(inverse_uncertainty, lambda_factor)
    
    # Normalize weights so they sum to Batch Size (maintains loss scale)
    dynamic_weights = dynamic_weights / dynamic_weights.mean()
    
    # 5. Apply Weights
    weighted_loss = (raw_loss * dynamic_weights).mean()
    
    return weighted_loss



# ==========================================
# 2. DATASET (Synthetic for Testing)
# ==========================================
# class SyntheticMalariaDataset(Dataset):
#     def __init__(self, num_samples=500):
#         self.num_samples = num_samples
#         self.data = []
#         self.targets = []
        
#         for _ in range(num_samples):
#             # 1. Base Noise
#             img = torch.randn(3, 44, 44)
            
#             # 2. Label
#             label = np.random.randint(0, 2)
            
#             # 3. Inject Signal: A "Parasite" Blob
#             # We add a bright spot in the center for Label 1.
#             # BatchNorm can't remove this because it varies spatially.
#             if label == 1:
#                 # Add intensity to a 10x10 patch in the middle
#                 img[0, 17:27, 17:27] += 5.0 
            
#             self.data.append(img)
#             self.targets.append(label)

#         self.data = torch.stack(self.data)
#         self.targets = torch.tensor(self.targets)

#     def __len__(self):
#         return self.num_samples

#     def __getitem__(self, idx):
#         return self.data[idx], self.targets[idx], idx

# class SyntheticMalariaDataset(Dataset):
#     def __init__(self, num_samples=500):
#         self.num_samples = num_samples
#         self.data = []
#         self.targets = []
#         self.size = 44
        
#         # Pre-compute Circular Mask
#         Y, X = torch.meshgrid(torch.arange(self.size), torch.arange(self.size))
#         center = self.size / 2 - 0.5
#         dist_from_center = torch.sqrt((X - center)**2 + (Y - center)**2)
#         self.mask = (dist_from_center <= 22).float().unsqueeze(0)

#         for _ in range(num_samples):
#             # 1. BASE TEXTURE (Cloudy Pink)
#             cloud_noise = torch.randn(1, 3, 11, 11)
#             cloud_noise = F.interpolate(cloud_noise, size=(self.size, self.size), mode='bilinear', align_corners=False)
#             cloud_noise = cloud_noise.squeeze(0)
            
#             # Base color: Light Purple
#             base_color = torch.tensor([0.8, 0.6, 0.8]).view(3, 1, 1)
#             img = base_color + (0.1 * cloud_noise)
            
#             label = np.random.randint(0, 2)
            
#             Y_grid, X_grid = torch.meshgrid(torch.arange(self.size), torch.arange(self.size))

#             if label == 1:
#                 # === PARASITE GENERATION ===
#                 # Logic: Small, Sharp, Dark, Variable Size
                
#                 cx = np.random.randint(14, 30)
#                 cy = np.random.randint(14, 30)
                
#                 # Randomize Size (Sigma) - matches paper variability
#                 sigma_x = np.random.uniform(1.5, 4.0) 
#                 sigma_y = np.random.uniform(1.5, 4.0) # Different sigmas = Oval shape
                
#                 # Calculate Gaussian Blob
#                 # exponent = - ((x-cx)^2 / 2sx^2 + (y-cy)^2 / 2sy^2)
#                 exponent = -(((X_grid - cx)**2) / (2 * sigma_x**2) + ((Y_grid - cy)**2) / (2 * sigma_y**2))
#                 spot = torch.exp(exponent)
                
#                 # Randomize Intensity (how much stain it absorbed)
#                 intensity = np.random.uniform(0.4, 0.8)
                
#                 # Subtract color (Dark Purple = remove Green/Red/Blue)
#                 img[0] -= (intensity * 0.6) * spot # Red
#                 img[1] -= (intensity * 0.9) * spot # Green (Main component of purple darkness)
#                 img[2] -= (intensity * 0.6) * spot # Blue
                
#             else:
#                 # === NON-PARASITE ===
#                 # Logic: Large, Faint, Diffuse (Artifacts)
                
#                 # 50% chance to have a "distractor" blob (WBC or stain artifact)
#                 if np.random.rand() > 0.5:
#                     cx = np.random.randint(10, 34)
#                     cy = np.random.randint(10, 34)
                    
#                     # Much larger size
#                     sigma = np.random.uniform(5.0, 10.0)
                    
#                     dist_sq = (X_grid - cx)**2 + (Y_grid - cy)**2
#                     blob = torch.exp(-0.5 * dist_sq / (sigma**2))
                    
#                     # Very faint intensity
#                     img -= 0.15 * blob

#             # Apply Mask & Clamp
#             img = img * self.mask
#             img = torch.clamp(img, 0, 1)
            
#             self.data.append(img)
#             self.targets.append(label)

#         self.data = torch.stack(self.data)
#         self.targets = torch.tensor(self.targets)

#     def __len__(self):
#         return self.num_samples

#     def __getitem__(self, idx):
#         return self.data[idx], self.targets[idx], idx


class SyntheticHardDataset(Dataset):
    def __init__(self, num_samples=1000):
        self.num_samples = num_samples
        self.data = []
        self.targets = []
        self.size = 44
        
        # Pre-compute Circular Mask
        Y, X = torch.meshgrid(torch.arange(self.size), torch.arange(self.size))
        center = self.size / 2 - 0.5
        dist_from_center = torch.sqrt((X - center)**2 + (Y - center)**2)
        self.mask = (dist_from_center <= 22).float().unsqueeze(0)

        for _ in range(num_samples):
            # Base Noise
            img = torch.randn(3, 44, 44) * 0.5 # Reduced base noise slightly
            label = np.random.randint(0, 2)
            
            # Helper to draw blob
            def draw_blob(img, intensity_r, intensity_g, intensity_b, sigma=3.0):
                cx = np.random.randint(15, 29)
                cy = np.random.randint(15, 29)
                Y_grid, X_grid = torch.meshgrid(torch.arange(44), torch.arange(44))
                dist_sq = (X_grid - cx)**2 + (Y_grid - cy)**2
                blob = torch.exp(-0.5 * dist_sq / (sigma**2))
                img[0] += intensity_r * blob
                img[1] += intensity_g * blob
                img[2] += intensity_b * blob
                return img

            if label == 1:
                # TRUE PARASITE: Purple Blob (High Red, High Blue, Low Green)
                # This is the signal we want to learn.
                img = draw_blob(img, 3.0, -1.0, 3.0) 
                
            else:
                # 50% chance of "Hard Negative" (Artifact)
                if np.random.rand() > 0.5:
                    # ARTIFACT: Red/Pink Blob (High Red, Normal Green, Low Blue)
                    # Looks similar to parasite but wrong color balance.
                    # Standard model will confuse this with Label 1.
                    img = draw_blob(img, 3.0, 0.5, 0.5) 
                else:
                    # Easy Negative (Empty)
                    pass

            # Apply Mask & Clamp
            img = img * self.mask
            img = torch.clamp(img, -3, 3) # Keep within reasonable bounds
            
            self.data.append(img)
            self.targets.append(label)

        self.data = torch.stack(self.data)
        self.targets = torch.tensor(self.targets)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx], idx

# ==========================================
# 3. TRAINING & ANALYSIS
# ==========================================
def train_model(model, train_loader, device, num_epochs=10, method='standard'):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    kl_weight = 0.0
    
    print(f"Starting training with method: {method.upper()}")
    model.train()
    
    # Store history for plotting later
    loss_history = []
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Anneal KL weight (Standard Bayesian Practice)
        kl_weight = min(0.01, kl_weight + 0.002) 

        for inputs, labels, _ in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Unpack 4 values (ignoring attention map)
            logits, mu, log_var, _ = model(inputs)
            
            # --- SWITCH: Standard vs Curriculum ---
            if method == 'curriculum':
                loss = curriculum_elbo_loss(
                    logits, labels, mu, log_var, 
                    current_epoch=epoch, 
                    max_epochs=num_epochs, 
                    kl_weight=kl_weight
                )
            else:
                # Standard: Just average the loss normally
                loss = elbo_loss(logits, labels, mu, log_var, kl_weight=kl_weight)
            # --------------------------------------
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        avg_loss = running_loss / len(train_loader)
        acc = 100 * correct / total
        loss_history.append(avg_loss)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] Method: {method} | Loss: {avg_loss:.4f} | Acc: {acc:.2f}%")
        
    return model, loss_history


# 
# Visualisation
#

def visualize_samples(dataset, num_samples=5):
    """
    Plots a grid: Top row = Non-Parasites (0), Bottom row = Parasites (1)
    """
    # containers
    zeros = []
    ones = []
    
    # Randomly search dataset until we have enough of both
    # (Inefficient for huge datasets, but fine for MPhil scale)
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    
    for idx in indices:
        img, label, _ = dataset[idx]
        if label == 0 and len(zeros) < num_samples:
            zeros.append(img)
        elif label == 1 and len(ones) < num_samples:
            ones.append(img)
        
        if len(zeros) == num_samples and len(ones) == num_samples:
            break
    
    # Plotting
    fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))
    fig.suptitle("Random Training Samples (Top: Healthy, Bottom: Parasite)", fontsize=16)
    
    for i in range(num_samples):
        # Class 0
        ax = axes[0, i]
        img_0 = zeros[i].permute(1, 2, 0).cpu().numpy() # CHW -> HWC
        # Normalize for display if needed (assuming 0-1 range here)
        img_0 = (img_0 - img_0.min()) / (img_0.max() - img_0.min())
        ax.imshow(img_0)
        ax.axis('off')
        if i == 0: ax.set_ylabel("Non-Parasite", fontsize=14)

        # Class 1
        ax = axes[1, i]
        img_1 = ones[i].permute(1, 2, 0).cpu().numpy()
        img_1 = (img_1 - img_1.min()) / (img_1.max() - img_1.min())
        ax.imshow(img_1)
        ax.axis('off')
        if i == 0: ax.set_ylabel("Parasite", fontsize=14)
        
    plt.tight_layout()
    plt.show()

def visualize_top_uncertain(dataset, uncertainty_records, num_show=5):
    """
    Plots the images that confused the model the most.
    """
    # Get top N records
    top_records = uncertainty_records[:num_show]
    
    fig, axes = plt.subplots(1, num_show, figsize=(15, 4))
    fig.suptitle(f"Top {num_show} Most Uncertain Images (High Variance)", fontsize=16)
    
    for i, record in enumerate(top_records):
        idx = record['index']
        unc = record['uncertainty_score']
        true_lbl = record['true_label']
        pred_lbl = record['predicted_label']
        
        # Retrieve image from dataset using index
        img, _, _ = dataset[idx]
        img = img.permute(1, 2, 0).cpu().numpy()
        img = (img - img.min()) / (img.max() - img.min())
        
        ax = axes[i]
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f"Uncertainty: {unc:.4f}\nTrue: {true_lbl} | Pred: {pred_lbl}", 
                     color=("green" if true_lbl==pred_lbl else "red"))
        
    plt.tight_layout()
    plt.show()


def analyze_uncertainty(model, val_loader, device):
    model.eval()
    uncertainty_records = []
    
    print("\n--- Starting Uncertainty Analysis ---")
    with torch.no_grad():
        for inputs, labels, indices in val_loader:
            inputs = inputs.to(device)
            logits, mu, log_var, _ = model(inputs)
            
            # Score = Mean Variance across channels
            variances = torch.exp(log_var)
            batch_uncertainty = variances.mean(dim=(1, 2, 3)) 
            
            for i in range(len(indices)):
                uncertainty_records.append({
                    "index": indices[i].item(),
                    "uncertainty_score": batch_uncertainty[i].item(),
                    "true_label": labels[i].item(),
                    "predicted_label": logits[i].argmax().item()
                })
    
    uncertainty_records.sort(key=lambda x: x["uncertainty_score"], reverse=True)
    
    print("\nTop 5 Most Uncertain Images:")
    print(f"{'Img Index':<10} | {'Uncertainty':<12} | {'True':<5} | {'Pred':<5}")
    print("-" * 45)
    for record in uncertainty_records[:5]:
        print(f"{record['index']:<10} | {record['uncertainty_score']:.4f}       | {record['true_label']:<5} | {record['predicted_label']:<5}")

    return uncertainty_records


def visualize_attention_maps(model, dataset, num_samples=5, device="cpu"):
    """
    Shows the original image next to the "Brain Scan" (Attention Map)
    """
    model.eval()
    
    # Get random parasites (Class 1) to see if it finds the dot
    indices = [i for i, (_, label, _) in enumerate(dataset) if label == 1]
    selected_indices = random.sample(indices, num_samples)
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(10, 3*num_samples))
    fig.suptitle("Explainability: Where is the model looking?", fontsize=16)
    
    with torch.no_grad():
        for i, idx in enumerate(selected_indices):
            img_tensor, label, _ = dataset[idx]
            input_batch = img_tensor.unsqueeze(0).to(device)
            
            # Forward pass
            logits, mu, log_var, att_map = model(input_batch)
            
            # 1. Original Image
            img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
            
            # 2. Attention Map processing
            # Resize 5x5 map to 44x44
            map_tensor = att_map[0, 0] # Take first item in batch, first channel
            # Upsample using bilinear interpolation
            map_resized = F.interpolate(att_map, size=(44, 44), mode='bilinear', align_corners=False)
            map_np = map_resized[0, 0].cpu().numpy()
            
            # Plot Original
            axes[i, 0].imshow(img_np)
            axes[i, 0].set_title("Original Parasite")
            axes[i, 0].axis('off')
            
            # Plot Heatmap
            axes[i, 1].imshow(map_np, cmap='jet')
            axes[i, 1].set_title("Attention Heatmap")
            axes[i, 1].axis('off')
            
            # Plot Overlay
            axes[i, 2].imshow(img_np)
            axes[i, 2].imshow(map_np, cmap='jet', alpha=0.5) # Alpha blends them
            axes[i, 2].set_title("Overlay")
            axes[i, 2].axis('off')
            
    plt.tight_layout()
    plt.show()





def profile_channel_uncertainty(model, loader, device):
    """
    Step 2: Pass data through and find which channels are 'noisy'.
    Returns: A list of (channel_index, average_uncertainty) sorted from High to Low.
    """
    model.eval()
    
    # Store accumulated variance for each of the 64 channels
    channel_uncertainties = torch.zeros(64).to(device)
    count = 0
    
    with torch.no_grad():
        for inputs, _, _ in loader:
            inputs = inputs.to(device)
            # We only care about log_var here
            _, _, log_var, _ = model(inputs)
            
            # log_var shape: [Batch, 64, 1, 1]
            # Convert to variance: exp(log_var)
            # Average over Batch (dim 0) -> Shape: [64, 1, 1]
            batch_vars = torch.exp(log_var).mean(dim=0).view(-1)
            
            channel_uncertainties += batch_vars
            count += 1
            
    # Average over all batches
    avg_uncertainties = channel_uncertainties / count
    
    # Sort: High Uncertainty first
    # argsort gives the indices
    sorted_indices = torch.argsort(avg_uncertainties, descending=True)
    
    return sorted_indices, avg_uncertainties

def prune_model(model, sorted_indices, prune_percent=0.5):
    """
    Step 3: 'Delete' the bad channels.
    We simulate deletion by setting weights to 0.
    """
    # Calculate how many to cut
    num_channels = 64
    num_to_prune = int(num_channels * prune_percent)
    
    # Get the indices of the "Bad" channels (Highest Uncertainty)
    indices_to_prune = sorted_indices[:num_to_prune]
    
    print(f"Pruning {num_to_prune} channels (Top {prune_percent*100}% most uncertain)...")
    
    # CORRECTED INDICES FOR BLOCK 3
    # 0: Conv, 1: BN, 2: ReLU
    # 3: Conv, 4: BN, 5: ReLU
    # 6: Conv (TARGET), 7: BN (TARGET), 8: ReLU, 9: MaxPool
    
    conv_layer = model.block3[6]  # The last Conv2d
    bn_layer = model.block3[7]    # The last BatchNorm
    
    with torch.no_grad():
        # 1. Zero out the CONVOLUTION filters
        for idx in indices_to_prune:
            # Conv weights are [Out, In, H, W]. We prune the Output filters.
            conv_layer.weight.data[idx, :, :, :] = 0.0
            if conv_layer.bias is not None:
                conv_layer.bias.data[idx] = 0.0
                
            # 2. Zero out the BATCH NORM stats
            # If we don't do this, BN might add a 'bias' term even if the input is 0
            bn_layer.weight.data[idx] = 0.0
            bn_layer.bias.data[idx] = 0.0
            bn_layer.running_mean[idx] = 0.0
            bn_layer.running_var[idx] = 1.0 # Set var to 1 to avoid div by zero
            
            # 3. Zero out the BAYESIAN ATTENTION Input connections
            # This ensures the uncertainty module 'knows' this channel is dead
            model.bca.fc_shared[0].weight.data[:, idx, :, :] = 0.0
            
    print("Pruning complete. Weights zeroed for Conv, BN, and Attention.")
    return model



def run_comparative_pruning(model, val_loader, device, prune_percent=0.5):
    """
    Runs 3 types of pruning and compares the damage.
    """
    print(f"\n--- STARTING COMPARATIVE STUDY (Pruning {prune_percent*100}%) ---")
    
    # 1. Get True Uncertainty Indices (YOUR METHOD)
    sorted_indices_unc, _ = profile_channel_uncertainty(model, val_loader, device)
    indices_unc = sorted_indices_unc[:int(64 * prune_percent)]
    
    # 2. Get Random Indices (BASELINE)
    all_indices = list(range(64))
    indices_rand = random.sample(all_indices, int(64 * prune_percent))
    indices_rand = torch.tensor(indices_rand).to(device)
    
    # 3. Define Evaluation Helper
    def evaluate_pruned_copy(indices, name):
        # Create a deep copy of the model so we don't destroy the original
        import copy
        model_copy = copy.deepcopy(model)
        
        # Prune
        model_copy = prune_model(model_copy, indices, prune_percent)
        
        # Eval
        correct = 0
        total = 0
        model_copy.eval()
        with torch.no_grad():
            for inputs, labels, _ in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                logits, _, _, _ = model_copy(inputs)
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        acc = 100 * correct / total
        print(f"[{name}] Zero-Shot Accuracy: {acc:.2f}%")
        return acc

    # --- RUN COMPARISONS ---
    acc_unc = evaluate_pruned_copy(indices_unc, "PROPOSED (Uncertainty)")
    acc_rand = evaluate_pruned_copy(indices_rand, "CONTROL (Random)")
    
    return acc_unc, acc_rand






# if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    
#     # 1. Setup Data
#     # Use Synthetic for now, swap to RealMalariaDataset for the thesis
#     #dataset = SyntheticMalariaDataset(num_samples=1000) 
#     dataset = SyntheticHardDataset(num_samples=1000) 
    
#     # Split
#     train_size = int(0.8 * len(dataset))
#     val_size = len(dataset) - train_size
#     train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
#     train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
#     # --- EXPERIMENT 1: BASELINE (Xiong et al.) ---
#     print("\n=== Running Baseline Experiment ===")
#     model_standard = MalariaClassificationNet().to(device)
#     model_standard, history_standard = train_model(model_standard, train_loader, device, num_epochs=10, method='standard')
    
#     # --- EXPERIMENT 2: YOUR CONTRIBUTION ---
#     print("\n=== Running Curriculum Experiment (Ours) ===")
#     model_curriculum = MalariaClassificationNet().to(device)
#     model_curriculum, history_curriculum = train_model(model_curriculum, train_loader, device, num_epochs=10, method='curriculum')

#     # --- PLOT COMPARISON ---
#     plt.figure(figsize=(10, 5))
#     plt.plot(history_standard, label='Standard (Baseline)', linestyle='--')
#     plt.plot(history_curriculum, label='Uncertainty Curriculum (Ours)', linewidth=2)
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.title('Training Convergence: Baseline vs. Proposed Curriculum')
#     plt.legend()
#     plt.grid(True)
#     plt.show()
    
#     # Analyze Final Uncertainty on the Curriculum Model
#     analyze_uncertainty(model_curriculum, val_loader, device)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    
    # 1. Data & Train Base Model (Same as before)
    dataset = SyntheticHardDataset(num_samples=1000) 
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    print("--- Training Base Model ---")
    base_model = MalariaClassificationNet().to(device)
    base_model, _ = train_model(base_model, train_loader, device, num_epochs=15, method='standard')
    
    # 2. The Sweep
    ratios = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    
    print(f"\n{'Ratio':<10} | {'Ours (Acc)':<12} | {'Random (Acc)':<12}")
    print("-" * 40)
    
    # Get uncertainty profile ONCE
    sorted_indices_unc, _ = profile_channel_uncertainty(base_model, val_loader, device)
    
    for ratio in ratios:
        # A. OUR METHOD
        import copy
        model_unc = copy.deepcopy(base_model)
        # Prune top N% uncertain
        indices_unc = sorted_indices_unc[:int(64 * ratio)]
        model_unc = prune_model(model_unc, indices_unc, ratio)
        
        # Eval Ours
        correct = 0
        total = 0
        model_unc.eval()
        with torch.no_grad():
            for inputs, labels, _ in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                logits, _, _, _ = model_unc(inputs)
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        acc_ours = 100 * correct / total
        
        # B. RANDOM CONTROL
        # Average of 3 random runs to be scientifically rigorous
        rand_accs = []
        for _ in range(3): 
            model_rand = copy.deepcopy(base_model)
            all_indices = list(range(64))
            indices_rand = random.sample(all_indices, int(64 * ratio))
            indices_rand = torch.tensor(indices_rand).to(device)
            
            model_rand = prune_model(model_rand, indices_rand, ratio)
            
            # Eval Random
            r_correct = 0
            r_total = 0
            model_rand.eval()
            with torch.no_grad():
                for inputs, labels, _ in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logits, _, _, _ = model_rand(inputs)
                    _, predicted = torch.max(logits.data, 1)
                    r_total += labels.size(0)
                    r_correct += (predicted == labels).sum().item()
            rand_accs.append(100 * r_correct / r_total)
        
        acc_rand_mean = sum(rand_accs) / len(rand_accs)
        
        print(f"{ratio*100:<10.0f}% | {acc_ours:<12.2f} | {acc_rand_mean:<12.2f}")