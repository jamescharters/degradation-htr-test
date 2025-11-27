import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.fft import fft2, ifft2
import random
import nibabel as nib
from skimage.transform import resize

# ============================================================================
# 1. DATA LOADING
# ============================================================================
def load_ixi_slice(filepath, slice_idx=None):
    try:
        nii_img = nib.load(filepath)
        data = nii_img.get_fdata()
        if slice_idx is None:
            slice_idx = data.shape[2] // 2
        img_slice = data[:, :, slice_idx]
        img_slice = np.rot90(img_slice) 
        img_slice = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min())
        img_slice = resize(img_slice, (256, 256), anti_aliasing=True)
        return torch.tensor(img_slice, dtype=torch.float32)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

# ============================================================================
# 2. MOTION SIMULATION
# ============================================================================
def create_motion_corrupted_image(clean_image, severity=5.0, motion_type='linear'):
    H, W = clean_image.shape
    num_lines = H
    t = torch.linspace(0, 1, num_lines)
    
    if motion_type == 'linear':
        rotation = severity * t
        translation = severity * t
    elif motion_type == 'sinusoidal':
        rotation = severity * torch.sin(np.pi * t)
        translation = severity * torch.sin(np.pi * t + np.pi/4)
    elif motion_type == 'exponential':
        rotation = severity * (1 - torch.exp(-5 * t))
        translation = severity * (1 - torch.exp(-4 * t))
    elif motion_type == 'quadratic':
        rotation = severity * (t ** 2)
        translation = severity * (t ** 2)
    else:
        raise ValueError(f"Unknown motion type: {motion_type}")
    
    kspace = fft2(clean_image)
    kspace_corrupted = kspace.clone()
    
    for line_idx in range(num_lines):
        ky = (line_idx - H//2) / H
        kx = (torch.arange(W, dtype=torch.float32) - W//2) / W
        phase = 2 * np.pi * (translation[line_idx] * ky * 0.05 + rotation[line_idx] * kx * 0.1)
        kspace_corrupted[line_idx, :] *= torch.exp(1j * phase)
    
    corrupted = ifft2(kspace_corrupted).real
    
    return corrupted, {
        'rotation': rotation,
        'translation': translation,
        'timestamps': t,
        'motion_type': motion_type
    }

# ============================================================================
# 3. MLP BASELINE MODEL (The Rival)
# ============================================================================

class MotionMLP(nn.Module):
    """
    Standard MLP to estimate motion curves.
    Replaces the KAN.
    """
    def __init__(self, input_dim=128, hidden_dim=64, output_points=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_points) # Direct discrete output
        )
    
    def forward(self, x, timestamps):
        # x: (Batch, 128) -> (Batch, 256)
        motion_curve = self.net(x)
        
        # Handle size mismatch via interpolation if necessary
        if motion_curve.shape[1] != len(timestamps):
            motion_curve = nn.functional.interpolate(
                motion_curve.unsqueeze(1), 
                size=len(timestamps), 
                mode='linear', 
                align_corners=True
            ).squeeze(1)
            
        return motion_curve

class MLPMotionModel(nn.Module):
    def __init__(self, image_size=256):
        super().__init__()
        self.image_size = image_size
        
        # --- MLP HEADS (Instead of KANs) ---
        self.rot_mlp = MotionMLP(input_dim=128, output_points=image_size)
        self.trans_mlp = MotionMLP(input_dim=128, output_points=image_size)
        
        # --- U-Net Encoder (Same as before) ---
        self.enc1 = self._conv_block(1, 32)
        self.enc2 = self._conv_block(32, 64)
        self.enc3 = self._conv_block(64, 128)
        self.pool = nn.MaxPool2d(2)
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # --- Motion Encoder & Decoder (Same as before) ---
        self.motion_map_encoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.bottleneck = self._conv_block(128 + 32, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec3 = self._conv_block(128, 64)
        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2 = self._conv_block(64, 32)
        self.final = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, 1),
            nn.Tanh()
        )

    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, corrupted_image, timestamps, return_motion_map=False):
        H, W = corrupted_image.shape
        M = len(timestamps)
        x = corrupted_image.unsqueeze(0).unsqueeze(0)
        
        x1 = self.enc1(x)           
        x2 = self.enc2(self.pool(x1))  
        x3 = self.enc3(self.pool(x2))  
        
        # MLP Prediction
        global_features = self.global_pool(x3).flatten(1)
        estimated_rotation = self.rot_mlp(global_features, timestamps).squeeze()
        estimated_translation = self.trans_mlp(global_features, timestamps).squeeze()
        
        motion_map = torch.zeros(H, W, 2, device=corrupted_image.device)
        for row_idx in range(H):
            line_idx = int((row_idx / H) * M)
            line_idx = min(line_idx, M - 1)
            motion_map[row_idx, :, 0] = estimated_rotation[line_idx]
            motion_map[row_idx, :, 1] = estimated_translation[line_idx]
            
        motion_flat = motion_map.reshape(-1, 2)
        motion_features = self.motion_map_encoder(motion_flat)
        motion_features = motion_features.reshape(H, W, 32).permute(2, 0, 1)
        
        motion_bottleneck = nn.functional.adaptive_avg_pool2d(
            motion_features.unsqueeze(0), (H // 4, W // 4)
        )
        
        x_fused = torch.cat([x3, motion_bottleneck], dim=1)
        x_bottle = self.bottleneck(x_fused)
        d = self.up3(x_bottle)
        d = torch.cat([d, x2], dim=1)
        d = self.dec3(d)
        d = self.up2(d)
        d = torch.cat([d, x1], dim=1)
        d = self.dec2(d)
        
        correction = self.final(d).squeeze()
        corrected = corrupted_image + 0.05 * correction
        
        output = {
            'rotation': estimated_rotation,
            'translation': estimated_translation
        }
        if return_motion_map:
            output['motion_map'] = motion_map
            output['correction_map'] = correction
            
        return corrected, output

# ============================================================================
# 4. UTILS & TRAINING
# ============================================================================
def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nMLP Model Parameters: {total_params:,}")
    return total_params

def compute_psnr(pred, target, max_val=1.0):
    mse = torch.mean((pred - target) ** 2)
    if mse < 1e-10: return torch.tensor(100.0)
    return 20 * torch.log10(torch.tensor(max_val) / torch.sqrt(mse))

def train_mlp_baseline(clean_image, num_epochs=500):
    print(f"Training MLP Baseline Model (The Competitor)...")
    model = MLPMotionModel(image_size=clean_image.shape[0])
    count_parameters(model)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    best_score = -float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        
        severity = 5.0 + (2.0 * epoch / num_epochs)
        motion_type = random.choice(['linear', 'sinusoidal', 'quadratic']) if epoch > 50 else 'linear'
        
        corrupted, true_motion = create_motion_corrupted_image(clean_image, severity, motion_type)
        corrected, est_motion = model(corrupted, true_motion['timestamps'])
        
        loss_img = torch.mean((corrected - clean_image) ** 2)
        loss_rot = torch.mean((est_motion['rotation'] - true_motion['rotation']) ** 2)
        loss_trans = torch.mean((est_motion['translation'] - true_motion['translation']) ** 2)
        
        loss = loss_img + 0.01 * (loss_rot + loss_trans)
        
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        with torch.no_grad():
            psnr = compute_psnr(corrected, clean_image).item()
            rot_mae = torch.mean(torch.abs(est_motion['rotation'] - true_motion['rotation'])).item()
            trans_mae = torch.mean(torch.abs(est_motion['translation'] - true_motion['translation'])).item()
            
            # Same checkpointing logic as KAN
            score = psnr - (rot_mae + trans_mae)
            if score > best_score:
                best_score = score
                torch.save(model.state_dict(), 'best_mlp_model.pth')
                
        if (epoch+1) % 50 == 0:
            print(f"Epoch {epoch+1} | PSNR: {psnr:.2f} | MAE: {(rot_mae+trans_mae)/2:.3f}")

    return model

def visualize_results(model, clean_image):
    # Important: Use same severity (10.0) as end of training for fair comparison
    corrupted, true_motion = create_motion_corrupted_image(clean_image, severity=10.0, motion_type='linear')
    
    model.eval()
    with torch.no_grad():
        corrected, est = model(corrupted, true_motion['timestamps'], return_motion_map=True)
        
    # Calculate Metrics
    rot_r2 = np.corrcoef(true_motion['rotation'], est['rotation'].numpy())[0,1]
    
    # Plot
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    
    # Images
    ax[0].imshow(corrected.numpy(), cmap='gray')
    ax[0].set_title("MLP Corrected Image")
    
    # Motion Curve (This is what you want to compare against KAN)
    t = true_motion['timestamps']
    ax[1].plot(t, true_motion['rotation'], 'b-', label='True', linewidth=3)
    ax[1].plot(t, est['rotation'].numpy(), 'r--', label='MLP Predicted', linewidth=2)
    ax[1].set_title(f"MLP Learned Rotation (R2={rot_r2:.3f})")
    ax[1].legend()
    
    plt.savefig('mlp_baseline_results.png')
    plt.show()
    print(f"MLP R2: {rot_r2:.3f}")

# ============================================================================
# MAIN
# ============================================================================
if __name__ == '__main__':
    filename = "IXI002-Guys-0828-T2.nii"
    print(f"Loading {filename}...")
    
    clean_image = load_ixi_slice(filename)
    
    if clean_image is not None:
        # Train
        train_mlp_baseline(clean_image, num_epochs=500)
        
        # Visualize Best
        model = MLPMotionModel(image_size=clean_image.shape[0])
        model.load_state_dict(torch.load('best_mlp_model.pth'))
        visualize_results(model, clean_image)
    else:
        print("Could not load image. Check filename.")