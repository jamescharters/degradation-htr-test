"""
KAN-based U-Net for Full-Image QSM Reconstruction
This uses KAN's adaptive basis functions in a convolutional architecture
to capture global dipole field structure.
"""

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from pathlib import Path

# ==========================================
# CONFIGURATION
# ==========================================
DATASET_PATH = Path("./data/OSF_QSM_Dataset")
SUBJECT_ID = "Subject1"
ORIENTATION = 1
EPOCHS = 200
BATCH_SIZE = 4  # Process image in overlapping tiles

torch.manual_seed(42)
np.random.seed(42)

# ==========================================
# KAN LAYER (Same as before)
# ==========================================

class KANLayer(nn.Module):
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3):
        super(KANLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        h = (1 / grid_size)
        grid = (torch.arange(-spline_order, grid_size + spline_order + 1) * h).expand(in_features, -1).contiguous()
        self.register_buffer("grid", grid)
        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = nn.Parameter(torch.Tensor(out_features, in_features, grid_size + spline_order))
        self.reset_parameters()
        self.base_activation = nn.SiLU()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=np.sqrt(5) * 0.1)
        nn.init.trunc_normal_(self.spline_weight, std=0.01)
    
    def b_splines(self, x):
        grid = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (x - grid[:, :-(k + 1)]) / (grid[:, k:-1] - grid[:, :-(k + 1)]) * bases[:, :, :-1] + \
                    (grid[:, k + 1:] - x) / (grid[:, k + 1:] - grid[:, 1:(-k)]) * bases[:, :, 1:]
        return bases.contiguous()
    
    def forward(self, x):
        base_output = F.linear(self.base_activation(x), self.base_weight)
        x_norm = (torch.tanh(x) + 1) / 2 
        spline_basis = self.b_splines(x_norm)
        spline_output = torch.einsum("bij,oij->bo", spline_basis, self.spline_weight)
        return base_output + spline_output

# ==========================================
# CONVOLUTIONAL KAN BLOCK
# ==========================================

class ConvKANBlock(nn.Module):
    """
    Convolutional block using KAN for the nonlinear activation
    Architecture: Conv -> KAN -> Conv
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, grid_size=5):
        super(ConvKANBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.norm1 = nn.InstanceNorm2d(out_channels)
        self.norm2 = nn.InstanceNorm2d(out_channels)
        
        # KAN acts on flattened spatial features
        self.use_kan = True
        self.kan = KANLayer(out_channels, out_channels, grid_size=grid_size)
    
    def forward(self, x):
        # First conv
        x = self.conv1(x)
        x = self.norm1(x)
        
        # Apply KAN channel-wise (treating channels as features)
        if self.use_kan:
            B, C, H, W = x.shape
            # Reshape to (B*H*W, C) for KAN
            x_flat = x.permute(0, 2, 3, 1).reshape(-1, C)
            x_flat = self.kan(x_flat)
            x = x_flat.reshape(B, H, W, C).permute(0, 3, 1, 2)
        else:
            x = F.relu(x)
        
        # Second conv
        x = self.conv2(x)
        x = self.norm2(x)
        
        return x


# ==========================================
# SIMPLE KAN-BASED U-NET
# ==========================================

class KANUNet(nn.Module):
    """
    Simple U-Net architecture using KAN blocks
    """
    def __init__(self, in_channels=1, out_channels=1, base_channels=16):
        super(KANUNet, self).__init__()
        
        # Encoder
        self.enc1 = ConvKANBlock(in_channels, base_channels, grid_size=5)
        self.enc2 = ConvKANBlock(base_channels, base_channels*2, grid_size=5)
        self.enc3 = ConvKANBlock(base_channels*2, base_channels*4, grid_size=5)
        
        # Bottleneck
        self.bottleneck = ConvKANBlock(base_channels*4, base_channels*4, grid_size=5)
        
        # Decoder
        self.dec3 = ConvKANBlock(base_channels*8, base_channels*2, grid_size=5)
        self.dec2 = ConvKANBlock(base_channels*4, base_channels, grid_size=5)
        self.dec1 = ConvKANBlock(base_channels*2, base_channels, grid_size=5)
        
        # Output
        self.out_conv = nn.Conv2d(base_channels, out_channels, 1)
        
        # Pooling and upsampling
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    
    # In the KANUNet class, replace the original forward method with this one:

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        
        x = self.pool(e1)
        e2 = self.enc2(x)
        
        x = self.pool(e2)
        e3 = self.enc3(x)
        
        x = self.pool(e3)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder with skip connections
        x = self.up(x)
        # The upsampled tensor `x` might be smaller than the skip connection `e3`
        # if the input size was odd. We need to crop `e3` to match `x`.
        diffY = e3.size()[2] - x.size()[2]
        diffX = e3.size()[3] - x.size()[3]
        e3_cropped = e3[:, :, diffY // 2 : e3.size()[2] - (diffY - diffY // 2),
                           diffX // 2 : e3.size()[3] - (diffX - diffX // 2)]
        x = torch.cat([x, e3_cropped], dim=1)
        x = self.dec3(x)
        
        x = self.up(x)
        # Crop e2
        diffY = e2.size()[2] - x.size()[2]
        diffX = e2.size()[3] - x.size()[3]
        e2_cropped = e2[:, :, diffY // 2 : e2.size()[2] - (diffY - diffY // 2),
                           diffX // 2 : e2.size()[3] - (diffX - diffX // 2)]
        x = torch.cat([x, e2_cropped], dim=1)
        x = self.dec2(x)
        
        x = self.up(x)
        # Crop e1 (this is where the original error occurs)
        diffY = e1.size()[2] - x.size()[2]
        diffX = e1.size()[3] - x.size()[3]
        e1_cropped = e1[:, :, diffY // 2 : e1.size()[2] - (diffY - diffY // 2),
                           diffX // 2 : e1.size()[3] - (diffX - diffX // 2)]
        x = torch.cat([x, e1_cropped], dim=1)
        x = self.dec1(x)
        
        # Output
        x = self.out_conv(x)
        
        return x


# ==========================================
# BASELINE CNN (for comparison)
# ==========================================

class SimpleUNet(nn.Module):
    """
    Standard U-Net with ReLU activations
    """
    def __init__(self, in_channels=1, out_channels=1, base_channels=16):
        super(SimpleUNet, self).__init__()
        
        # Encoder
        self.enc1 = self._make_block(in_channels, base_channels)
        self.enc2 = self._make_block(base_channels, base_channels*2)
        self.enc3 = self._make_block(base_channels*2, base_channels*4)
        
        # Bottleneck
        self.bottleneck = self._make_block(base_channels*4, base_channels*4)
        
        # Decoder
        self.dec3 = self._make_block(base_channels*8, base_channels*2)
        self.dec2 = self._make_block(base_channels*4, base_channels)
        self.dec1 = self._make_block(base_channels*2, base_channels)
        
        # Output
        self.out_conv = nn.Conv2d(base_channels, out_channels, 1)
        
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    
    def _make_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    # In the SimpleUNet class, replace the original forward method with this one:
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)

        x = self.pool(e1)
        e2 = self.enc2(x)

        x = self.pool(e2)
        e3 = self.enc3(x)

        x = self.pool(e3)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        x = self.up(x)
        # Crop e3 to match x's size
        diffY = e3.size()[2] - x.size()[2]
        diffX = e3.size()[3] - x.size()[3]
        e3_cropped = e3[:, :, diffY // 2 : e3.size()[2] - (diffY - diffY // 2),
                           diffX // 2 : e3.size()[3] - (diffX - diffX // 2)]
        x = torch.cat([x, e3_cropped], dim=1)
        x = self.dec3(x)

        x = self.up(x)
        # Crop e2 to match x's size
        diffY = e2.size()[2] - x.size()[2]
        diffX = e2.size()[3] - x.size()[3]
        e2_cropped = e2[:, :, diffY // 2 : e2.size()[2] - (diffY - diffY // 2),
                           diffX // 2 : e2.size()[3] - (diffX - diffX // 2)]
        x = torch.cat([x, e2_cropped], dim=1)
        x = self.dec2(x)

        x = self.up(x)
        # Crop e1 to match x's size
        diffY = e1.size()[2] - x.size()[2]
        diffX = e1.size()[3] - x.size()[3]
        e1_cropped = e1[:, :, diffY // 2 : e1.size()[2] - (diffY - diffY // 2),
                           diffX // 2 : e1.size()[3] - (diffX - diffX // 2)]
        x = torch.cat([x, e1_cropped], dim=1)
        x = self.dec1(x)

        # Output
        x = self.out_conv(x)

        return x


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ==========================================
# DATA LOADING
# ==========================================

def load_qsm_full_image(dataset_path, subject_id, orientation):
    """Load full 2D slice for image-to-image learning"""
    train_path = dataset_path / "train_data" / subject_id
    test_path = dataset_path / "test_data" / subject_id
    subject_path = train_path if train_path.exists() else test_path
    
    phi_file = subject_path / f"phi{orientation}.nii.gz"
    cosmos_file = subject_path / f"cosmos{orientation}.nii.gz"
    mask_file = subject_path / f"mask{orientation}.nii.gz"
    
    phi_3d = nib.load(str(phi_file)).get_fdata().astype(np.float32)
    cosmos_3d = nib.load(str(cosmos_file)).get_fdata().astype(np.float32)
    mask_3d = nib.load(str(mask_file)).get_fdata().astype(np.float32)
    
    slice_idx = phi_3d.shape[2] // 2
    
    phi_2d = phi_3d[:, :, slice_idx]
    cosmos_2d = cosmos_3d[:, :, slice_idx]
    mask_2d = mask_3d[:, :, slice_idx]
    
    # Normalize
    phi_mean = phi_2d[mask_2d > 0.5].mean()
    phi_std = phi_2d[mask_2d > 0.5].std()
    phi_2d = (phi_2d - phi_mean) / phi_std
    
    cosmos_mean = cosmos_2d[mask_2d > 0.5].mean()
    cosmos_std = cosmos_2d[mask_2d > 0.5].std()
    cosmos_2d = (cosmos_2d - cosmos_mean) / cosmos_std
    
    return phi_2d, cosmos_2d, mask_2d, (cosmos_mean, cosmos_std)


# ==========================================
# MAIN EXPERIMENT
# ==========================================

print("="*60)
print("KAN-UNet vs CNN-UNet: FULL-IMAGE QSM")
print("="*60)

# Load data
phi, cosmos, mask, norm_params = load_qsm_full_image(DATASET_PATH, SUBJECT_ID, ORIENTATION)
H, W = phi.shape

# Split into train/test regions
split_row = int(H * 0.5)
phi_train = phi[:split_row, :]
cosmos_train = cosmos[:split_row, :]
mask_train = mask[:split_row, :]

phi_test = phi[split_row:, :]
cosmos_test = cosmos[split_row:, :]
mask_test = mask[split_row:, :]

print(f"\nData: {H}×{W} pixels")
print(f"Train: {phi_train.shape}")
print(f"Test:  {phi_test.shape}")

# Convert to tensors
def to_batch(img):
    """Convert (H,W) to (1,1,H,W) batch"""
    return torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

X_train = to_batch(phi_train)
Y_train = to_batch(cosmos_train)
M_train = to_batch(mask_train)

X_test = to_batch(phi_test)
Y_test = to_batch(cosmos_test)
M_test = to_batch(mask_test)

# Models
print("\nInitializing models...")
model_kan = KANUNet(in_channels=1, out_channels=1, base_channels=8)
model_cnn = SimpleUNet(in_channels=1, out_channels=1, base_channels=8)

p_kan = count_params(model_kan)
p_cnn = count_params(model_cnn)

print(f"  KAN-UNet params: {p_kan:,}")
print(f"  CNN-UNet params: {p_cnn:,}")
print(f"  Ratio: {p_kan/p_cnn:.2f}x")

# Optimizers
opt_kan = optim.AdamW(model_kan.parameters(), lr=0.001, weight_decay=1e-5)
opt_cnn = optim.Adam(model_cnn.parameters(), lr=0.001)

scheduler_kan = optim.lr_scheduler.CosineAnnealingLR(opt_kan, EPOCHS)
scheduler_cnn = optim.lr_scheduler.CosineAnnealingLR(opt_cnn, EPOCHS)

criterion = nn.MSELoss()

# Training
print(f"\nTraining for {EPOCHS} epochs...")
loss_k_hist, loss_c_hist = [], []
test_k_hist, test_c_hist = [], []

for epoch in range(1, EPOCHS+1):
    # KAN training
    model_kan.train()
    opt_kan.zero_grad()
    pred_kan = model_kan(X_train)
    loss_k = criterion(pred_kan * M_train, Y_train * M_train)
    loss_k.backward()
    torch.nn.utils.clip_grad_norm_(model_kan.parameters(), 1.0)
    opt_kan.step()
    scheduler_kan.step()
    
    # CNN training
    model_cnn.train()
    opt_cnn.zero_grad()
    pred_cnn = model_cnn(X_train)
    loss_c = criterion(pred_cnn * M_train, Y_train * M_train)
    loss_c.backward()
    opt_cnn.step()
    scheduler_cnn.step()
    
    loss_k_hist.append(loss_k.item())
    loss_c_hist.append(loss_c.item())
    
    # Validation
    if epoch % 10 == 0:
        model_kan.eval()
        model_cnn.eval()
        with torch.no_grad():
            pred_k_test = model_kan(X_test)
            pred_c_test = model_cnn(X_test)
            test_loss_k = criterion(pred_k_test * M_test, Y_test * M_test).item()
            test_loss_c = criterion(pred_c_test * M_test, Y_test * M_test).item()
        
        test_k_hist.append(test_loss_k)
        test_c_hist.append(test_loss_c)
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch}:")
            print(f"  Train - CNN: {loss_c.item():.6f} | KAN: {loss_k.item():.6f}")
            print(f"  Test  - CNN: {test_loss_c:.6f} | KAN: {test_loss_k:.6f}")

# Final evaluation
print("\nFinal reconstruction...")
model_kan.eval()
model_cnn.eval()

with torch.no_grad():
    # Full image reconstruction
    X_full = to_batch(phi)
    recon_kan = model_kan(X_full).squeeze().numpy()
    recon_cnn = model_cnn(X_full).squeeze().numpy()
    
    # Denormalize
    cosmos_mean, cosmos_std = norm_params
    recon_kan = recon_kan * cosmos_std + cosmos_mean
    recon_cnn = recon_cnn * cosmos_std + cosmos_mean
    cosmos_denorm = cosmos * cosmos_std + cosmos_mean

# Calculate MSE on test region only
mask_test_region = mask[split_row:, :] > 0.5
mse_kan = np.mean((recon_kan[split_row:, :][mask_test_region] - cosmos_denorm[split_row:, :][mask_test_region])**2)
mse_cnn = np.mean((recon_cnn[split_row:, :][mask_test_region] - cosmos_denorm[split_row:, :][mask_test_region])**2)

# Report
print("\n" + "="*60)
print("RESULTS: FULL-IMAGE QSM RECONSTRUCTION")
print("="*60)
print(f"{'Metric':<30} | {'CNN-UNet':<15} | {'KAN-UNet':<15}")
print("-" * 65)
print(f"{'Parameters':<30} | {p_cnn:<15,} | {p_kan:<15,}")
print(f"{'Param Ratio':<30} | {'1.0x':<15} | {p_kan/p_cnn:.2f}x")
print(f"{'Train Loss (Final)':<30} | {loss_c_hist[-1]:<15.6f} | {loss_k_hist[-1]:<15.6f}")
print(f"{'Test Loss (Final)':<30} | {test_c_hist[-1]:<15.6f} | {test_k_hist[-1]:<15.6f}")
print(f"{'Test Recon MSE':<30} | {mse_cnn:<15.6f} | {mse_kan:<15.6f}")
print(f"{'Improvement':<30} | {'baseline':<15} | {(1 - mse_kan/mse_cnn)*100:+.1f}%")
print("-" * 65)

if mse_kan < mse_cnn:
    print("✓ KAN-UNet outperforms CNN-UNet!")
elif mse_kan < mse_cnn * 1.05:
    print("≈ Competitive performance")
else:
    print("⚠ CNN-UNet performs better")

# Visualization
fig = plt.figure(figsize=(18, 10))
gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

# Row 1: Inputs and GT
ax1 = fig.add_subplot(gs[0, 0])
ax1.imshow(mask, cmap='gray')
ax1.axhline(y=split_row, color='r', linestyle='--', linewidth=2)
ax1.set_title('Mask (Train/Test Split)')
ax1.axis('off')

ax2 = fig.add_subplot(gs[0, 1])
ax2.imshow(phi, cmap='twilight')
ax2.set_title('Input: Phi (Normalized)')
ax2.axis('off')

ax3 = fig.add_subplot(gs[0, 2])
ax3.imshow(cosmos_denorm, cmap='gray', vmin=-0.15, vmax=0.15)
ax3.set_title('Ground Truth: COSMOS')
ax3.axis('off')

ax4 = fig.add_subplot(gs[0, 3])
ax4.plot(loss_c_hist, label='CNN', alpha=0.7)
ax4.plot(loss_k_hist, label='KAN', alpha=0.7)
ax4.set_xlabel('Epoch')
ax4.set_ylabel('Training Loss')
ax4.set_title('Training Curves')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Row 2: Reconstructions
ax5 = fig.add_subplot(gs[1, 0])
ax5.imshow(recon_cnn, cmap='gray', vmin=-0.15, vmax=0.15)
ax5.set_title(f'CNN-UNet\nTest MSE: {mse_cnn:.6f}')
ax5.axis('off')

ax6 = fig.add_subplot(gs[1, 1])
ax6.imshow(recon_kan, cmap='gray', vmin=-0.15, vmax=0.15)
ax6.set_title(f'KAN-UNet\nTest MSE: {mse_kan:.6f}')
ax6.axis('off')

# Row 2: Error maps
err_cnn = np.abs(recon_cnn - cosmos_denorm)
err_cnn[:split_row, :] = 0
ax7 = fig.add_subplot(gs[1, 2])
ax7.imshow(err_cnn, cmap='hot', vmin=0, vmax=0.05)
ax7.set_title('CNN Error (Test)')
ax7.axis('off')

err_kan = np.abs(recon_kan - cosmos_denorm)
err_kan[:split_row, :] = 0
ax8 = fig.add_subplot(gs[1, 3])
ax8.imshow(err_kan, cmap='hot', vmin=0, vmax=0.05)
ax8.set_title('KAN Error (Test)')
ax8.axis('off')

# Row 3: Line profiles (center column)
center_col = W // 2
ax9 = fig.add_subplot(gs[2, :2])
ax9.plot(cosmos_denorm[:, center_col], 'k-', label='Ground Truth', linewidth=2)
ax9.plot(recon_cnn[:, center_col], 'b--', label='CNN', alpha=0.7)
ax9.plot(recon_kan[:, center_col], 'r--', label='KAN', alpha=0.7)
ax9.axvline(x=split_row, color='gray', linestyle=':', label='Train/Test Split')
ax9.set_xlabel('Row')
ax9.set_ylabel('Susceptibility')
ax9.set_title(f'Center Column Profile (x={center_col})')
ax9.legend()
ax9.grid(True, alpha=0.3)

# Test region statistics
ax10 = fig.add_subplot(gs[2, 2:])
test_metrics = {
    'CNN': [loss_c_hist[-1], test_c_hist[-1], mse_cnn],
    'KAN': [loss_k_hist[-1], test_k_hist[-1], mse_kan]
}
x = np.arange(3)
width = 0.35
labels = ['Train Loss', 'Test Loss', 'Test MSE']
ax10.bar(x - width/2, test_metrics['CNN'], width, label='CNN', alpha=0.8)
ax10.bar(x + width/2, test_metrics['KAN'], width, label='KAN', alpha=0.8)
ax10.set_ylabel('Loss / MSE')
ax10.set_title('Performance Comparison')
ax10.set_xticks(x)
ax10.set_xticklabels(labels)
ax10.legend()
ax10.grid(True, alpha=0.3, axis='y')

plt.savefig(f'kan_unet_qsm_{SUBJECT_ID}.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\n✓ Results saved to kan_unet_qsm_{SUBJECT_ID}.png")