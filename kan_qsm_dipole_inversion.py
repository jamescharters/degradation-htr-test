"""
KAN for QSM Dipole Inversion (the ACTUAL QSM problem)
Task: Predict susceptibility (COSMOS) from local field (phi)

This is more relevant to QSM research than phase unwrapping!
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
EPOCHS = 300

torch.manual_seed(42)
np.random.seed(42)

# ==========================================
# ARCHITECTURES (Same as before)
# ==========================================

class KANLayer(nn.Module):
    def __init__(self, in_features, out_features, grid_size=10, spline_order=3):
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

class SimpleKAN(nn.Module):
    def __init__(self, layers_hidden, grid_size=10):
        super(SimpleKAN, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers_hidden) - 1):
            self.layers.append(KANLayer(layers_hidden[i], layers_hidden[i+1], grid_size=grid_size))  
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ==========================================
# DATA LOADING FOR QSM TASK
# ==========================================

def load_qsm_data(dataset_path, subject_id, orientation):
    """
    Load QSM data for dipole inversion task
    Input: local field (phi)
    Target: susceptibility (COSMOS)
    """
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
    
    return phi_2d, cosmos_2d, mask_2d

def extract_patches_qsm(phi, cosmos, mask, patch_size=5, threshold=0.5):
    """
    Extract patches for QSM dipole inversion
    Input: local field patch
    Output: center susceptibility value
    """
    H, W = phi.shape
    half = patch_size // 2
    
    patches, targets = [], []
    
    for r in range(half, H-half, 2):
        for c in range(half, W-half, 2):
            if mask[r, c] < threshold:
                continue
            
            # Input: local field neighborhood
            patch = phi[r-half:r+half+1, c-half:c+half+1].flatten()
            
            # Target: center susceptibility value
            target = cosmos[r, c]
            
            patches.append(patch)
            targets.append(target)
    
    return np.array(patches), np.array(targets)

# ==========================================
# MAIN EXPERIMENT
# ==========================================

print("="*60)
print("KAN vs MLP: QSM DIPOLE INVERSION")
print("Task: Predict Susceptibility from Local Field")
print("="*60)

# Load data
phi, cosmos, mask = load_qsm_data(DATASET_PATH, SUBJECT_ID, ORIENTATION)
H, W = phi.shape
split_row = int(H * 0.5)

print(f"\nData ranges:")
print(f"  Phi: [{phi.min():.3f}, {phi.max():.3f}]")
print(f"  COSMOS: [{cosmos.min():.3f}, {cosmos.max():.3f}]")

# Split data
phi_train, cosmos_train, mask_train = phi[:split_row], cosmos[:split_row], mask[:split_row]
phi_test, cosmos_test, mask_test = phi[split_row:], cosmos[split_row:], mask[split_row:]

# Extract patches
print("\nExtracting patches...")
X_train_np, Y_train_np = extract_patches_qsm(phi_train, cosmos_train, mask_train)
X_test_np, Y_test_np = extract_patches_qsm(phi_test, cosmos_test, mask_test)

# Normalize
phi_std = phi[mask > 0.5].std()
X_train = torch.tensor(X_train_np / phi_std, dtype=torch.float32)
X_test = torch.tensor(X_test_np / phi_std, dtype=torch.float32)
Y_train = torch.tensor(Y_train_np, dtype=torch.float32).unsqueeze(1)
Y_test = torch.tensor(Y_test_np, dtype=torch.float32).unsqueeze(1)

print(f"Train patches: {len(X_train)}")
print(f"Test patches: {len(X_test)}")

# Models - IMPROVED ARCHITECTURES
patch_dim = X_train.shape[1]

# KAN: Deeper network with more capacity
model_kan = SimpleKAN([patch_dim, 32, 16, 1], grid_size=8)  # 3 layers, smaller grid

# MLP: Match parameters more closely
model_mlp = SimpleMLP(patch_dim, 48, 1)  # Reduced from 64

p_kan = count_params(model_kan)
p_mlp = count_params(model_mlp)

print(f"\nModel architectures:")
print(f"  KAN: {patch_dim}→32→16→1 (grid={8})")
print(f"  MLP: {patch_dim}→48→48→1")
print(f"  KAN params: {p_kan}")
print(f"  MLP params: {p_mlp}")
print(f"  Compression: {p_kan/p_mlp:.2f}x")

# Better optimizers and learning rates
opt_kan = optim.AdamW(model_kan.parameters(), lr=0.002, weight_decay=1e-4)
opt_mlp = optim.Adam(model_mlp.parameters(), lr=0.001)

# Learning rate schedulers
scheduler_kan = optim.lr_scheduler.ReduceLROnPlateau(opt_kan, 'min', patience=20, factor=0.5)
scheduler_mlp = optim.lr_scheduler.ReduceLROnPlateau(opt_mlp, 'min', patience=20, factor=0.5)

criterion = nn.MSELoss()

# Training with validation monitoring
print(f"\nTraining ({EPOCHS} epochs)...")
loss_k_hist, loss_m_hist = [], []
best_test_k, best_test_m = float('inf'), float('inf')

for epoch in range(1, EPOCHS+1):
    # KAN
    model_kan.train()
    opt_kan.zero_grad()
    loss_k = criterion(model_kan(X_train), Y_train)
    loss_k.backward()
    torch.nn.utils.clip_grad_norm_(model_kan.parameters(), 1.0)  # Gradient clipping
    opt_kan.step()
    
    # MLP
    model_mlp.train()
    opt_mlp.zero_grad()
    loss_m = criterion(model_mlp(X_train), Y_train)
    loss_m.backward()
    opt_mlp.step()
    
    loss_k_hist.append(loss_k.item())
    loss_m_hist.append(loss_m.item())
    
    # Validation every 10 epochs
    if epoch % 10 == 0:
        model_kan.eval()
        model_mlp.eval()
        with torch.no_grad():
            val_loss_k = criterion(model_kan(X_test), Y_test).item()
            val_loss_m = criterion(model_mlp(X_test), Y_test).item()
        
        # Update learning rates
        scheduler_kan.step(val_loss_k)
        scheduler_mlp.step(val_loss_m)
        
        # Track best
        if val_loss_k < best_test_k:
            best_test_k = val_loss_k
        if val_loss_m < best_test_m:
            best_test_m = val_loss_m
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch}:")
            print(f"  Train - MLP: {loss_m.item():.6f} | KAN: {loss_k.item():.6f}")
            print(f"  Val   - MLP: {val_loss_m:.6f} | KAN: {val_loss_k:.6f}")

# Evaluation
model_kan.eval()
model_mlp.eval()
with torch.no_grad():
    test_loss_k = criterion(model_kan(X_test), Y_test).item()
    test_loss_m = criterion(model_mlp(X_test), Y_test).item()

# Full image reconstruction
print("\nReconstrucing full images...")
recon_kan = np.zeros((H, W))
recon_mlp = np.zeros((H, W))
half = 2

with torch.no_grad():
    for r in range(half, H-half):
        for c in range(half, W-half):
            if mask[r, c] < 0.5:
                continue
            
            patch = phi[r-half:r+half+1, c-half:c+half+1].flatten()
            inp = torch.tensor(patch / phi_std, dtype=torch.float32).unsqueeze(0)
            
            recon_kan[r, c] = model_kan(inp).item()
            recon_mlp[r, c] = model_mlp(inp).item()

# Calculate MSE on test region
mask_test_region = mask[split_row:, :] > 0.5
mse_kan = np.mean((recon_kan[split_row:, :][mask_test_region] - cosmos[split_row:, :][mask_test_region])**2)
mse_mlp = np.mean((recon_mlp[split_row:, :][mask_test_region] - cosmos[split_row:, :][mask_test_region])**2)

# Report
print("\n" + "="*60)
print("RESULTS: QSM DIPOLE INVERSION")
print("="*60)
print(f"{'Metric':<30} | {'MLP':<15} | {'KAN':<15}")
print("-" * 65)
print(f"{'Parameters':<30} | {p_mlp:<15} | {p_kan:<15}")
print(f"{'Param Ratio':<30} | {'1.0x':<15} | {p_kan/p_mlp:.2f}x")
print(f"{'Train Loss (Final)':<30} | {loss_m_hist[-1]:<15.6f} | {loss_k_hist[-1]:<15.6f}")
print(f"{'Test Loss (Final)':<30} | {test_loss_m:<15.6f} | {test_loss_k:<15.6f}")
print(f"{'Test Loss (Best)':<30} | {best_test_m:<15.6f} | {best_test_k:<15.6f}")
print(f"{'Test Recon MSE':<30} | {mse_mlp:<15.6f} | {mse_kan:<15.6f}")
print(f"{'Improvement':<30} | {'baseline':<15} | {(1 - mse_kan/mse_mlp)*100:+.1f}%")
print("-" * 65)

if mse_kan < mse_mlp:
    print("✓ KAN outperforms MLP on unseen data!")
elif mse_kan < mse_mlp * 1.1:
    print("≈ KAN competitive with MLP (within 10%)")
else:
    print("⚠ MLP outperforms KAN - may need architecture tuning")

# Visualization - IMPROVED
fig, axes = plt.subplots(2, 4, figsize=(18, 8))

axes[0,0].imshow(mask, cmap='gray')
axes[0,0].axhline(y=split_row, color='r', linestyle='--', linewidth=2)
axes[0,0].set_title("Mask (Train/Test Split)")
axes[0,0].axis('off')

axes[0,1].imshow(phi, cmap='twilight')
axes[0,1].set_title("Input: Local Field (Phi)")
axes[0,1].axis('off')

axes[0,2].imshow(cosmos, cmap='gray', vmin=-0.15, vmax=0.15)
axes[0,2].set_title("Ground Truth: COSMOS")
axes[0,2].axis('off')

# Training curves
axes[0,3].plot(loss_m_hist, label='MLP', alpha=0.7)
axes[0,3].plot(loss_k_hist, label='KAN', alpha=0.7)
axes[0,3].set_xlabel('Epoch')
axes[0,3].set_ylabel('Training Loss')
axes[0,3].set_title('Training Curves')
axes[0,3].legend()
axes[0,3].set_yscale('log')
axes[0,3].grid(True, alpha=0.3)

axes[1,0].imshow(recon_mlp, cmap='gray', vmin=-0.15, vmax=0.15)
axes[1,0].set_title(f"MLP Reconstruction\nTest MSE: {mse_mlp:.6f}")
axes[1,0].axis('off')

axes[1,1].imshow(recon_kan, cmap='gray', vmin=-0.15, vmax=0.15)
axes[1,1].set_title(f"KAN Reconstruction\nTest MSE: {mse_kan:.6f}")
axes[1,1].axis('off')

# Error maps
error_mlp = np.abs(recon_mlp - cosmos)
error_mlp[:split_row, :] = 0
axes[1,2].imshow(error_mlp, cmap='hot', vmin=0, vmax=0.05)
axes[1,2].set_title("MLP Error (Test)")
axes[1,2].axis('off')

error_kan = np.abs(recon_kan - cosmos)
error_kan[:split_row, :] = 0
axes[1,3].imshow(error_kan, cmap='hot', vmin=0, vmax=0.05)
axes[1,3].set_title("KAN Error (Test)")
axes[1,3].axis('off')

plt.tight_layout()
plt.savefig(f'kan_qsm_dipole_inversion_{SUBJECT_ID}.png', dpi=150)
plt.show()

print(f"\n✓ Results saved!")