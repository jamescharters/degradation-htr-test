"""
Modified version of your KAN experiment using REAL OSF QSM data
This replaces synthetic phase generation with actual MRI measurements
"""

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import time
from pathlib import Path

# ==========================================
# 1. CONFIGURATION
# ==========================================
DATASET_PATH = Path("./data/OSF_QSM_Dataset")
SUBJECT_ID = "Subject1"  # Update based on your actual folder names
ORIENTATION = 1  # Orientation number (1-18 typically)
SLICE_NUM = None  # None = auto-select middle slice
PATCH_THRESHOLD = 0.05 
EPOCHS = 300

torch.manual_seed(42)
np.random.seed(42)

# ==========================================
# 2. LOAD REAL QSM DATA (REPLACES SYNTHETIC)
# ==========================================

def load_real_qsm_data(dataset_path, subject_id, orientation, slice_idx):
    """
    Load real QSM data from OSF dataset
    Returns data in the same format as your synthetic generator
    
    Based on README.txt:
    - phi{N}.nii.gz = local field map (INPUT)
    - cosmos{N}.nii.gz = COSMOS map (GROUND TRUTH)
    - mask{N}.nii.gz = brain mask
    where N is the orientation number
    """
    # Check both train_data and test_data
    train_path = dataset_path / "train_data" / subject_id
    test_path = dataset_path / "test_data" / subject_id
    
    if train_path.exists():
        subject_path = train_path
        split = "train"
    elif test_path.exists():
        subject_path = test_path
        split = "test"
    else:
        raise FileNotFoundError(
            f"Subject {subject_id} not found.\n"
            f"Checked: {train_path} and {test_path}"
        )
    
    print(f"Loading {subject_id} from {split} split (Orientation {orientation})...")
    
    # Find files based on README naming convention
    phi_file = subject_path / f"phi{orientation}.nii.gz"
    cosmos_file = subject_path / f"cosmos{orientation}.nii.gz"
    mask_file = subject_path / f"mask{orientation}.nii.gz"
    
    # Alternative extensions if .nii.gz doesn't exist
    if not phi_file.exists():
        phi_file = subject_path / f"phi{orientation}.nii"
    if not cosmos_file.exists():
        cosmos_file = subject_path / f"cosmos{orientation}.nii"
    if not mask_file.exists():
        mask_file = subject_path / f"mask{orientation}.nii"
    
    if not phi_file.exists():
        raise FileNotFoundError(
            f"phi{orientation} file not found in {subject_path}\n"
            f"Available files: {[f.name for f in subject_path.glob('*.nii*')]}"
        )
    
    # Load 3D volumes
    phi_img = nib.load(str(phi_file))
    phi_3d = phi_img.get_fdata().astype(np.float32)
    
    cosmos_img = nib.load(str(cosmos_file))
    cosmos_3d = cosmos_img.get_fdata().astype(np.float32)
    
    if mask_file.exists():
        mask_img = nib.load(str(mask_file))
        mask_3d = mask_img.get_fdata().astype(np.float32)
    else:
        print("⚠ Mask not found, creating from COSMOS threshold...")
        mask_3d = (np.abs(cosmos_3d) > 0.01).astype(np.float32)
    
    # Auto-select middle slice if not specified
    if slice_idx is None:
        slice_idx = phi_3d.shape[2] // 2
    
    # Extract 2D slice
    phi_2d = phi_3d[:, :, slice_idx]
    cosmos_2d = cosmos_3d[:, :, slice_idx]
    mask_2d = mask_3d[:, :, slice_idx]
    
    print(f"✓ Loaded slice {slice_idx}/{phi_3d.shape[2]}")
    print(f"  Phi (local field) range: [{phi_2d.min():.3f}, {phi_2d.max():.3f}]")
    print(f"  COSMOS range: [{cosmos_2d.min():.3f}, {cosmos_2d.max():.3f}]")
    print(f"  Mask coverage: {mask_2d.sum() / mask_2d.size * 100:.1f}%")
    
    # Map to your variable names
    # phi = local field (already processed, phase-like)
    # cosmos = ground truth susceptibility
    magnitude = mask_2d
    phase_wrapped = phi_2d  # phi is the local field
    phase_true = cosmos_2d  # COSMOS is ground truth
    
    return magnitude, phase_true, phase_wrapped, slice_idx


# ==========================================
# 3. YOUR ORIGINAL ARCHITECTURES (UNCHANGED)
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
# 4. YOUR PATCH EXTRACTION (UNCHANGED)
# ==========================================

def extract_patches(phase_wrapped, phase_true, magnitude):
    H, W = phase_wrapped.shape
    patches, targets = [], []
    
    for r in range(1, H-2, 2):
        for c in range(1, W-2, 2):
            if np.mean(magnitude[r-1:r+2, c-1:c+2]) < PATCH_THRESHOLD: 
                continue

            patch = phase_wrapped[r-1:r+2, c-1:c+2].flatten()
            
            dx_naive = phase_wrapped[r, c+1] - phase_wrapped[r, c]
            dy_naive = phase_wrapped[r+1, c] - phase_wrapped[r, c]
            dx_true = phase_true[r, c+1] - phase_true[r, c]
            dy_true = phase_true[r+1, c] - phase_true[r, c]
            
            kx = (dx_true - dx_naive) / (2 * np.pi)
            ky = (dy_true - dy_naive) / (2 * np.pi)
            
            patches.append(patch)
            targets.append([np.round(kx), np.round(ky)])
            
    return np.array(patches), np.array(targets)


def solver_fft(gx, gy):
    rows, cols = gx.shape
    wx = np.fft.fftfreq(cols) * 2 * np.pi
    wy = np.fft.fftfreq(rows) * 2 * np.pi
    fx, fy = np.meshgrid(wx, wy)
    k2 = fx**2 + fy**2
    k2[0, 0] = 1.0 
    G_x_f = np.fft.fft2(gx)
    G_y_f = np.fft.fft2(gy)
    Z_f = -1j * (fx * G_x_f + fy * G_y_f) / k2
    Z_f[0, 0] = 0.0
    return np.real(np.fft.ifft2(Z_f))


# ==========================================
# 5. MAIN EXPERIMENT (MODIFIED FOR REAL DATA)
# ==========================================

print("="*60)
print("KAN vs MLP: REAL QSM DATA TEST")
print("Using OSF Multi-Orientation Dataset")
print("="*60)

# Load REAL data instead of synthetic
mag, p_true, p_wrapped, actual_slice = load_real_qsm_data(
    DATASET_PATH, SUBJECT_ID, ORIENTATION, SLICE_NUM
)
H, W = mag.shape
split_row = int(H * 0.5)

print(f"\nSplitting Image at Row {split_row}")

# Extract patches (same as before)
X_np_train, Y_np_train = extract_patches(
    p_wrapped[:split_row, :], 
    p_true[:split_row, :], 
    mag[:split_row, :]
)
X_train = torch.tensor(X_np_train, dtype=torch.float32) / np.pi
Y_train = torch.tensor(Y_np_train, dtype=torch.float32)

X_np_test, Y_np_test = extract_patches(
    p_wrapped[split_row:, :], 
    p_true[split_row:, :], 
    mag[split_row:, :]
)
X_test = torch.tensor(X_np_test, dtype=torch.float32) / np.pi
Y_test = torch.tensor(Y_np_test, dtype=torch.float32)

print(f"Training Patches: {len(X_train)}")
print(f"Testing Patches: {len(X_test)}")

# Models (same as before)
model_kan = SimpleKAN([9, 16, 2], grid_size=10)
model_mlp = SimpleMLP(9, 64, 2)

p_kan = count_params(model_kan)
p_mlp = count_params(model_mlp)

opt_kan = optim.AdamW(model_kan.parameters(), lr=0.005)
opt_mlp = optim.Adam(model_mlp.parameters(), lr=0.002)
criterion = nn.MSELoss()

# Training
print(f"\nTraining on REAL data ({EPOCHS} epochs)...")
loss_k_hist, loss_m_hist = [], []

for epoch in range(1, EPOCHS+1):
    opt_kan.zero_grad()
    loss_k = criterion(model_kan(X_train), Y_train)
    loss_k.backward()
    opt_kan.step()
    
    opt_mlp.zero_grad()
    loss_m = criterion(model_mlp(X_train), Y_train)
    loss_m.backward()
    opt_mlp.step()
    
    loss_k_hist.append(loss_k.item())
    loss_m_hist.append(loss_m.item())
    
    if epoch % 50 == 0:
        print(f"Epoch {epoch}: MLP {loss_m.item():.5f} | KAN {loss_k.item():.5f}")

# Evaluation
model_kan.eval()
model_mlp.eval()
with torch.no_grad():
    test_loss_k = criterion(model_kan(X_test), Y_test).item()
    test_loss_m = criterion(model_mlp(X_test), Y_test).item()

# Full reconstruction
print("\nRunning Full Image Reconstruction...")
gx_naive = np.zeros((H,W))
gy_naive = np.zeros((H,W))
gx_naive[:, :-1] = p_wrapped[:, 1:] - p_wrapped[:, :-1]
gy_naive[:-1, :] = p_wrapped[1:, :] - p_wrapped[:-1, :]

gx_k, gy_k = np.zeros((H,W)), np.zeros((H,W))
gx_m, gy_m = np.zeros((H,W)), np.zeros((H,W))

with torch.no_grad():
    for r in range(1, H-1):
        for c in range(1, W-1):
            if mag[r,c] < PATCH_THRESHOLD: continue
            patch = p_wrapped[r-1:r+2, c-1:c+2].flatten()
            inp = torch.tensor(patch, dtype=torch.float32).unsqueeze(0) / np.pi
            
            k_k = np.round(model_kan(inp).numpy()[0])
            k_m = np.round(model_mlp(inp).numpy()[0])
            
            gx_k[r,c] = gx_naive[r,c] + (2 * np.pi * k_k[0])
            gy_k[r,c] = gy_naive[r,c] + (2 * np.pi * k_k[1])
            gx_m[r,c] = gx_naive[r,c] + (2 * np.pi * k_m[0])
            gy_m[r,c] = gy_naive[r,c] + (2 * np.pi * k_m[1])

rec_k = solver_fft(gx_k, gy_k)
rec_m = solver_fft(gx_m, gy_m)

# Calculate MSE on test half
mask_bottom = np.zeros_like(mag, dtype=bool)
mask_bottom[split_row:, :] = (mag[split_row:, :] > PATCH_THRESHOLD)

rec_k -= np.mean(rec_k[mask_bottom])
rec_m -= np.mean(rec_m[mask_bottom])
gt_norm = p_true - np.mean(p_true[mask_bottom])

mse_k_test = np.mean((rec_k[mask_bottom] - gt_norm[mask_bottom])**2)
mse_m_test = np.mean((rec_m[mask_bottom] - gt_norm[mask_bottom])**2)

# Report
print("\n" + "="*60)
print("RESULTS: REAL QSM DATA")
print("="*60)
print(f"{'Metric':<30} | {'MLP':<15} | {'KAN':<15}")
print("-" * 65)
print(f"{'Parameters':<30} | {p_mlp:<15} | {p_kan:<15}")
print(f"{'Compression':<30} | {'1.0x':<15} | {p_kan/p_mlp:.2f}x")
print(f"{'Train Loss':<30} | {loss_m_hist[-1]:<15.5f} | {loss_k_hist[-1]:<15.5f}")
print(f"{'Test Loss':<30} | {test_loss_m:<15.5f} | {test_loss_k:<15.5f}")
print(f"{'Test Recon MSE':<30} | {mse_m_test:<15.5f} | {mse_k_test:<15.5f}")
print("-" * 65)

if mse_k_test <= mse_m_test:
    print(">> ✓ KAN generalizes as well or better than MLP on REAL data!")
else:
    print(">> Small generalization gap observed on real data.")

# Visualization
fig, ax = plt.subplots(2, 3, figsize=(15, 9))
ax[0,0].imshow(mag, cmap='gray')
ax[0,0].set_title("Magnitude/Mask")
ax[0,0].axhline(y=split_row, color='r', linestyle='--')

ax[0,1].imshow(p_wrapped, cmap='twilight')
ax[0,1].set_title("Input: Local Field")

ax[0,2].imshow(gt_norm, cmap='twilight')
ax[0,2].set_title("Ground Truth: COSMOS")

ax[1,0].imshow(rec_m, cmap='twilight')
ax[1,0].set_title(f"MLP Recon\nTest MSE: {mse_m_test:.4f}")

ax[1,1].imshow(rec_k, cmap='twilight')
ax[1,1].set_title(f"KAN Recon\nTest MSE: {mse_k_test:.4f}")

err_map = (rec_k - gt_norm)**2
err_map[:split_row, :] = 0
ax[1,2].imshow(err_map, cmap='hot')
ax[1,2].set_title("KAN Error (Test Half)")

plt.tight_layout()
save_name = f'kan_qsm_real_{SUBJECT_ID}_ori{ORIENTATION}_slice{actual_slice}.png'
plt.savefig(save_name, dpi=150)
plt.show()

print(f"\n✓ Results saved to: {save_name}")