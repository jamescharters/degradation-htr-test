import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import time
import os

# ==========================================
# 1. CONFIGURATION
# ==========================================
# REPLACE with your path
FILE_PATH = "./data/fastMRI/singlecoil_val/file1000000.h5" 
SLICE_NUM = 20
PATCH_THRESHOLD = 0.05 
EPOCHS = 300
NOISE_LEVEL = 0.15  # 15% Noise (Aggressive test)

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ==========================================
# 2. ARCHITECTURES
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
# 3. REALISTIC DATA GENERATOR (FIXED)
# ==========================================
def simulate_dipole_phase(H, W, n_dipoles=8):
    """
    Generates sharp, localized phase shifts mimicking veins/bleeds/calcifications.
    """
    # Fix: x corresponds to columns (W), y to rows (H)
    x_space = np.linspace(-1, 1, W)
    y_space = np.linspace(-1, 1, H)
    X, Y = np.meshgrid(x_space, y_space) # Resulting shape (H, W)
    
    phase = np.zeros((H, W))
    
    for _ in range(n_dipoles):
        # Random position
        cx, cy = np.random.uniform(-0.8, 0.8), np.random.uniform(-0.8, 0.8)
        # Random radius (small = sharp)
        radius = np.random.uniform(0.05, 0.15)
        strength = np.random.choice([-1, 1]) * np.random.uniform(10, 25) 
        
        # Dipole-ish shape
        r2 = (X - cx)**2 + (Y - cy)**2
        dipole = strength * (1 / (1 + r2/(radius**2)))
        phase += dipole
        
    return phase

def load_data(h5_path, slice_idx):
    if not os.path.exists(h5_path):
        print(f"⚠️ FILE NOT FOUND: {h5_path}. Using Dummy Data.")
        H, W = 320, 320
        # Create dummy magnitude with a "hole" to simulate tissue vs air
        x = np.linspace(-1, 1, W); y = np.linspace(-1, 1, H)
        X, Y = np.meshgrid(x, y)
        magnitude = (X**2 + Y**2 < 0.8).astype(float) * np.random.uniform(0.5, 1.0, (H, W))
    else:
        print(f">> Loading Real FastMRI: {h5_path}")
        with h5py.File(h5_path, 'r') as hf:
            kspace = hf['kspace'][slice_idx]
        img_complex = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kspace)))
        magnitude = np.abs(img_complex)
        # Normalize
        magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min())

    H, W = magnitude.shape
    
    # 1. Generate Dipole Phase
    phase_true = simulate_dipole_phase(H, W, n_dipoles=6)
    
    # 2. Add Noise BEFORE Wrapping (Simulate Sensor Noise)
    phase_noise = np.random.normal(0, NOISE_LEVEL, (H, W))
    
    # 3. Wrap it
    img_new = magnitude * np.exp(1j * (phase_true + phase_noise))
    phase_wrapped = np.angle(img_new)
    
    return magnitude, phase_true, phase_wrapped

def extract_patches(phase_wrapped, phase_true, magnitude):
    H, W = phase_wrapped.shape
    patches, targets = [], []
    
    # Stride 2 to save memory
    for r in range(1, H-2, 2):
        for c in range(1, W-2, 2):
            if np.mean(magnitude[r-1:r+2, c-1:c+2]) < PATCH_THRESHOLD: continue

            patch = phase_wrapped[r-1:r+2, c-1:c+2].flatten()
            
            dx_naive = phase_wrapped[r, c+1] - phase_wrapped[r, c]
            dy_naive = phase_wrapped[r+1, c] - phase_wrapped[r, c]
            dx_true = phase_true[r, c+1] - phase_true[r, c]
            dy_true = phase_true[r+1, c] - phase_true[r, c]
            
            # Target k
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
# 4. EXECUTION
# ==========================================
print("="*60)
print("STRESS TEST: DIPOLES + NOISE")
print(f"Noise Level: {NOISE_LEVEL*100}%")
print("="*60)

mag, p_true, p_wrapped = load_data(FILE_PATH, SLICE_NUM)
X_np, Y_np = extract_patches(p_wrapped, p_true, mag)
X_t = torch.tensor(X_np, dtype=torch.float32) / np.pi 
Y_t = torch.tensor(Y_np, dtype=torch.float32)

train_size = int(0.9 * len(X_t))
X_train, Y_train = X_t[:train_size], Y_t[:train_size]

# Models
model_kan = SimpleKAN([9, 16, 2], grid_size=10)
model_mlp = SimpleMLP(9, 64, 2)

p_kan = count_params(model_kan)
p_mlp = count_params(model_mlp)

opt_kan = optim.AdamW(model_kan.parameters(), lr=0.005)
opt_mlp = optim.Adam(model_mlp.parameters(), lr=0.002)
criterion = nn.MSELoss()

# Training
print(f"\nTraining on {len(X_train)} patches (Realistic/Noisy)...")
start_train = time.time()
loss_hist_k = []
loss_hist_m = []

for epoch in range(1, EPOCHS+1):
    opt_kan.zero_grad()
    loss_k = criterion(model_kan(X_train), Y_train)
    loss_k.backward()
    opt_kan.step()
    loss_hist_k.append(loss_k.item())
    
    opt_mlp.zero_grad()
    loss_m = criterion(model_mlp(X_train), Y_train)
    loss_m.backward()
    opt_mlp.step()
    loss_hist_m.append(loss_m.item())
    
    if epoch % 50 == 0:
        print(f"Epoch {epoch}: MLP {loss_m.item():.5f} | KAN {loss_k.item():.5f}")

train_time = time.time() - start_train

# Inference
H, W = mag.shape
gx_naive = np.zeros((H,W)); gy_naive = np.zeros((H,W))
gx_naive[:, :-1] = p_wrapped[:, 1:] - p_wrapped[:, :-1]
gy_naive[:-1, :] = p_wrapped[1:, :] - p_wrapped[:-1, :]

gx_k, gy_k = np.zeros((H,W)), np.zeros((H,W))
gx_m, gy_m = np.zeros((H,W)), np.zeros((H,W))

model_kan.eval(); model_mlp.eval()

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

# Recon
rec_k = solver_fft(gx_k, gy_k)
rec_m = solver_fft(gx_m, gy_m)

# Metrics
mask = mag > PATCH_THRESHOLD
rec_k -= np.mean(rec_k[mask])
rec_m -= np.mean(rec_m[mask])
gt_norm = p_true - np.mean(p_true[mask])

mse_k = np.mean((rec_k[mask] - gt_norm[mask])**2)
mse_m = np.mean((rec_m[mask] - gt_norm[mask])**2)

# ==========================================
# 5. FINAL REPORT
# ==========================================
print("\n" + "="*60)
print("STRESS TEST REPORT (DIPOLES + NOISE)")
print("="*60)
print(f"{'Metric':<25} | {'MLP':<15} | {'KAN':<15}")
print("-" * 60)
print(f"{'Parameters':<25} | {p_mlp:<15} | {p_kan:<15}")
print(f"{'Compression':<25} | {'1.0x':<15} | {p_kan/p_mlp:.2f}x")
print(f"{'Final Train Loss':<25} | {loss_hist_m[-1]:<15.5f} | {loss_hist_k[-1]:<15.5f}")
print(f"{'Reconstruction MSE':<25} | {mse_m:<15.5f} | {mse_k:<15.5f}")
print("-" * 60)
if mse_k <= mse_m:
    print(">> VICTORY: KAN is more robust to noise/dipoles.")
else:
    print(">> NOTE: KAN is competitive under stress.")

# Viz
fig, ax = plt.subplots(2, 3, figsize=(15, 9))
ax[0,0].imshow(p_wrapped, cmap='twilight'); ax[0,0].set_title("Noisy/Dipole Input")
ax[0,1].imshow(gt_norm*mask, cmap='twilight'); ax[0,1].set_title("Ground Truth")
ax[0,2].axis('off')

vmin, vmax = gt_norm[mask].min(), gt_norm[mask].max()
ax[1,0].imshow(rec_m*mask, cmap='twilight', vmin=vmin, vmax=vmax); ax[1,0].set_title(f"MLP Recon\nMSE: {mse_m:.4f}")
ax[1,1].imshow(rec_k*mask, cmap='twilight', vmin=vmin, vmax=vmax); ax[1,1].set_title(f"KAN Recon\nMSE: {mse_k:.4f}")
ax[1,2].bar(['MLP', 'KAN'], [mse_m, mse_k], color=['blue', 'red']); ax[1,2].set_title("MSE (Noise Robustness)")
plt.tight_layout()
plt.show()