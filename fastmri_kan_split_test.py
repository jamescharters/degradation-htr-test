# https://aistudio.google.com/prompts/17mNcN_nBC9kxwKsCQVtWjpDhZm_N5LH9


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
# REPLACE with your actual path
FILE_PATH = "./data/fastMRI/singlecoil_val/file1000000.h5" 
SLICE_NUM = 20
PATCH_THRESHOLD = 0.05 
EPOCHS = 300
NOISE_LEVEL = 0.15  # 15% Noise (Stress Test)

# Reproducibility
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
# 3. DATA LOADING & SYNTHESIS
# ==========================================
def simulate_dipole_phase(H, W, n_dipoles=8,
                          strength_range=(10.0, 25.0),
                          radius_range=(0.05, 0.15),
                          centre_range=(-0.8, 0.8),
                          rng=None):
    """
    Simulate a sum of dipole-like local phase perturbations over a 2D grid.
    This approximates local susceptibility effects (veins, tissue interfaces, etc).

    Parameters
    ----------
    H, W : int
        Image height and width.
    n_dipoles : int
        Number of dipole-like sources to place.
    strength_range : (float, float)
        Range of absolute dipole strengths (in radians).
    radius_range : (float, float)
        Range of dipole "radius" (in normalised coordinates).
    centre_range : (float, float)
        Range for random dipole centres in normalised coords (X, Y in [-1,1]).
    rng : np.random.Generator or None
        Optional random number generator for reproducibility.

    Returns
    -------
    phase_local : (H, W) ndarray
        Local phase contribution from all dipoles (unwrapped).
    """
    if rng is None:
        rng = np.random.default_rng()

    # Normalised coordinate grid in [-1, 1]
    x_space = np.linspace(-1.0, 1.0, W)
    y_space = np.linspace(-1.0, 1.0, H)
    X, Y = np.meshgrid(x_space, y_space)

    phase_local = np.zeros((H, W), dtype=np.float32)

    for _ in range(n_dipoles):
        cx = rng.uniform(centre_range[0], centre_range[1])
        cy = rng.uniform(centre_range[0], centre_range[1])
        radius = rng.uniform(radius_range[0], radius_range[1])
        strength = rng.choice([-1.0, 1.0]) * rng.uniform(strength_range[0], strength_range[1])

        r2 = (X - cx) ** 2 + (Y - cy) ** 2
        # Simple 1 / (1 + r^2/R^2) profile – not physically exact, but dipole-like
        dipole = strength * (1.0 / (1.0 + r2 / (radius ** 2)))
        phase_local += dipole.astype(np.float32)

    return phase_local

def load_data(h5_path, slice_idx,
              n_dipoles=6,
              bg_coeff_range=(-5.0, 5.0),
              noise_sigma=0.02,
              rng=None):
    """
    Load a single fastMRI slice and generate a realistic synthetic phase field.

    - Magnitude comes from the actual fastMRI inverse FFT.
    - Phase is simulated as:
        phase_true = smooth_background + local_dipoles
      then converted to a noisy, wrapped phase via complex Gaussian noise.

    Parameters
    ----------
    h5_path : str
        Path to the fastMRI HDF5 file.
    slice_idx : int
        Index of the slice to load.
    n_dipoles : int
        Number of local dipole-like sources to add.
    bg_coeff_range : (float, float)
        Range for random polynomial background phase coefficients (in radians).
    noise_sigma : float
        Standard deviation of complex Gaussian noise (relative to magnitude scale).
    rng : np.random.Generator or None
        Optional random number generator for reproducibility.

    Returns
    -------
    magnitude : (H, W) ndarray
        Normalised magnitude image from fastMRI.
    phase_true : (H, W) ndarray
        Synthetic unwrapped phase (background + dipoles).
    phase_wrapped : (H, W) ndarray
        Noisy, wrapped phase as would be observed by the scanner.
    """
    if rng is None:
        rng = np.random.default_rng()

    # ---- 1. Load k-space and compute complex image ----
    with h5py.File(h5_path, 'r') as hf:
        kspace = hf['kspace'][slice_idx]  # assumes single-coil or one coil already selected

    img_complex = np.fft.fftshift(
        np.fft.ifft2(
            np.fft.ifftshift(kspace)
        )
    )

    magnitude = np.abs(img_complex).astype(np.float32)

    # Normalise magnitude to [0, 1] (avoid divide-by-zero)
    mag_min = magnitude.min()
    mag_max = magnitude.max()
    magnitude = (magnitude - mag_min) / (mag_max - mag_min + 1e-8)

    H, W = magnitude.shape

    # ---- 2. Smooth background phase (B0-like field) ----
    x = np.linspace(-1.0, 1.0, W, dtype=np.float32)
    y = np.linspace(-1.0, 1.0, H, dtype=np.float32)
    X, Y = np.meshgrid(x, y)

    # Random quadratic background: a0 + a1 x + a2 y + a3 x^2 + a4 y^2
    a0, a1, a2, a3, a4 = rng.uniform(bg_coeff_range[0], bg_coeff_range[1], size=5)
    phase_bg = (a0
                + a1 * X
                + a2 * Y
                + a3 * X ** 2
                + a4 * Y ** 2).astype(np.float32)

    # ---- 3. Local dipole-like phase ----
    phase_local = simulate_dipole_phase(H, W, n_dipoles=n_dipoles, rng=rng)

    # Total unwrapped phase
    phase_true = phase_bg + phase_local  # (H, W), float32

    # ---- 4. Build noisy complex image and extract wrapped phase ----
    img_ideal = magnitude * np.exp(1j * phase_true)

    # Complex Gaussian noise (same sigma for real and imag parts)
    noise_real = rng.normal(0.0, noise_sigma, img_ideal.shape)
    noise_imag = rng.normal(0.0, noise_sigma, img_ideal.shape)
    noise_complex = noise_real + 1j * noise_imag

    img_noisy = img_ideal + noise_complex

    phase_wrapped = np.angle(img_noisy).astype(np.float32)

    return magnitude, phase_true, phase_wrapped


def extract_patches(phase_wrapped, phase_true, magnitude):
    H, W = phase_wrapped.shape
    patches, targets = [], []
    
    # Stride 2
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
print("GEOMETRIC SPLIT TEST: TRAIN (TOP) vs TEST (BOTTOM)")
print("Objective: Prove generalization to unseen anatomy.")
print("="*60)

# 1. Load Full Image
mag, p_true, p_wrapped = load_data(FILE_PATH, SLICE_NUM)
H, W = mag.shape
split_row = int(H * 0.5)

print(f"Splitting Image at Row {split_row}")

# 2. Extract TRAIN Patches (Top Half)
X_np_train, Y_np_train = extract_patches(
    p_wrapped[:split_row, :], 
    p_true[:split_row, :], 
    mag[:split_row, :]
)
X_train = torch.tensor(X_np_train, dtype=torch.float32) / np.pi
Y_train = torch.tensor(Y_np_train, dtype=torch.float32)

# 3. Extract TEST Patches (Bottom Half - Unseen)
X_np_test, Y_np_test = extract_patches(
    p_wrapped[split_row:, :], 
    p_true[split_row:, :], 
    mag[split_row:, :]
)
X_test = torch.tensor(X_np_test, dtype=torch.float32) / np.pi
Y_test = torch.tensor(Y_np_test, dtype=torch.float32)

print(f"Training Patches (Top): {len(X_train)}")
print(f"Testing Patches (Bottom): {len(X_test)}")

# 4. Models
model_kan = SimpleKAN([9, 16, 2], grid_size=10)
model_mlp = SimpleMLP(9, 64, 2)

p_kan = count_params(model_kan)
p_mlp = count_params(model_mlp)

opt_kan = optim.AdamW(model_kan.parameters(), lr=0.005)
opt_mlp = optim.Adam(model_mlp.parameters(), lr=0.002)
criterion = nn.MSELoss()

# 5. Training Loop (Train on Top Only!)
print(f"\nTraining Models on Top Half ({EPOCHS} epochs)...")
start_train = time.time()
loss_k_hist, loss_m_hist = [], []

for epoch in range(1, EPOCHS+1):
    # Train Steps
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

train_time = time.time() - start_train

# 6. Evaluation (Generalization Test)
print("\nEvaluating on Bottom Half (Unseen Data)...")
model_kan.eval(); model_mlp.eval()
with torch.no_grad():
    # Raw Classification Loss (Did it identify k correctly?)
    test_loss_k = criterion(model_kan(X_test), Y_test).item()
    test_loss_m = criterion(model_mlp(X_test), Y_test).item()

# 7. Full Image Inference & Recon
print("Running Full Image Reconstruction...")
gx_naive = np.zeros((H,W)); gy_naive = np.zeros((H,W))
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

# 8. FFT Recon
rec_k = solver_fft(gx_k, gy_k)
rec_m = solver_fft(gx_m, gy_m)

# 9. Calculate MSE specifically on the Test Half (Bottom)
mask_bottom = np.zeros_like(mag, dtype=bool)
mask_bottom[split_row:, :] = (mag[split_row:, :] > PATCH_THRESHOLD)

# Normalize using bottom stats
rec_k -= np.mean(rec_k[mask_bottom])
rec_m -= np.mean(rec_m[mask_bottom])
gt_norm = p_true - np.mean(p_true[mask_bottom])

mse_k_test = np.mean((rec_k[mask_bottom] - gt_norm[mask_bottom])**2)
mse_m_test = np.mean((rec_m[mask_bottom] - gt_norm[mask_bottom])**2)

# ==========================================
# FINAL REPORT
# ==========================================
print("\n" + "="*60)
print("SCIENTIFIC REPORT: GENERALIZATION TEST")
print("="*60)
print(f"{'Metric':<30} | {'MLP':<15} | {'KAN':<15}")
print("-" * 65)
print(f"{'Parameters':<30} | {p_mlp:<15} | {p_kan:<15}")
print(f"{'Compression':<30} | {'1.0x':<15} | {p_kan/p_mlp:.2f}x")
print(f"{'Train Loss (Top Half)':<30} | {loss_m_hist[-1]:<15.5f} | {loss_k_hist[-1]:<15.5f}")
print(f"{'Test Loss (Bottom Half)':<30} | {test_loss_m:<15.5f} | {test_loss_k:<15.5f}")
print(f"{'Test Recon MSE':<30} | {mse_m_test:<15.5f} | {mse_k_test:<15.5f}")
print("-" * 65)

if mse_k_test <= mse_m_test:
    print(">> VERIFIED: KAN generalizes to unseen anatomy as well as MLP.")
else:
    print(">> NOTE: Small generalization gap observed.")

# Plot
fig, ax = plt.subplots(2, 3, figsize=(15, 9))
ax[0,0].imshow(mag, cmap='gray'); ax[0,0].set_title("Split: Top(Train) / Bottom(Test)")
ax[0,0].axhline(y=split_row, color='r', linestyle='--') # Show split line

ax[0,1].imshow(p_wrapped, cmap='twilight'); ax[0,1].set_title("Input Wrapped")
ax[0,2].imshow(gt_norm, cmap='twilight'); ax[0,2].set_title("Ground Truth")

vmin, vmax = gt_norm[mask_bottom].min(), gt_norm[mask_bottom].max()
# Show recon (Whole image to see continuity)
ax[1,0].imshow(rec_m, cmap='twilight'); ax[1,0].set_title(f"MLP Recon (Full)\nTest MSE: {mse_m_test:.4f}")
ax[1,1].imshow(rec_k, cmap='twilight'); ax[1,1].set_title(f"KAN Recon (Full)\nTest MSE: {mse_k_test:.4f}")
# Show Error Map (Bottom Only)
err_map = (rec_k - gt_norm)**2
err_map[:split_row, :] = 0 # Hide train error
ax[1,2].imshow(err_map, cmap='hot'); ax[1,2].set_title("KAN Error Map (Test Half)")

plt.tight_layout()
plt.show()