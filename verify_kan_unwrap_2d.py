import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time

# ==========================================
# 1. ARCHITECTURES
# ==========================================

# --- KAN Implementation ---
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

# --- MLP Implementation (Baseline) ---
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

# ==========================================
# 2. SOLVERS (INTEGRATORS)
# ==========================================

def solver_naive(gx, gy):
    """Cumulative Sum Integration (Sensitive to local errors)"""
    recon = np.cumsum(gx, axis=1)
    col_accum = np.cumsum(gy[:, 1]) 
    for r in range(gx.shape[0]):
        recon[r, :] = recon[r, :] - recon[r, 1] + col_accum[r]
    return recon

def solver_fft(gx, gy):
    """Frankot-Chellappa Spectral Integration (Global Consistency)"""
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
# 3. DATA & UTILS
# ==========================================

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def generate_data(size=64):
    x = np.linspace(-3, 3, size)
    y = np.linspace(-3, 3, size)
    X, Y = np.meshgrid(x, y)
    # Complex surface with hills and valleys
    Z = 3*(1-X)**2 * np.exp(-(X**2) - (Y+1)**2) - 10*(X/5 - X**3 - Y**5)*np.exp(-X**2-Y**2)
    Z_true = Z * 2.5 # Scale amplitude
    Z_wrapped = np.angle(np.exp(1j * Z_true))
    return Z_true, Z_wrapped

def extract_patches(img_wrapped, img_true):
    H, W = img_wrapped.shape
    patches, targets = [], []
    for r in range(1, H-1):
        for c in range(1, W-1):
            patch = img_wrapped[r-1:r+2, c-1:c+2].flatten()
            gx = img_true[r, c+1] - img_true[r, c]
            gy = img_true[r+1, c] - img_true[r, c]
            patches.append(patch)
            targets.append([gx, gy])
    return np.array(patches), np.array(targets)

# ==========================================
# 4. MAIN EXECUTION
# ==========================================

print("="*60)
print("KAN vs MLP: MRI PHASE UNWRAPPING VIABILITY TEST")
print("="*60)

# 1. Setup Data
Z_true, Z_wrapped = generate_data(size=50)
X_np, Y_np = extract_patches(Z_wrapped, Z_true)

X_t = torch.tensor(X_np, dtype=torch.float32) / np.pi 
Y_t = torch.tensor(Y_np, dtype=torch.float32)

train_size = int(0.85 * len(X_t))
X_train, Y_train = X_t[:train_size], Y_t[:train_size]
X_test, Y_test = X_t[train_size:], Y_t[train_size:]

print(f"Data Generated. Image Size: {Z_true.shape}")
print(f"Training Samples: {len(X_train)} | Test Samples: {len(X_test)}")

# 2. Setup Models
# KAN: [9 inputs -> 16 hidden -> 2 outputs]
model_kan = SimpleKAN([9, 16, 2], grid_size=10)
# MLP: [9 inputs -> 64 hidden -> 64 hidden -> 2 outputs]
model_mlp = SimpleMLP(9, 64, 2)

p_kan = count_params(model_kan)
p_mlp = count_params(model_mlp)

print(f"\nModel Complexity:")
print(f" > KAN Parameters: {p_kan}")
print(f" > MLP Parameters: {p_mlp}")
print(f" > Ratio: KAN is {p_kan/p_mlp:.2f}x size of MLP")

# 3. Training Loop
opt_kan = optim.AdamW(model_kan.parameters(), lr=0.005)
opt_mlp = optim.Adam(model_mlp.parameters(), lr=0.002)
criterion = nn.MSELoss()

epochs = 600
print(f"\nStarting Training ({epochs} epochs)...")
start_time = time.time()

for epoch in range(1, epochs + 1):
    # KAN Step
    opt_kan.zero_grad()
    loss_kan = criterion(model_kan(X_train), Y_train)
    loss_kan.backward()
    opt_kan.step()

    # MLP Step
    opt_mlp.zero_grad()
    loss_mlp = criterion(model_mlp(X_train), Y_train)
    loss_mlp.backward()
    opt_mlp.step()

    if epoch % 100 == 0 or epoch == 1:
        elapsed = time.time() - start_time
        print(f" Epoch {epoch:03d} | Time: {elapsed:.1f}s | KAN Loss: {loss_kan.item():.5f} | MLP Loss: {loss_mlp.item():.5f}")

total_time = time.time() - start_time
print(f"Training Complete in {total_time:.2f}s")

# 4. Inference (Full Image)
print("\nRunning Inference on Full Image...")
H, W = Z_wrapped.shape
gx_kan, gy_kan = np.zeros((H,W)), np.zeros((H,W))
gx_mlp, gy_mlp = np.zeros((H,W)), np.zeros((H,W))

model_kan.eval()
model_mlp.eval()

with torch.no_grad():
    for r in range(1, H-1):
        for c in range(1, W-1):
            patch = Z_wrapped[r-1:r+2, c-1:c+2].flatten()
            inp = torch.tensor(patch, dtype=torch.float32).unsqueeze(0) / np.pi
            
            # KAN Pred
            out_k = model_kan(inp).numpy()[0]
            gx_kan[r,c], gy_kan[r,c] = out_k[0], out_k[1]
            
            # MLP Pred
            out_m = model_mlp(inp).numpy()[0]
            gx_mlp[r,c], gy_mlp[r,c] = out_m[0], out_m[1]

# 5. Solvers
print("Applying Solvers (Naive vs FFT)...")

# Apply Solvers to KAN
kan_naive = solver_naive(gx_kan, gy_kan)
kan_fft   = solver_fft(gx_kan, gy_kan)

# Apply Solvers to MLP
mlp_naive = solver_naive(gx_mlp, gy_mlp)
mlp_fft   = solver_fft(gx_mlp, gy_mlp)

# 6. Metrics & Reporting
def calc_mse(pred, target):
    # Crop borders to avoid artifacts
    return np.mean((pred[5:-5, 5:-5] - target[5:-5, 5:-5])**2)

# Normalize Reconstructions (Mean Center)
gt = Z_true - np.mean(Z_true)
kan_naive -= np.mean(kan_naive)
kan_fft   -= np.mean(kan_fft)
mlp_naive -= np.mean(mlp_naive)
mlp_fft   -= np.mean(mlp_fft)

mse_k_n = calc_mse(kan_naive, gt)
mse_k_f = calc_mse(kan_fft, gt)
mse_m_n = calc_mse(mlp_naive, gt)
mse_m_f = calc_mse(mlp_fft, gt)

print("\n" + "="*60)
print("FINAL SCIENTIFIC REPORT")
print("="*60)
print(f"{'Method':<20} | {'Solver':<10} | {'MSE (Lower is better)':<20}")
print("-" * 56)
print(f"{'MLP (Baseline)':<20} | {'Naive':<10} | {mse_m_n:.5f}")
print(f"{'MLP (Baseline)':<20} | {'FFT':<10}   | {mse_m_f:.5f}")
print("-" * 56)
print(f"{'KAN (Proposed)':<20} | {'Naive':<10} | {mse_k_n:.5f}")
print(f"{'KAN (Proposed)':<20} | {'FFT':<10}   | {mse_k_f:.5f}")
print("-" * 56)

winner = "KAN + FFT" if mse_k_f < mse_m_f else "MLP + FFT"
print(f">> BEST PERFORMANCE: {winner}")

# 7. Visualization
fig, ax = plt.subplots(2, 4, figsize=(16, 8))

# Row 1: MLP
ax[0,0].imshow(Z_true, cmap='twilight'); ax[0,0].set_title("Ground Truth")
ax[0,1].imshow(gx_mlp, cmap='coolwarm'); ax[0,1].set_title("MLP Gradient X")
ax[0,2].imshow(mlp_naive, cmap='twilight'); ax[0,2].set_title(f"MLP Naive\nMSE: {mse_m_n:.4f}")
ax[0,3].imshow(mlp_fft, cmap='twilight'); ax[0,3].set_title(f"MLP FFT\nMSE: {mse_m_f:.4f}")

# Row 2: KAN
ax[1,0].imshow(Z_wrapped, cmap='twilight'); ax[1,0].set_title("Input Wrapped")
ax[1,1].imshow(gx_kan, cmap='coolwarm'); ax[1,1].set_title("KAN Gradient X")
ax[1,2].imshow(kan_naive, cmap='twilight'); ax[1,2].set_title(f"KAN Naive\nMSE: {mse_k_n:.4f}")
ax[1,3].imshow(kan_fft, cmap='twilight'); ax[1,3].set_title(f"KAN FFT\nMSE: {mse_k_f:.4f}")

plt.tight_layout()
plt.show()