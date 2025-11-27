import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
import copy

# ==========================================
# 1. ARCHITECTURES
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

# ==========================================
# 2. SOLVERS (FFT Only - The Winner)
# ==========================================
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
# 3. DATA & UTILS
# ==========================================
def generate_data(size=64):
    x = np.linspace(-3, 3, size)
    y = np.linspace(-3, 3, size)
    X, Y = np.meshgrid(x, y)
    Z = 3*(1-X)**2 * np.exp(-(X**2) - (Y+1)**2) - 10*(X/5 - X**3 - Y**5)*np.exp(-X**2-Y**2)
    Z_true = Z * 2.5 
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

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ==========================================
# 4. MAIN EXECUTION
# ==========================================
print("="*60)
print("MRI PHASE UNWRAPPING: MLP vs KAN vs HYBRID-KAN")
print("="*60)

Z_true, Z_wrapped = generate_data(size=50)
X_np, Y_np = extract_patches(Z_wrapped, Z_true)
X_t = torch.tensor(X_np, dtype=torch.float32) / np.pi 
Y_t = torch.tensor(Y_np, dtype=torch.float32)

train_size = int(0.85 * len(X_t))
X_train, Y_train = X_t[:train_size], Y_t[:train_size]
X_test, Y_test = X_t[train_size:], Y_t[train_size:]

# --- Define Models ---
model_mlp = SimpleMLP(9, 64, 2)
model_kan_mse = SimpleKAN([9, 16, 2], grid_size=10)
model_kan_hybrid = copy.deepcopy(model_kan_mse) # Same start

# --- Training Setup ---
opt_mlp = optim.Adam(model_mlp.parameters(), lr=0.002)
opt_kan_mse = optim.AdamW(model_kan_mse.parameters(), lr=0.005)
opt_kan_hybrid = optim.AdamW(model_kan_hybrid.parameters(), lr=0.005)

criterion = nn.MSELoss()
epochs = 600

print(f"Training 3 Models ({epochs} epochs)...")
print(f"1. MLP (Baseline)")
print(f"2. KAN (Standard MSE)")
print(f"3. KAN (Hybrid: MSE + L1 Sparsity)")

start_time = time.time()

for epoch in range(1, epochs + 1):
    # 1. MLP
    opt_mlp.zero_grad()
    pred_mlp = model_mlp(X_train)
    loss_mlp = criterion(pred_mlp, Y_train)
    loss_mlp.backward()
    opt_mlp.step()

    # 2. KAN (MSE)
    opt_kan_mse.zero_grad()
    pred_k1 = model_kan_mse(X_train)
    loss_k1 = criterion(pred_k1, Y_train)
    loss_k1.backward()
    opt_kan_mse.step()

    # 3. KAN (Hybrid / L1 Sparsity)
    opt_kan_hybrid.zero_grad()
    pred_k2 = model_kan_hybrid(X_train)
    mse_part = criterion(pred_k2, Y_train)
    
    # HYBRID LOSS: Penalize the absolute value of gradients (Sparsity)
    # This acts like TV Denoising, forcing small noise -> 0
    l1_part = torch.mean(torch.abs(pred_k2)) 
    loss_k2 = mse_part + (0.01 * l1_part) # Lambda = 0.01
    
    loss_k2.backward()
    opt_kan_hybrid.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}: MLP {loss_mlp.item():.4f} | KAN-MSE {loss_k1.item():.4f} | KAN-Hybrid {loss_k2.item():.4f}")

# --- Inference ---
print("\nRunning Inference...")
H, W = Z_wrapped.shape
gx_m, gy_m = np.zeros((H,W)), np.zeros((H,W))
gx_k1, gy_k1 = np.zeros((H,W)), np.zeros((H,W))
gx_k2, gy_k2 = np.zeros((H,W)), np.zeros((H,W))

model_mlp.eval(); model_kan_mse.eval(); model_kan_hybrid.eval()

with torch.no_grad():
    for r in range(1, H-1):
        for c in range(1, W-1):
            patch = Z_wrapped[r-1:r+2, c-1:c+2].flatten()
            inp = torch.tensor(patch, dtype=torch.float32).unsqueeze(0) / np.pi
            
            o_m = model_mlp(inp).numpy()[0]
            gx_m[r,c], gy_m[r,c] = o_m[0], o_m[1]
            
            o_k1 = model_kan_mse(inp).numpy()[0]
            gx_k1[r,c], gy_k1[r,c] = o_k1[0], o_k1[1]

            o_k2 = model_kan_hybrid(inp).numpy()[0]
            gx_k2[r,c], gy_k2[r,c] = o_k2[0], o_k2[1]

# --- Solvers (FFT) ---
rec_m = solver_fft(gx_m, gy_m)
rec_k1 = solver_fft(gx_k1, gy_k1)
rec_k2 = solver_fft(gx_k2, gy_k2)

# --- Metrics ---
gt = Z_true - np.mean(Z_true)
rec_m -= np.mean(rec_m)
rec_k1 -= np.mean(rec_k1)
rec_k2 -= np.mean(rec_k2)

def get_mse(pred, targ): return np.mean((pred[5:-5, 5:-5] - targ[5:-5, 5:-5])**2)

mse_m = get_mse(rec_m, gt)
mse_k1 = get_mse(rec_k1, gt)
mse_k2 = get_mse(rec_k2, gt)

print("\n" + "="*60)
print("FINAL RESULTS (FFT RECONSTRUCTION)")
print("="*60)
print(f"1. MLP Baseline MSE:       {mse_m:.5f}")
print(f"2. KAN (MSE) MSE:          {mse_k1:.5f}")
print(f"3. KAN (Hybrid L1) MSE:    {mse_k2:.5f}")
print("-"*60)
if mse_k2 < mse_k1:
    print(">> SUCCESS: Hybrid Loss improved KAN performance.")
if mse_k2 < mse_m:
    print(">> VICTORY: Hybrid KAN outperformed MLP.")
else:
    print(">> NOTE: KAN is competitive/comparable to MLP.")

# --- Plotting ---
fig, ax = plt.subplots(2, 4, figsize=(16, 8))

# Top Row: Gradients (X-direction)
ax[0,0].imshow(Z_true, cmap='twilight'); ax[0,0].set_title("Ground Truth")
ax[0,1].imshow(gx_m, cmap='coolwarm'); ax[0,1].set_title("MLP Gradient X")
ax[0,2].imshow(gx_k1, cmap='coolwarm'); ax[0,2].set_title("KAN (MSE) Grad X\n(Note the noise)")
ax[0,3].imshow(gx_k2, cmap='coolwarm'); ax[0,3].set_title("KAN (Hybrid) Grad X\n(Cleaner?)")

# Bottom Row: Reconstructions
vmin, vmax = gt.min(), gt.max()
ax[1,0].imshow(Z_wrapped, cmap='twilight'); ax[1,0].set_title("Input Wrapped")
ax[1,1].imshow(rec_m, cmap='twilight', vmin=vmin, vmax=vmax); ax[1,1].set_title(f"MLP Recon\nMSE: {mse_m:.4f}")
ax[1,2].imshow(rec_k1, cmap='twilight', vmin=vmin, vmax=vmax); ax[1,2].set_title(f"KAN (MSE) Recon\nMSE: {mse_k1:.4f}")
ax[1,3].imshow(rec_k2, cmap='twilight', vmin=vmin, vmax=vmax); ax[1,3].set_title(f"KAN (Hybrid) Recon\nMSE: {mse_k2:.4f}")

plt.tight_layout()
plt.show()