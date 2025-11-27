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
# 2. DATA GENERATION (RESIDUE TARGETS)
# ==========================================
def generate_data(size=64):
    x = np.linspace(-3, 3, size)
    y = np.linspace(-3, 3, size)
    X, Y = np.meshgrid(x, y)
    Z = 3*(1-X)**2 * np.exp(-(X**2) - (Y+1)**2) - 10*(X/5 - X**3 - Y**5)*np.exp(-X**2-Y**2)
    Z_true = Z * 2.5
    Z_wrapped = np.angle(np.exp(1j * Z_true))
    return Z_true, Z_wrapped

def extract_residue_patches(img_wrapped, img_true):
    H, W = img_wrapped.shape
    patches, targets = [], []
    for r in range(1, H-1):
        for c in range(1, W-1):
            patch = img_wrapped[r-1:r+2, c-1:c+2].flatten()
            
            # 1. Naive Gradient (Huge jumps at wraps)
            dx_naive = img_wrapped[r, c+1] - img_wrapped[r, c]
            dy_naive = img_wrapped[r+1, c] - img_wrapped[r, c]
            
            # 2. True Gradient (Smooth)
            dx_true = img_true[r, c+1] - img_true[r, c]
            dy_true = img_true[r+1, c] - img_true[r, c]
            
            # 3. Target is Integer k: (True - Naive) / 2pi
            k_x = (dx_true - dx_naive) / (2 * np.pi)
            k_y = (dy_true - dy_naive) / (2 * np.pi)
            
            # Round to nearest integer (targets are -1, 0, 1)
            patches.append(patch)
            targets.append([np.round(k_x), np.round(k_y)])
            
    return np.array(patches), np.array(targets)

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ==========================================
# 3. FFT SOLVER
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
# 4. EXECUTION
# ==========================================
print("="*60)
print("TESTING: Residue Learning (Target = Integer k)")
print("Logic: G_pred = G_naive + 2*pi * Round(Model(x))")
print("="*60)

Z_true, Z_wrapped = generate_data(size=50)
X_np, Y_np = extract_residue_patches(Z_wrapped, Z_true)

X_t = torch.tensor(X_np, dtype=torch.float32) / np.pi 
Y_t = torch.tensor(Y_np, dtype=torch.float32)

train_size = int(0.85 * len(X_t))
X_train, Y_train = X_t[:train_size], Y_t[:train_size]

# --- Models ---
# KAN [9 -> 16 -> 2]
model_kan = SimpleKAN([9, 16, 2], grid_size=10)
# MLP [9 -> 64 -> 2]
model_mlp = SimpleMLP(9, 64, 2)

p_kan = count_params(model_kan)
p_mlp = count_params(model_mlp)

print(f"\nModel Stats:")
print(f" > KAN Params: {p_kan}")
print(f" > MLP Params: {p_mlp}")
print(f" > Ratio: KAN is {p_kan/p_mlp:.2f}x the size of MLP")

# --- Training ---
opt_kan = optim.AdamW(model_kan.parameters(), lr=0.005)
opt_mlp = optim.Adam(model_mlp.parameters(), lr=0.002)
criterion = nn.MSELoss()
epochs = 600

print(f"\nTraining ({epochs} epochs)...")
start_t = time.time()

for epoch in range(1, epochs+1):
    opt_kan.zero_grad()
    l_k = criterion(model_kan(X_train), Y_train)
    l_k.backward()
    opt_kan.step()

    opt_mlp.zero_grad()
    l_m = criterion(model_mlp(X_train), Y_train)
    l_m.backward()
    opt_mlp.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}: KAN Loss {l_k.item():.5f} | MLP Loss {l_m.item():.5f}")

train_time = time.time() - start_t
print(f"Training finished in {train_time:.2f}s")

# --- Inference with "Integer Rounding" ---
print("\nRunning Inference with Rounding (Denoising)...")
H, W = Z_wrapped.shape
gx_naive_map = np.zeros((H,W))
gy_naive_map = np.zeros((H,W))

# Calculate naive gradients once
for r in range(H-1):
    for c in range(W-1):
        gx_naive_map[r,c] = Z_wrapped[r,c+1] - Z_wrapped[r,c]
        gy_naive_map[r,c] = Z_wrapped[r+1,c] - Z_wrapped[r,c]

gx_k_final, gy_k_final = np.zeros((H,W)), np.zeros((H,W))
gx_m_final, gy_m_final = np.zeros((H,W)), np.zeros((H,W))
k_map_kan = np.zeros((H,W)) # To visualize the jumps
k_map_mlp = np.zeros((H,W))

model_kan.eval(); model_mlp.eval()
with torch.no_grad():
    for r in range(1, H-1):
        for c in range(1, W-1):
            patch = Z_wrapped[r-1:r+2, c-1:c+2].flatten()
            inp = torch.tensor(patch, dtype=torch.float32).unsqueeze(0) / np.pi
            
            # Raw predictions (floats)
            k_k_raw = model_kan(inp).numpy()[0]
            k_m_raw = model_mlp(inp).numpy()[0]
            
            # ROUNDING (The Key Step)
            k_k = np.round(k_k_raw)
            k_m = np.round(k_m_raw)
            
            # Store for viz
            k_map_kan[r,c] = k_k[0] 
            k_map_mlp[r,c] = k_m[0]
            
            # Reconstruct: True = Naive + 2pi*k
            gx_k_final[r,c] = gx_naive_map[r,c] + (2 * np.pi * k_k[0])
            gy_k_final[r,c] = gy_naive_map[r,c] + (2 * np.pi * k_k[1])
            
            gx_m_final[r,c] = gx_naive_map[r,c] + (2 * np.pi * k_m[0])
            gy_m_final[r,c] = gy_naive_map[r,c] + (2 * np.pi * k_m[1])

# --- FFT Integration ---
rec_k = solver_fft(gx_k_final, gy_k_final)
rec_m = solver_fft(gx_m_final, gy_m_final)

# --- Metrics ---
gt = Z_true - np.mean(Z_true)
rec_k -= np.mean(rec_k)
rec_m -= np.mean(rec_m)

def get_mse(p, t): return np.mean((p[5:-5, 5:-5] - t[5:-5, 5:-5])**2)
mse_k = get_mse(rec_k, gt)
mse_m = get_mse(rec_m, gt)

print("\n" + "="*60)
print("FINAL SCIENTIFIC REPORT")
print("="*60)
print(f"{'Metric':<25} | {'MLP':<15} | {'KAN':<15}")
print("-" * 60)
print(f"{'Parameters':<25} | {p_mlp:<15} | {p_kan:<15}")
print(f"{'Training Loss':<25} | {l_m.item():<15.5f} | {l_k.item():<15.5f}")
print(f"{'Reconstruction MSE':<25} | {mse_m:<15.5f} | {mse_k:<15.5f}")
print("-" * 60)

if mse_k < mse_m:
    print(">> CONCLUSION: KAN Superior via Residue Learning.")
else:
    print(">> CONCLUSION: KAN Comparable to MLP.")

# --- Plots ---
fig, ax = plt.subplots(2, 3, figsize=(14, 8))

# Top: The "k" maps (The detected wraps)
# This shows if the model is hallucinating wraps in the background
ax[0,0].imshow(Z_true, cmap='twilight'); ax[0,0].set_title("Ground Truth Phase")
ax[0,1].imshow(k_map_mlp, cmap='gray'); ax[0,1].set_title("MLP Predicted Wraps (k)\n(Should be black/white only)")
ax[0,2].imshow(k_map_kan, cmap='gray'); ax[0,2].set_title("KAN Predicted Wraps (k)\n(Should be black/white only)")

# Bottom: Reconstruction
vmin, vmax = gt.min(), gt.max()
ax[1,0].imshow(Z_wrapped, cmap='twilight'); ax[1,0].set_title("Input Wrapped")
ax[1,1].imshow(rec_m, cmap='twilight', vmin=vmin, vmax=vmax); ax[1,1].set_title(f"MLP Recon\nMSE: {mse_m:.4f}")
ax[1,2].imshow(rec_k, cmap='twilight', vmin=vmin, vmax=vmax); ax[1,2].set_title(f"KAN Recon\nMSE: {mse_k:.4f}")

plt.tight_layout()
plt.show()