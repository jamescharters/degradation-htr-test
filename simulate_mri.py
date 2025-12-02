import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time

# ==========================================
# 1. FAST MATRIX KAN (No Recursion, Stable)
# ==========================================
class MatrixKANLayer(nn.Module):
    def __init__(self, in_feat, out_feat, grid_size=20, scale=0.1):
        super().__init__()
        self.grid_size = grid_size
        
        # Grid [-1, 1]
        h = 2.0 / grid_size
        grid = torch.arange(-1, grid_size + 2) * h - 1.0 
        self.register_buffer("grid", grid) 
        
        # Coeffs: (Out, In, Grid+1)
        self.coeffs = nn.Parameter(torch.randn(out_feat, in_feat, grid_size + 1) * scale)
        self.base_w = nn.Parameter(torch.randn(out_feat, in_feat) * 0.1)

    def forward(self, x):
        x = torch.clamp(x, -1, 1)
        
        # Linear Base
        base = F.linear(F.silu(x), self.base_w)
        
        # B-Spline Interpolation (Linear Basis for Stability/Speed)
        # Map x to [0, grid_size]
        x_grid = (x + 1) / 2 * self.grid_size
        idx = torch.floor(x_grid).long().clamp(0, self.grid_size - 1)
        t = x_grid - idx # Fractional part (0..1)
        
        # Vectorized Lookup
        # We need to gather coeffs for each input dimension
        # Shape: (Batch, In) -> (Batch, Out)
        y = torch.zeros_like(base)
        
        for i in range(x.shape[1]): # Loop over input dims (only 2, very fast)
            id_i = idx[:, i]   # (Batch,)
            t_i = t[:, i].unsqueeze(1) # (Batch, 1)
            
            # Gather Coeffs: (Out, Grid)
            c_layer = self.coeffs[:, i, :] 
            
            # Left and Right control points
            c_left = c_layer[:, id_i].t()   # (Batch, Out)
            c_right = c_layer[:, (id_i+1).clamp(0, self.grid_size)].t()
            
            # Interpolate
            y += c_left * (1 - t_i) + c_right * t_i
            
        return base + y

class FastKAN(nn.Module):
    def __init__(self):
        super().__init__()
        # Grid 32 is necessary for High Freq Phase
        self.l1 = MatrixKANLayer(2, 8, grid_size=64)
        self.l2 = MatrixKANLayer(8, 2, grid_size=64)
    def forward(self, x): return self.l2(self.l1(x))

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        # 4 Layers, Tanh (Fair comparison for Physics)
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 2)
        )
    def forward(self, x): return self.net(x)

# ==========================================
# 2. EXPERIMENT
# ==========================================
if __name__ == "__main__":
    if torch.backends.mps.is_available(): device = torch.device("mps")
    elif torch.cuda.is_available(): device = torch.device("cuda")
    else: device = torch.device("cpu")
    print(f"Device: {device}")

    # --- 1. GENERATE HIGH FREQUENCY DATA ---
    print("Generating High-Frequency Phase Wraps...")
    N = 256
    x = torch.linspace(-1, 1, N)
    X, Y = torch.meshgrid(x, x, indexing='xy')
    
    # FREQUENCY = 8.0 (Forces ~3 wraps from center to edge)
    # This guarantees the "Sawtooth" exists
    phase_true = 8.0 * (X**2 + Y**2) 
    
    # Create Complex Signal (Smooth Real/Imag components)
    mag = torch.ones_like(X)
    signal = mag * torch.exp(1j * phase_true)
    
    # Inputs/Targets
    inputs = torch.stack([X.flatten(), Y.flatten()], dim=1).to(device)
    targets = torch.stack([signal.real.flatten(), signal.imag.flatten()], dim=1).to(device)

    # --- 2. TRAIN ---
    mlp = MLP().to(device)
    kan = FastKAN().to(device)
    
    print(f"MLP Params: {sum(p.numel() for p in mlp.parameters())}")
    print(f"KAN Params: {sum(p.numel() for p in kan.parameters())}")

    def train(model, name):
        opt = optim.Adam(model.parameters(), lr=0.01)
        sched = optim.lr_scheduler.ExponentialLR(opt, gamma=0.995)
        loss_fn = nn.MSELoss()
        
        t0 = time.time()
        # 1000 Epochs to ensure convergence
        for i in range(1000):
            opt.zero_grad()
            pred = model(inputs)
            loss = loss_fn(pred, targets)
            loss.backward()
            opt.step()
            sched.step()
        print(f"{name} Final Loss: {loss.item():.6f} ({time.time()-t0:.1f}s)")
        return model

    train(mlp, "MLP")
    train(kan, "KAN")

    # --- 3. VISUALIZE ---
    inputs = inputs.cpu()
    mlp = mlp.cpu()
    kan = kan.cpu()
    
    with torch.no_grad():
        p_mlp = mlp(inputs)
        p_kan = kan(inputs)
    
    # Calculate Phase
    # The Model predicts (Real, Imag). We compute atan2.
    phase_gt = torch.angle(signal).flatten().cpu().numpy()
    phase_mlp = torch.atan2(p_mlp[:,1], p_mlp[:,0]).numpy()
    phase_kan = torch.atan2(p_kan[:,1], p_kan[:,0]).numpy()
    
    # Reshape for Image
    phase_gt_img = phase_gt.reshape(N, N)
    phase_mlp_img = phase_mlp.reshape(N, N)
    phase_kan_img = phase_kan.reshape(N, N)

    # PLOTTING
    fig, ax = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1: Images
    ax[0,0].imshow(phase_gt_img, cmap='twilight', vmin=-np.pi, vmax=np.pi)
    ax[0,0].set_title("Ground Truth (High Freq)")
    ax[0,1].imshow(phase_mlp_img, cmap='twilight', vmin=-np.pi, vmax=np.pi)
    ax[0,1].set_title("MLP (Note Blur rings)")
    ax[0,2].imshow(phase_kan_img, cmap='twilight', vmin=-np.pi, vmax=np.pi)
    ax[0,2].set_title("KAN (Sharper rings)")
    
    # Row 2: Cross Sections (The Proof)
    mid = N // 2
    
    # SLICE 1: Full View
    ax[1,0].plot(phase_gt_img[mid, :], 'k', linewidth=1, label="Truth")
    ax[1,0].plot(phase_mlp_img[mid, :], 'orange', alpha=0.7, label="MLP")
    ax[1,0].set_title("1D Cross Section (Full)")
    ax[1,0].legend()
    
    # SLICE 2: KAN Comparison
    ax[1,1].plot(phase_gt_img[mid, :], 'k', linewidth=1, label="Truth")
    ax[1,1].plot(phase_kan_img[mid, :], 'b', alpha=0.7, label="KAN")
    ax[1,1].set_title("KAN vs Truth")
    ax[1,1].legend()

    # SLICE 3: ZOOM IN ON THE JUMP
    # Find the first jump index roughly
    zoom_center = np.argmax(np.diff(phase_gt_img[mid, :])) # Find biggest jump
    start, end = zoom_center - 10, zoom_center + 10
    
    ax[1,2].plot(np.arange(start, end), phase_gt_img[mid, start:end], 'k.-', linewidth=2, label="Truth")
    ax[1,2].plot(np.arange(start, end), phase_mlp_img[mid, start:end], 'orange', label="MLP")
    ax[1,2].plot(np.arange(start, end), phase_kan_img[mid, start:end], 'b--', label="KAN")
    ax[1,2].set_title("ZOOM: The Vertical Jump")
    ax[1,2].legend()

    plt.tight_layout()
    plt.show()