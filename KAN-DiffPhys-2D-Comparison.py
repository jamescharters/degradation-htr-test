import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import time

# ---------------------------------------------
# 1. SETUP & UTILS
# ---------------------------------------------
def get_device():
    if torch.backends.mps.is_available(): return torch.device("mps")
    elif torch.cuda.is_available(): return torch.device("cuda")
    else: return torch.device("cpu")
device = get_device()
print(f"Running on: {device}")

# ==========================================
# 2. PHYSICS & DATA GENERATION
# ==========================================
def solve_diffusion(k_map, direction='left-right', steps=50):
    H, W = k_map.shape
    u = torch.zeros((H, W), device=device)
    if direction == 'left-right': u[:, :W//2] = 1.0
    elif direction == 'top-bottom': u[:H//2, :] = 1.0
    dt = 0.01
    history = []
    for _ in range(steps):
        u_up    = torch.roll(u, -1, dims=0); u_up[-1, :] = u[-1, :]
        u_down  = torch.roll(u, 1, dims=0);  u_down[0, :] = u[0, :]
        u_left  = torch.roll(u, -1, dims=1); u_left[:, -1] = u[:, -1]
        u_right = torch.roll(u, 1, dims=1);  u_right[:, 0] = u[:, 0]
        laplacian = (u_up + u_down + u_left + u_right - 4*u)
        u = u + dt * (k_map * laplacian)
        history.append(u.clone())
    return torch.stack(history)

RES = 32
DIRECTIONS = ['left-right', 'top-bottom']

def create_ground_truth(res):
    k_true = torch.ones((res, res), device=device) * 1.0
    mid = res // 2
    k_true[mid-5:mid+5, mid-5:mid+5] = 0.1 
    return k_true

K_TRUE = create_ground_truth(RES)
OBSERVATIONS = {}
print("Simulating ground truth data...")
for direction in DIRECTIONS:
    traj = solve_diffusion(K_TRUE, direction=direction, steps=100)
    OBSERVATIONS[direction] = traj + torch.randn_like(traj) * 0.02

# ==========================================
# 3. THE FOUR ARCHITECTURES
# ==========================================

# --- A: MLP-PINN (The "Strawman") ---
class MLP_PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.u_net = nn.Sequential(nn.Linear(2, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))
        self.k_net = nn.Sequential(nn.Linear(2, 64), nn.Tanh(), nn.Linear(64, 1))
    def get_k_map(self, res):
        coords = torch.stack(torch.meshgrid(torch.linspace(-1,1,res), torch.linspace(-1,1,res), indexing='ij'), -1).view(-1,2).to(device)
        return torch.sigmoid(self.k_net(coords)).view(res,res) * 1.99 + 0.01

# --- B: MLP-DiffPhys ---
class MLP_DiffPhys(nn.Module):
    def __init__(self):
        super().__init__()
        self.k_net = nn.Sequential(nn.Linear(2, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))
    def get_k_map(self, res):
        coords = torch.stack(torch.meshgrid(torch.linspace(-1,1,res), torch.linspace(-1,1,res), indexing='ij'), -1).view(-1,2).to(device)
        return torch.sigmoid(self.k_net(coords)).view(res,res) * 1.99 + 0.01

# --- C: KAN Layer & KAN-PINN ---
class SpatialRBF(nn.Module):
    def __init__(self, out_features, grid_res=25):
        super().__init__()
        x = torch.linspace(-1, 1, grid_res, device=device)
        self.grid = torch.stack(torch.meshgrid(x, x, indexing='ij'), dim=-1).view(-1, 2)
        self.spline = nn.Parameter(torch.randn(grid_res**2, out_features) * 0.1)
        self.base = nn.Parameter(torch.randn(out_features, 2) * 0.1)
    def forward(self, x):
        base = F.linear(x, self.base)
        dist_sq = torch.sum((x.unsqueeze(1) - self.grid.unsqueeze(0))**2, dim=2)
        return base + torch.matmul(torch.exp(-dist_sq * 100.0), self.spline)

class KAN_PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.u_net = nn.Sequential(SpatialRBF(16), nn.Linear(16, 1))
        self.k_net = nn.Sequential(SpatialRBF(16), nn.Linear(16, 1))
    def get_k_map(self, res):
        coords = torch.stack(torch.meshgrid(torch.linspace(-1,1,res), torch.linspace(-1,1,res), indexing='ij'), -1).view(-1,2).to(device)
        return torch.sigmoid(self.k_net(coords)).view(res,res) * 1.99 + 0.01

# --- D: KAN-DiffPhys (Your Novelty) ---
class KAN_DiffPhys(nn.Module):
    def __init__(self):
        super().__init__()
        self.k_net = nn.Sequential(SpatialRBF(16), nn.Linear(16, 1))
    def get_k_map(self, res):
        coords = torch.stack(torch.meshgrid(torch.linspace(-1,1,res), torch.linspace(-1,1,res), indexing='ij'), -1).view(-1,2).to(device)
        return torch.sigmoid(self.k_net(coords)).view(res,res) * 1.99 + 0.01

# ==========================================
# 4. TRAINING FUNCTIONS
# ==========================================

def train_diffphys(model, label, epochs=1001, lr=0.01):
    print(f"\n--- Training {label} ---")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)
    for epoch in range(epochs):
        optimizer.zero_grad()
        k_pred = model.get_k_map(RES)
        
        loss = 0
        for d in DIRECTIONS:
            sim_traj = solve_diffusion(k_pred, d, steps=100)
            loss += torch.mean((sim_traj - OBSERVATIONS[d])**2)
            
        if epoch > epochs // 2:
            k_dx = torch.abs(k_pred[1:,:]-k_pred[:-1,:]); k_dy = torch.abs(k_pred[:,1:]-k_pred[:,:-1])
            loss += 1e-4 * (torch.mean(k_dx) + torch.mean(k_dy))
        
        loss.backward()
        optimizer.step(); scheduler.step()
        if epoch % 500 == 0: print(f"Ep {epoch} | Loss: {loss.item():.6f}")
    return model

# PINN training is different (no simulator)
def train_pinn(model, label, epochs=5001, lr=0.001):
    # This is a simplified PINN for brevity. It learns from the ground truth trajectory.
    # In a real paper, you'd only have sparse data points.
    print(f"\n--- Training {label} ---")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # We need coordinates for PINN training
    x_coords = torch.linspace(-1, 1, RES, device=device, requires_grad=True)
    y_coords = torch.linspace(-1, 1, RES, device=device, requires_grad=True)
    coords_flat = torch.stack(torch.meshgrid(x_coords, y_coords, indexing='ij'),-1).view(-1,2)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        k_pred = model.get_k_map(RES)
        
        # PINN loss is based on residuals
        loss = 0
        for d in DIRECTIONS:
            # We are "cheating" by using the full observation trajectory for PINN
            # In reality this is just sparse data points.
            u_data = OBSERVATIONS[d] # Shape [T, H, W]
            
            # PINN needs derivatives
            u_t = (u_data[1:] - u_data[:-1]) / (1/len(u_data)) # Finite diff on time
            u_now = u_data[:-1]
            
            u_up=torch.roll(u_now,-1,1); u_down=torch.roll(u_now,1,1)
            u_left=torch.roll(u_now,-1,2); u_right=torch.roll(u_now,1,2)
            laplacian = (u_up+u_down+u_left+u_right - 4*u_now)
            
            residual = u_t - k_pred * laplacian
            loss += torch.mean(residual**2)

        loss.backward()
        optimizer.step()
        if epoch % 1000 == 0: print(f"Ep {epoch} | Residual Loss: {loss.item():.6f}")
    return model

# ==========================================
# 5. EXECUTE THE SHOOTOUT
# ==========================================
start_time = time.time()

# MLP-PINN is very unstable and slow to train, so we give it more epochs
mlp_pinn_model = train_pinn(MLP_PINN().to(device), "MLP-PINN")
kan_pinn_model = train_pinn(KAN_PINN().to(device), "KAN-PINN", epochs=2001)
mlp_diffphys_model = train_diffphys(MLP_DiffPhys().to(device), "MLP-DiffPhys")
kan_diffphys_model = train_diffphys(KAN_DiffPhys().to(device), "KAN-DiffPhys (Novelty)")

print(f"\nTotal training time: {time.time() - start_time:.2f} seconds")

# ==========================================
# 6. FINAL VISUALIZATION
# ==========================================
with torch.no_grad():
    k_mlp_pinn = mlp_pinn_model.get_k_map(RES).cpu().numpy()
    k_kan_pinn = kan_pinn_model.get_k_map(RES).cpu().numpy()
    k_mlp_diff = mlp_diffphys_model.get_k_map(RES).cpu().numpy()
    k_kan_diff = kan_diffphys_model.get_k_map(RES).cpu().numpy()

fig, axs = plt.subplots(1, 5, figsize=(25, 5))

def plot_ax(ax, data, title):
    im = ax.imshow(data, cmap='viridis', vmin=0, vmax=1.1)
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plot_ax(axs[0], K_TRUE.cpu(), "Ground Truth")
plot_ax(axs[1], k_mlp_pinn, "MLP-PINN (Baseline 1)")
plot_ax(axs[2], k_kan_pinn, "KAN-PINN (Baseline 2)")
plot_ax(axs[3], k_mlp_diff, "MLP-DiffPhys (Baseline 3)")
plot_ax(axs[4], k_kan_diff, "KAN-DiffPhys (Novelty)")

plt.tight_layout()
plt.show()