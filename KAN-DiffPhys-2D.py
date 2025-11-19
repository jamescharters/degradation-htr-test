import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------
# 1. SETUP
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
print("Simulating Data...")
for direction in DIRECTIONS:
    traj = solve_diffusion(K_TRUE, direction=direction, steps=100)
    OBSERVATIONS[direction] = traj + torch.randn_like(traj) * 0.02

# ==========================================
# 3. KAN ARCHITECTURE (High Resolution & Sharp)
# ==========================================
class SpatialRBF(nn.Module):
    def __init__(self, out_features, grid_res=25): # Higher grid resolution
        super().__init__()
        x = torch.linspace(-1, 1, grid_res, device=device)
        self.grid = torch.stack(torch.meshgrid(x, x, indexing='ij'), dim=-1).view(-1, 2)
        self.spline_weight = nn.Parameter(torch.randn(grid_res**2, out_features, device=device) * 0.1)
        self.base_weight = nn.Parameter(torch.randn(out_features, 2, device=device) * 0.1)
    
    def forward(self, x):
        base = F.linear(x, self.base_weight)
        dist_sq = torch.sum((x.unsqueeze(1) - self.grid.unsqueeze(0))**2, dim=2)
        # Sharper basis functions (gamma=100.0) allow for finer details
        basis = torch.exp(-dist_sq * 100.0) 
        return base + torch.matmul(basis, self.spline_weight)

class RockFinderKAN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            SpatialRBF(32, grid_res=30), # High-res input layer
            nn.Tanh(),
            nn.Linear(32, 1)
        )
    def get_k_map(self, res):
        x = torch.linspace(-1, 1, res, device=device)
        coords = torch.stack(torch.meshgrid(x, x, indexing='ij'), dim=-1).view(-1, 2)
        k_flat = torch.sigmoid(self.net(coords)) * 1.99 + 0.01
        return k_flat.view(res, res)

# ==========================================
# 4. TRAINING (With Annealing)
# ==========================================
def train_model(model, label):
    print(f"\n--- Training {label} ---")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    # Learning Rate Scheduler: Drop LR by half every 500 steps
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)
    
    # Train for more epochs to allow for fine-tuning
    for epoch in range(1501):
        optimizer.zero_grad()
        k_pred = model.get_k_map(RES)
        
        total_loss = 0
        for d in DIRECTIONS:
            sim_traj = solve_diffusion(k_pred, d, steps=100)
            total_loss += torch.mean((sim_traj - OBSERVATIONS[d])**2)
            
        # TV Annealing: Only apply regularization in the second half of training
        if epoch > 750:
            k_dx = torch.abs(k_pred[1:,:] - k_pred[:-1,:])
            k_dy = torch.abs(k_pred[:,1:] - k_pred[:,:-1])
            loss_tv = torch.mean(k_dx) + torch.mean(k_dy)
            loss = total_loss + 1e-4 * loss_tv
        else:
            loss = total_loss
        
        loss.backward()
        optimizer.step()
        scheduler.step() # Step the scheduler
        
        if epoch % 250 == 0:
            print(f"Ep {epoch} | Loss: {loss.item():.6f} | LR: {scheduler.get_last_lr()[0]:.6f}")
    return model

kan_model = train_model(RockFinderKAN().to(device), "KAN")

# ==========================================
# 5. FINAL VISUALIZATION
# ==========================================
with torch.no_grad():
    k_kan = kan_model.get_k_map(RES).cpu().numpy()

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(K_TRUE.cpu(), cmap='viridis', vmin=0, vmax=1.1)
plt.title("Ground Truth")

plt.subplot(1, 2, 2)
plt.imshow(k_kan, cmap='viridis', vmin=0, vmax=1.1)
plt.title("Final KAN-DiffPhys Reconstruction")

plt.tight_layout()
plt.show()