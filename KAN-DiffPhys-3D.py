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

# Noise level for experiments
NOISE_LEVEL = 0.02  # Try 0.02, 0.05, 0.1 to test robustness

# ==========================================
# 2. 3D PHYSICS & DATA GENERATION
# ==========================================
def solve_diffusion_3d(k_map, direction='x', steps=50):
    """Solve 3D diffusion equation with 6-point stencil"""
    D, H, W = k_map.shape
    u = torch.zeros((D, H, W), device=device)
    
    # Initial conditions based on direction
    if direction == 'x': u[:, :, :W//2] = 1.0
    elif direction == 'y': u[:, :H//2, :] = 1.0
    elif direction == 'z': u[:D//2, :, :] = 1.0
    
    dt = 0.005  # Smaller timestep for 3D stability
    history = []
    
    for _ in range(steps):
        # 6-point stencil (x, y, z directions)
        u_xp = torch.roll(u, -1, dims=2); u_xp[:, :, -1] = u[:, :, -1]
        u_xm = torch.roll(u, 1, dims=2);  u_xm[:, :, 0] = u[:, :, 0]
        u_yp = torch.roll(u, -1, dims=1); u_yp[:, -1, :] = u[:, -1, :]
        u_ym = torch.roll(u, 1, dims=1);  u_ym[:, 0, :] = u[:, 0, :]
        u_zp = torch.roll(u, -1, dims=0); u_zp[-1, :, :] = u[-1, :, :]
        u_zm = torch.roll(u, 1, dims=0);  u_zm[0, :, :] = u[0, :, :]
        
        laplacian = (u_xp + u_xm + u_yp + u_ym + u_zp + u_zm - 6*u)
        u = u + dt * (k_map * laplacian)
        history.append(u.clone())
    
    return torch.stack(history)

RES = 24  # Smaller for 3D (24^3 = 13,824 points)
DIRECTIONS = ['x', 'y', 'z']

def create_ground_truth_3d(res):
    """Create highly irregular tumor with dendritic tentacles and sharp features"""
    k_true = torch.ones((res, res, res), device=device) * 1.0
    
    center = res // 2
    z, y, x = torch.meshgrid(
        torch.arange(res, device=device),
        torch.arange(res, device=device),
        torch.arange(res, device=device),
        indexing='ij'
    )
    
    # Main tumor body (irregular ellipsoid)
    dist_main = torch.sqrt(
        ((z - center) / 5.0)**2 + 
        ((y - center) / 4.5)**2 + 
        ((x - center) / 5.5)**2
    )
    
    # Multiple finger-like protrusions (dendritic growth pattern)
    tentacle1 = (
        ((z - center) / 8.0)**2 + 
        ((y - (center - 5)) / 1.5)**2 + 
        ((x - (center + 2)) / 1.2)**2
    ) < 1.0
    
    tentacle2 = (
        ((z - (center + 4)) / 1.8)**2 + 
        ((y - (center + 1)) / 6.0)**2 + 
        ((x - (center - 1)) / 1.5)**2
    ) < 1.0
    
    tentacle3 = (
        ((z - (center - 2)) / 1.3)**2 + 
        ((y - (center + 3)) / 1.6)**2 + 
        ((x - center) / 7.0)**2
    ) < 1.0
    
    tentacle4 = (
        ((z - (center + 2)) / 2.0)**2 + 
        ((y - center) / 1.4)**2 + 
        ((x - (center + 6)) / 3.5)**2
    ) < 1.0
    
    # Irregular satellite nodules
    nodule1 = ((z - (center - 4))**2 + (y - (center + 4))**2 + (x - (center - 3))**2) < 6
    nodule2 = ((z - (center + 5))**2 + (y - (center - 3))**2 + (x - (center + 4))**2) < 5
    nodule3 = ((z - (center - 3))**2 + (y - (center - 4))**2 + (x - (center + 5))**2) < 4
    
    # Sharp concave region
    concave = (
        ((z - (center + 1)) / 2.5)**2 + 
        ((y - (center - 2)) / 2.0)**2 + 
        ((x - (center - 4)) / 2.8)**2
    ) < 1.0
    
    # Combine all regions with SHARP boundaries
    tumor_mask = (
        (dist_main < 1.0) | tentacle1 | tentacle2 | tentacle3 | tentacle4 | 
        nodule1 | nodule2 | nodule3
    ) & ~concave
    
    k_true[tumor_mask] = 0.12
    
    # Sharp necrotic core
    core_dist = torch.sqrt(
        ((z - center) / 2.2)**2 + 
        ((y - (center + 0.5)) / 1.8)**2 + 
        ((x - (center - 0.5)) / 2.5)**2
    )
    k_true[core_dist < 1.0] = 0.05
    
    # Sharp calcifications
    calc1 = ((z - (center + 1))**2 + (y - (center - 1))**2 + (x - center)**2) < 3
    calc2 = ((z - (center - 2))**2 + (y - (center + 2))**2 + (x - (center + 2))**2) < 2.5
    calc3 = ((z - center)**2 + (y - (center + 1))**2 + (x - (center - 2))**2) < 2
    k_true[calc1 | calc2 | calc3] = 0.75
    
    return k_true

K_TRUE = create_ground_truth_3d(RES)
OBSERVATIONS = {}
print(f"Simulating 3D Data with noise level: {NOISE_LEVEL}")
for direction in DIRECTIONS:
    traj = solve_diffusion_3d(K_TRUE, direction=direction, steps=80)
    OBSERVATIONS[direction] = traj + torch.randn_like(traj) * NOISE_LEVEL
    print(f"  {direction}: {traj.shape}")

# ==========================================
# 3. ARCHITECTURES: KAN vs MLP
# ==========================================

# KAN Architecture with Spatial RBF
class SpatialRBF(nn.Module):
    """High-resolution 2D spatial basis functions"""
    def __init__(self, out_features, grid_res=30):
        super().__init__()
        x = torch.linspace(-1, 1, grid_res, device=device)
        self.grid = torch.stack(torch.meshgrid(x, x, indexing='ij'), dim=-1).view(-1, 2)
        self.spline_weight = nn.Parameter(torch.randn(grid_res**2, out_features, device=device) * 0.1)
        self.base_weight = nn.Parameter(torch.randn(out_features, 2, device=device) * 0.1)
        self.gamma = 80.0
    
    def forward(self, x):
        base = F.linear(x, self.base_weight)
        dist_sq = torch.sum((x.unsqueeze(1) - self.grid.unsqueeze(0))**2, dim=2)
        basis = torch.exp(-dist_sq * self.gamma)
        return base + torch.matmul(basis, self.spline_weight)

class SliceKAN(nn.Module):
    """KAN-based slice predictor"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            SpatialRBF(32, grid_res=30),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
    
    def forward(self, coords_2d):
        return torch.sigmoid(self.net(coords_2d)) * 1.99 + 0.01

# MLP Architecture (Fair Comparison)
class SliceMLP(nn.Module):
    """MLP-based slice predictor with similar parameter count"""
    def __init__(self):
        super().__init__()
        # Tuned to have similar parameter count to KAN
        # KAN has ~900 RBF centers * 32 + 32*2 + 32*1 ≈ 28,896 + 64 + 32 = 28,992 params per slice
        # MLP: 2->128->128->32->1 = 256 + 16,384 + 4,096 + 32 = 20,768 params
        # Add one more layer: 2->128->128->128->32->1 = 256 + 16,384 + 16,384 + 4,096 + 32 = 37,152 params
        self.net = nn.Sequential(
            nn.Linear(2, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        
    def forward(self, coords_2d):
        return torch.sigmoid(self.net(coords_2d)) * 1.99 + 0.01

class VolumeModel(nn.Module):
    """Stack of slice networks for 3D volume"""
    def __init__(self, num_slices, slice_type='kan'):
        super().__init__()
        if slice_type == 'kan':
            self.slices = nn.ModuleList([SliceKAN() for _ in range(num_slices)])
        else:
            self.slices = nn.ModuleList([SliceMLP() for _ in range(num_slices)])
        self.num_slices = num_slices
    
    def get_k_map(self, res):
        x = torch.linspace(-1, 1, res, device=device)
        y_coords, x_coords = torch.meshgrid(x, x, indexing='ij')
        coords_2d = torch.stack([y_coords.flatten(), x_coords.flatten()], dim=-1)
        
        slices = []
        for slice_net in self.slices:
            k_slice = slice_net(coords_2d).view(res, res)
            slices.append(k_slice)
        
        return torch.stack(slices, dim=0)

# ==========================================
# 4. TRAINING WITH ADAPTIVE TV
# ==========================================
def adaptive_tv_weight(epoch, max_epochs, initial_weight=5e-4, final_weight=1e-5):
    """Annealing schedule: strong regularization early, weak late"""
    if epoch < max_epochs * 0.4:
        return 0.0
    progress = (epoch - max_epochs * 0.4) / (max_epochs * 0.6)
    return initial_weight * (final_weight / initial_weight) ** progress

def train_model_3d(model, label, max_epochs=1500):
    print(f"\n{'='*60}")
    print(f"Training {label}")
    print(f"{'='*60}")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=400, gamma=0.5)
    
    loss_history = []
    physics_loss_history = []
    tv_loss_history = []
    
    for epoch in range(max_epochs):
        optimizer.zero_grad()
        k_pred = model.get_k_map(RES)
        
        # Physics loss
        physics_loss = 0
        for d in DIRECTIONS:
            sim_traj = solve_diffusion_3d(k_pred, d, steps=80)
            physics_loss += torch.mean((sim_traj - OBSERVATIONS[d])**2)
        
        # Adaptive TV regularization
        tv_weight = adaptive_tv_weight(epoch, max_epochs)
        if tv_weight > 0:
            k_dz = torch.abs(k_pred[1:, :, :] - k_pred[:-1, :, :])
            k_dx = torch.abs(k_pred[:, :, 1:] - k_pred[:, :, :-1])
            k_dy = torch.abs(k_pred[:, 1:, :] - k_pred[:, :-1, :])
            tv_loss = torch.mean(k_dz) + torch.mean(k_dx) + torch.mean(k_dy)
            loss = physics_loss + tv_weight * tv_loss
        else:
            tv_loss = torch.tensor(0.0)
            loss = physics_loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        loss_history.append(loss.item())
        physics_loss_history.append(physics_loss.item())
        tv_loss_history.append(tv_loss.item())
        
        if epoch % 300 == 0:
            print(f"Epoch {epoch:4d} | Total: {loss.item():.6f} | "
                  f"Physics: {physics_loss.item():.6f} | TV: {tv_loss.item():.6f} | "
                  f"TV_weight: {tv_weight:.2e} | LR: {scheduler.get_last_lr()[0]:.6f}")
    
    print(f"\n{'='*60}")
    print(f"Training Complete - {label}")
    print(f"{'='*60}")
    print(f"Final Total Loss:   {loss_history[-1]:.6f}")
    print(f"Final Physics Loss: {physics_loss_history[-1]:.6f}")
    print(f"Loss Reduction:     {100*(1-loss_history[-1]/loss_history[0]):.1f}%")
    
    return model, loss_history, physics_loss_history, tv_loss_history

# ==========================================
# 5. TRAIN BOTH MODELS
# ==========================================
print("\n" + "="*60)
print("COMPARING KAN vs MLP ARCHITECTURES")
print("="*60)

# Train KAN
kan_model, kan_loss, kan_phys, kan_tv = train_model_3d(
    VolumeModel(num_slices=RES, slice_type='kan').to(device), 
    "KAN (Spatial RBF)"
)

# Train MLP
mlp_model, mlp_loss, mlp_phys, mlp_tv = train_model_3d(
    VolumeModel(num_slices=RES, slice_type='mlp').to(device),
    "MLP (4-layer)"
)

# ==========================================
# 6. COMPARATIVE VISUALIZATION
# ==========================================
with torch.no_grad():
    k_kan = kan_model.get_k_map(RES).cpu().numpy()
    k_mlp = mlp_model.get_k_map(RES).cpu().numpy()

k_true_np = K_TRUE.cpu().numpy()

# Calculate metrics
kan_mse = np.mean((k_kan - k_true_np)**2)
kan_mae = np.mean(np.abs(k_kan - k_true_np))
mlp_mse = np.mean((k_mlp - k_true_np)**2)
mlp_mae = np.mean(np.abs(k_mlp - k_true_np))

print(f"\n{'='*60}")
print(f"FINAL COMPARISON (Noise Level: {NOISE_LEVEL})")
print(f"{'='*60}")
print(f"KAN Reconstruction:")
print(f"  MSE: {kan_mse:.6f}")
print(f"  MAE: {kan_mae:.6f}")
print(f"\nMLP Reconstruction:")
print(f"  MSE: {mlp_mse:.6f}")
print(f"  MAE: {mlp_mae:.6f}")
print(f"\nKAN Improvement:")
print(f"  MSE: {100*(1-kan_mse/mlp_mse):.1f}% better")
print(f"  MAE: {100*(1-kan_mae/mlp_mae):.1f}% better")

# Main comparison plot
fig = plt.figure(figsize=(18, 10))
mid = RES // 2

# XY slices
plt.subplot(3, 3, 1)
plt.imshow(k_true_np[mid, :, :], cmap='viridis', vmin=0, vmax=1.1)
plt.title("Ground Truth (XY)", fontsize=12, fontweight='bold')
plt.axis('off')

plt.subplot(3, 3, 2)
plt.imshow(k_kan[mid, :, :], cmap='viridis', vmin=0, vmax=1.1)
plt.title(f"KAN (MSE: {kan_mse:.4f})", fontsize=12, fontweight='bold')
plt.axis('off')

plt.subplot(3, 3, 3)
plt.imshow(k_mlp[mid, :, :], cmap='viridis', vmin=0, vmax=1.1)
plt.title(f"MLP (MSE: {mlp_mse:.4f})", fontsize=12, fontweight='bold')
plt.axis('off')

# XZ slices
plt.subplot(3, 3, 4)
plt.imshow(k_true_np[:, mid, :], cmap='viridis', vmin=0, vmax=1.1)
plt.title("Ground Truth (XZ)", fontsize=12, fontweight='bold')
plt.axis('off')

plt.subplot(3, 3, 5)
plt.imshow(k_kan[:, mid, :], cmap='viridis', vmin=0, vmax=1.1)
plt.title("KAN Reconstruction", fontsize=12, fontweight='bold')
plt.axis('off')

plt.subplot(3, 3, 6)
plt.imshow(k_mlp[:, mid, :], cmap='viridis', vmin=0, vmax=1.1)
plt.title("MLP Reconstruction", fontsize=12, fontweight='bold')
plt.axis('off')

# YZ slices
plt.subplot(3, 3, 7)
plt.imshow(k_true_np[:, :, mid], cmap='viridis', vmin=0, vmax=1.1)
plt.title("Ground Truth (YZ)", fontsize=12, fontweight='bold')
plt.axis('off')

plt.subplot(3, 3, 8)
plt.imshow(k_kan[:, :, mid], cmap='viridis', vmin=0, vmax=1.1)
plt.title("KAN Reconstruction", fontsize=12, fontweight='bold')
plt.axis('off')

plt.subplot(3, 3, 9)
plt.imshow(k_mlp[:, :, mid], cmap='viridis', vmin=0, vmax=1.1)
plt.title("MLP Reconstruction", fontsize=12, fontweight='bold')
plt.axis('off')

plt.tight_layout()
plt.show()

# Loss comparison curves
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(kan_loss, 'b-', linewidth=2, label='KAN', alpha=0.8)
axes[0].plot(mlp_loss, 'r-', linewidth=2, label='MLP', alpha=0.8)
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Total Loss', fontsize=12)
axes[0].set_title('Total Loss Comparison', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

axes[1].plot(kan_phys, 'b-', linewidth=2, label='KAN', alpha=0.8)
axes[1].plot(mlp_phys, 'r-', linewidth=2, label='MLP', alpha=0.8)
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Physics Loss', fontsize=12)
axes[1].set_title('Physics Loss Comparison', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()