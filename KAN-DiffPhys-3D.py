import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import marching_cubes # For 3D surface plotting

# --- [SETUP, 3D PHYSICS, KAN ARCHITECTURE are IDENTICAL to the previous 3D example] ---
# ... (Re-using the same get_device, solve_diffusion_3d, SpatialRBF3D, ObjectFinderKAN3D classes)

def get_device():
    if torch.backends.mps.is_available(): return torch.device("mps")
    elif torch.cuda.is_available(): return torch.device("cuda")
    else: return torch.device("cpu")
device = get_device()
print(f"Running on: {device}")

def solve_diffusion_3d(k_map, direction='left-right', steps=50):
    D, H, W = k_map.shape
    u = torch.zeros((D, H, W), device=device)
    if direction == 'left-right':   u[:, :, :W//2] = 1.0
    elif direction == 'top-bottom': u[:, :H//2, :] = 1.0
    elif direction == 'front-back': u[:D//2, :, :] = 1.0
    dt = 0.01
    history = []
    for _ in range(steps):
        u_front = torch.roll(u, -1, dims=0); u_front[-1,:,:] = u[-1,:,:]
        u_back  = torch.roll(u,  1, dims=0); u_back[0,:,:]  = u[0,:,:]
        u_up    = torch.roll(u, -1, dims=1); u_up[:,-1,:]   = u[:,-1,:]
        u_down  = torch.roll(u,  1, dims=1); u_down[:,0,:]   = u[:,0,:]
        u_left  = torch.roll(u, -1, dims=2); u_left[:,:,-1]  = u[:,:,-1]
        u_right = torch.roll(u,  1, dims=2); u_right[:,:,0]  = u[:,:,0]
        laplacian = (u_front + u_back + u_up + u_down + u_left + u_right - 6*u)
        u = u + dt * (k_map * laplacian)
        history.append(u.clone())
    return torch.stack(history)

class SpatialRBF3D(nn.Module):
    def __init__(self, out_features, grid_res=10):
        super().__init__()
        x = torch.linspace(-1, 1, grid_res, device=device)
        grid_pts = torch.stack(torch.meshgrid(x, x, x, indexing='ij'), dim=-1)
        self.grid = grid_pts.view(-1, 3)
        self.spline_weight = nn.Parameter(torch.randn(grid_res**3, out_features) * 0.1)
        self.base_weight = nn.Parameter(torch.randn(out_features, 3) * 0.1)
    def forward(self, x):
        base = F.linear(x, self.base_weight)
        dist_sq = torch.sum((x.unsqueeze(1) - self.grid.unsqueeze(0))**2, dim=2)
        basis = torch.exp(-dist_sq * 50.0)
        return base + torch.matmul(basis, self.spline_weight)

class ObjectFinderKAN3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # Increased grid_res for more complex shape
            SpatialRBF3D(32, grid_res=14),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
    def get_k_map(self, res):
        x = torch.linspace(-1, 1, res, device=device)
        coords_3d = torch.stack(torch.meshgrid(x, x, x, indexing='ij'), dim=-1).view(-1, 3)
        k_flat = torch.sigmoid(self.net(coords_3d)) * 1.99 + 0.01
        return k_flat.view(res, res, res)
# --------------------------------------------------------------------------------------


# ==========================================
# 2. IRREGULAR 3D SHAPE & DATA GENERATION
# ==========================================

def create_ground_truth_irregular_3d(res):
    """Creates a 'dumbbell' shape made of two overlapping spheres."""
    k_true = torch.ones((res, res, res), device=device) * 1.0
    
    # Create coordinate grid
    x = torch.linspace(-1, 1, res, device=device)
    z, y, x = torch.meshgrid(x, x, x, indexing='ij')

    # Define two spheres
    radius1, radius2 = 0.4, 0.3
    center1 = (0.0, 0.0, -0.3)
    center2 = (0.0, 0.0, 0.4)
    
    sphere1_mask = (x - center1[0])**2 + (y - center1[1])**2 + (z - center1[2])**2 < radius1**2
    sphere2_mask = (x - center2[0])**2 + (y - center2[1])**2 + (z - center2[2])**2 < radius2**2
    
    # Combine the masks and set the low-diffusion value
    k_true[sphere1_mask | sphere2_mask] = 0.1
    return k_true

RES_3D = 24 # Slightly higher res to see shape better
DIRECTIONS_3D = ['left-right', 'top-bottom', 'front-back']
K_TRUE_3D_IRREGULAR = create_ground_truth_irregular_3d(RES_3D)

OBSERVATIONS_3D = {}
print("Simulating 3D Data for Irregular Shape...")
for direction in DIRECTIONS_3D:
    traj = solve_diffusion_3d(K_TRUE_3D_IRREGULAR, direction=direction, steps=60)
    OBSERVATIONS_3D[direction] = traj + torch.randn_like(traj) * 0.02


# --- [TRAINING is IDENTICAL to the previous 3D example] ---
def train_model_3d(model, label):
    print(f"\n--- Training {label} ---")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)
    
    # More epochs for the harder problem
    for epoch in range(2501):
        optimizer.zero_grad()
        k_pred = model.get_k_map(RES_3D)
        total_loss = 0
        for d in DIRECTIONS_3D:
            sim_traj = solve_diffusion_3d(k_pred, d, steps=60)
            total_loss += torch.mean((sim_traj - OBSERVATIONS_3D[d])**2)
        if epoch > 1250: # Start TV regularization later
            k_dz = torch.abs(k_pred[1:,:,:] - k_pred[:-1,:,:])
            k_dy = torch.abs(k_pred[:,1:,:] - k_pred[:,:-1,:])
            k_dx = torch.abs(k_pred[:,:,1:] - k_pred[:,:,:-1])
            loss_tv = torch.mean(k_dz) + torch.mean(k_dy) + torch.mean(k_dx)
            loss = total_loss + 2e-4 * loss_tv # Slightly stronger TV
        else:
            loss = total_loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        if epoch % 250 == 0:
            print(f"Ep {epoch} | Loss: {loss.item():.6f} | LR: {scheduler.get_last_lr()[0]:.6f}")
    return model

kan_model_3d = train_model_3d(ObjectFinderKAN3D().to(device), "3D KAN on Irregular Shape")
# -----------------------------------------------------------


# ==========================================
# 5. ADVANCED 3D VISUALIZATION
# ==========================================
with torch.no_grad():
    k_kan_3d = kan_model_3d.get_k_map(RES_3D).cpu().numpy()

k_true_np = K_TRUE_3D_IRREGULAR.cpu().numpy()

# Isosurface value: we define the "object" as any region where k < 0.5
level = 0.5

# Use scikit-image's marching_cubes algorithm to find the surface
verts_true, faces_true, _, _ = marching_cubes(k_true_np, level=level)
verts_kan, faces_kan, _, _ = marching_cubes(k_kan_3d, level=level)

# Create the 3D plot
fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(121, projection='3d')
ax1.set_title("Ground Truth Shape")
ax1.plot_trisurf(verts_true[:, 0], verts_true[:, 1], faces_true, verts_true[:, 2],
                 cmap='Spectral', lw=1)

ax2 = fig.add_subplot(122, projection='3d')
ax2.set_title("KAN Reconstruction")
ax2.plot_trisurf(verts_kan[:, 0], verts_kan[:, 1], faces_kan, verts_kan[:, 2],
                 cmap='Spectral', lw=1)
plt.tight_layout()
plt.show()