import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import marching_cubes
import time

# ==========================================
# 1. SETUP & UTILITIES
# ==========================================
def get_device():
    if torch.backends.mps.is_available(): return torch.device("mps")
    elif torch.cuda.is_available(): return torch.device("cuda")
    else: return torch.device("cpu")
device = get_device()
print(f"INFO: Running on: {device}")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ==========================================
# 2. 3D PHYSICS SIMULATOR (Unchanged)
# ==========================================
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

# ==========================================
# 3. GROUND TRUTH DATA (New Shape)
# ==========================================
def create_ground_truth_sharp_3d(res):
    """Creates a shape made of a sphere and a cube to test sharp edges."""
    print("INFO: Creating ground truth with sharp edges (sphere + cube)...")
    k_true = torch.ones((res, res, res), device=device) * 1.0
    
    # Create coordinate grid
    x_coords = torch.linspace(-1, 1, res, device=device)
    z, y, x = torch.meshgrid(x_coords, x_coords, x_coords, indexing='ij')

    # Define a sphere
    # radius = 0.4
    # center1 = (0.0, 0.0, -0.4)
    # sphere_mask = (x - center1[0])**2 + (y - center1[1])**2 + (z - center1[2])**2 < radius**2
    
    # Define a cube
    s = 0.4 # half-width of the cube
    center2 = (0.0, 0.0, 0.4)
    cube_mask = ( (x > center2[0]-s) & (x < center2[0]+s) &
                  (y > center2[1]-s) & (y < center2[1]+s) &
                  (z > center2[2]-s) & (z < center2[2]+s) )
    
    #k_true[sphere_mask | cube_mask] = 0.1
    k_true[cube_mask] = 0.1
    return k_true

# ==========================================
# 4. MODEL ARCHITECTURES
# ==========================================

# --- Baseline: MLP with Positional Encoding (strong competitor) ---
class PositionalEncoder(nn.Module):
    def __init__(self, input_dims, num_freqs):
        super().__init__()
        self.freq_bands = 2**torch.linspace(0, num_freqs-1, num_freqs, device=device) * torch.pi
        self.output_dims = input_dims * (2 * num_freqs + 1)
        
    def forward(self, x):
        encoded = [x]
        for freq in self.freq_bands:
            encoded.append(torch.sin(x * freq))
            encoded.append(torch.cos(x * freq))
        return torch.cat(encoded, dim=-1)

class ObjectFinderMLP3D(nn.Module):
    def __init__(self, num_freqs=8, hidden_dim=128, num_layers=4):
        super().__init__()
        self.encoder = PositionalEncoder(3, num_freqs)
        
        layers = []
        in_dim = self.encoder.output_dims
        for i in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.SiLU()) # Swish activation is a good modern choice
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.net = nn.Sequential(*layers)
        print(f"INFO: Initialized MLP model.")

    def get_k_map(self, res):
        x = torch.linspace(-1, 1, res, device=device)
        coords_3d = torch.stack(torch.meshgrid(x, x, x, indexing='ij'), dim=-1).view(-1, 3)
        encoded_coords = self.encoder(coords_3d)
        k_flat = torch.sigmoid(self.net(encoded_coords)) * 1.99 + 0.01
        return k_flat.view(res, res, res)

# --- Novelty: KAN-based model ---
class SpatialRBF3D(nn.Module):
    def __init__(self, out_features, grid_res=12):
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
    def __init__(self, grid_res=12):
        super().__init__()
        self.net = nn.Sequential(
            SpatialRBF3D(32, grid_res=grid_res),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        print(f"INFO: Initialized KAN model with grid_res={grid_res}.")

    def get_k_map(self, res):
        x = torch.linspace(-1, 1, res, device=device)
        coords_3d = torch.stack(torch.meshgrid(x, x, x, indexing='ij'), dim=-1).view(-1, 3)
        k_flat = torch.sigmoid(self.net(coords_3d)) * 1.99 + 0.01
        return k_flat.view(res, res, res)

# ==========================================
# 5. GENERIC TRAINING LOOP
# ==========================================
def train_model(model, model_label, observations, res, epochs, steps):
    print(f"\n--- Training {model_label} ({count_parameters(model):,} parameters) ---")
    start_time = time.time()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)
    
    for epoch in range(epochs + 1):
        optimizer.zero_grad()
        k_pred = model.get_k_map(res)
        
        total_loss = 0
        for d in observations.keys():
            sim_traj = solve_diffusion_3d(k_pred, d, steps=steps)
            total_loss += torch.mean((sim_traj - observations[d])**2)
        
        # Annealed Total Variation Regularization
        if epoch > epochs // 2:
            k_dz = torch.abs(k_pred[1:,:,:] - k_pred[:-1,:,:])
            k_dy = torch.abs(k_pred[:,1:,:] - k_pred[:,:-1,:])
            k_dx = torch.abs(k_pred[:,:,1:] - k_pred[:,:,:-1])
            loss_tv = torch.mean(k_dz) + torch.mean(k_dy) + torch.mean(k_dx)
            loss = total_loss + 2e-4 * loss_tv
        else:
            loss = total_loss
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if epoch % 250 == 0:
            print(f"Ep {epoch}/{epochs} | Loss: {loss.item():.6f} | LR: {scheduler.get_last_lr()[0]:.6f}")

    end_time = time.time()
    print(f"--- Finished training {model_label} in {end_time - start_time:.2f} seconds ---")
    return model.get_k_map(res).detach()

# ==========================================
# 6. MAIN EXECUTION BLOCK
# ==========================================
if __name__ == '__main__':
    # --- Configuration ---
    RES_3D = 24
    SIM_STEPS = 60
    TRAIN_EPOCHS = 2500
    DIRECTIONS_3D = ['left-right', 'top-bottom', 'front-back']

    # --- Data Generation ---
    K_TRUE_SHARP = create_ground_truth_sharp_3d(RES_3D)
    OBSERVATIONS_3D = {}
    print("INFO: Simulating 3D observation data...")
    for direction in DIRECTIONS_3D:
        traj = solve_diffusion_3d(K_TRUE_SHARP, direction=direction, steps=SIM_STEPS)
        OBSERVATIONS_3D[direction] = traj + torch.randn_like(traj) * 0.02

    # --- Train Models ---
    mlp_model = ObjectFinderMLP3D().to(device)
    k_mlp = train_model(mlp_model, "MLP-DiffPhys", OBSERVATIONS_3D, RES_3D, TRAIN_EPOCHS, SIM_STEPS)
    
    kan_model = ObjectFinderKAN3D(grid_res=12).to(device) # grid_res=12 makes it very parameter efficient
    k_kan = train_model(kan_model, "KAN-DiffPhys", OBSERVATIONS_3D, RES_3D, TRAIN_EPOCHS, SIM_STEPS)

    # --- Visualization ---
    print("INFO: Generating visualizations...")
    k_true_np = K_TRUE_SHARP.cpu().numpy()
    k_mlp_np = k_mlp.cpu().numpy()
    k_kan_np = k_kan.cpu().numpy()

    level = 0.5 # Isosurface level to define the object boundary
    
    verts_true, faces_true, _, _ = marching_cubes(k_true_np, level=level)
    verts_mlp, faces_mlp, _, _ = marching_cubes(k_mlp_np, level=level)
    verts_kan, faces_kan, _, _ = marching_cubes(k_kan_np, level=level)

    fig = plt.figure(figsize=(18, 7))
    
    # Ground Truth Plot
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.set_title("1. Ground Truth Shape")
    ax1.plot_trisurf(verts_true[:, 0], verts_true[:, 1], faces_true, verts_true[:, 2], cmap='viridis', lw=0.5)
    
    # MLP Reconstruction Plot
    ax2 = fig.add_subplot(132, projection='3d')
    mlp_params = count_parameters(mlp_model)
    ax2.set_title(f"2. MLP Reconstruction\n({mlp_params:,} parameters)")
    ax2.plot_trisurf(verts_mlp[:, 0], verts_mlp[:, 1], faces_mlp, verts_mlp[:, 2], cmap='viridis', lw=0.5)
    
    # KAN Reconstruction Plot
    ax3 = fig.add_subplot(133, projection='3d')
    kan_params = count_parameters(kan_model)
    ax3.set_title(f"3. KAN Reconstruction (Novel)\n({kan_params:,} parameters)")
    ax3.plot_trisurf(verts_kan[:, 0], verts_kan[:, 1], faces_kan, verts_kan[:, 2], cmap='viridis', lw=0.5)

    plt.tight_layout()
    plt.show()