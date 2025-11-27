import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

def get_device():
    if torch.backends.mps.is_available(): return torch.device("mps")
    elif torch.cuda.is_available(): return torch.device("cuda")
    else: return torch.device("cpu")
device = get_device()
print(f"Running on: {device}")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ==========================================
# 1. WAVE PHYSICS ENGINE
# ==========================================
def solve_wave_equation(c_map, source_pos, steps=150):
    """
    Solves the Wave Equation: d2u/dt2 = c^2 * Laplacian(u)
    c_map: Wave Speed (Stiffness). High c = Hard Tumor.
    """
    H, W = c_map.shape
    u = torch.zeros((H, W), device=device)
    u_prev = torch.zeros((H, W), device=device)
    
    # Source location
    mask = torch.zeros_like(u)
    mid = H // 2
    
    if source_pos == 'top':    mask[5, mid] = 1.0
    if source_pos == 'bottom': mask[-5, mid] = 1.0
    if source_pos == 'left':   mask[mid, 5] = 1.0
    if source_pos == 'right':  mask[mid, -5] = 1.0
    
    dt = 0.01
    history = []
    
    # Precompute c^2 for speed
    c2 = c_map ** 2
    
    for i in range(steps):
        # 1. Laplacian (Finite Difference)
        u_up    = torch.roll(u, -1, 0); u_down  = torch.roll(u, 1, 0)
        u_left  = torch.roll(u, -1, 1); u_right = torch.roll(u, 1, 1)
        laplacian = (u_up + u_down + u_left + u_right - 4*u)
        
        # 2. Source Pulse (A "Ping" at the start)
        # Ricker Wavelet-ish pulse
        val = 0.0
        if i < 20:
            val = np.sin(i * 0.5) * 10.0
        
        # 3. Verlet Integration (2nd Order Time)
        # Damping factor 0.99 prevents numerical explosion (Absorbing medium)
        u_next = 2*u - u_prev + (dt**2) * (c2 * laplacian + val * mask)
        u_next = u_next * 0.99 
        
        u_prev = u.clone()
        u = u_next.clone()
        
        history.append(u.clone())
        
    return torch.stack(history)

RES = 64
DIRECTIONS = ['left', 'top'] # Pinging from two sides

def create_glioblastoma(res):
    # Background Sound Speed = 1.0
    c_true = torch.ones((res, res), device=device) * 1.0
    
    x = torch.linspace(-1, 1, res, device=device)
    y = torch.linspace(-1, 1, res, device=device)
    gx, gy = torch.meshgrid(x, y, indexing='ij')
    r = torch.sqrt(gx**2 + gy**2)
    theta = torch.atan2(gy, gx)
    
    # Irregular Shape
    radius = 0.4 + 0.1 * torch.sin(3 * theta) + 0.05 * torch.cos(7 * theta + 1.0)
    mask = r < radius
    
    # Tumor is HARD (High Sound Speed = 2.0)
    # This causes reflections!
    c_true[mask] = 2.0 
    return c_true

C_TRUE = create_glioblastoma(RES)
OBSERVATIONS = {}
print("Simulating Wave Physics...")
for d in DIRECTIONS:
    # FIX: Use 'source_pos' instead of 'direction'
    traj = solve_wave_equation(C_TRUE, source_pos=d, steps=150)
    OBSERVATIONS[d] = traj + torch.randn_like(traj) * 0.02

# ==========================================
# 2. HIGH-DEF KAN ARCHITECTURE
# ==========================================
class SpatialRBF(nn.Module):
    def __init__(self, out_features, grid_res, sharpness):
        super().__init__()
        x = torch.linspace(-1, 1, grid_res, device=device)
        self.grid = torch.stack(torch.meshgrid(x, x, indexing='ij'), dim=-1).view(-1, 2)
        self.spline = nn.Parameter(torch.randn(grid_res**2, out_features, device=device) * 0.02)
        self.base = nn.Parameter(torch.randn(out_features, 2, device=device) * 0.02)
        self.sharpness = sharpness
    
    def forward(self, x):
        base = F.linear(x, self.base)
        dist_sq = torch.sum((x.unsqueeze(1) - self.grid.unsqueeze(0))**2, dim=2)
        return base + torch.matmul(torch.exp(-dist_sq * self.sharpness), self.spline)

class WaveFinderKAN(nn.Module):
    def __init__(self):
        super().__init__()
        # High Res, High Sharpness
        self.net = nn.Sequential(
            SpatialRBF(32, grid_res=64, sharpness=200.0), 
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        # Initialize to Background Speed (1.0)
        # Inverse Sigmoid(0.5) = 0.0, so bias 0 is fine if we scale output
        with torch.no_grad(): self.net[-1].bias.fill_(0.0)

    def get_c_map(self, res):
        x = torch.linspace(-1, 1, res, device=device)
        coords = torch.stack(torch.meshgrid(x, x, indexing='ij'), dim=-1).view(-1, 2)
        
        logits = self.net(coords)
        
        # Binary forcing (Steep Sigmoid)
        prob = torch.sigmoid(logits * 3.0)
        
        # Map 0->1 to Speed 1.0->2.0
        c_phys = prob * 1.0 + 1.0
        
        # Radial Mask (Prevent edge cheating)
        r = torch.sqrt(coords[:,0]**2 + coords[:,1]**2).view(res, res)
        mask = 1.0 - torch.sigmoid(20.0 * (r - 0.8))
        
        return c_phys.view(res, res) * mask + 1.0 * (1 - mask)

# ==========================================
# 3. TRAINING LOOP
# ==========================================
model = WaveFinderKAN().to(device)
# Lower LR is crucial for Wave Equation because gradients are oscillatory
optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

print("Starting Waveform Inversion...")
loss_hist = []

for epoch in range(1001):
    optimizer.zero_grad()
    c_pred = model.get_c_map(RES)
    
    loss_data = 0
    for d in DIRECTIONS:
        # Run Simulation
        sim_traj = solve_wave_equation(c_pred, d, steps=150)
        
        # Compare Movies (Trajectory Matching)
        loss_data += torch.mean((sim_traj - OBSERVATIONS[d])**2)
        
    # TV Regularization (Annealed)
    if epoch > 400:
        dx = torch.abs(c_pred[1:,:] - c_pred[:-1,:])
        dy = torch.abs(c_pred[:,1:] - c_pred[:,:-1])
        loss = loss_data + 5e-5 * (torch.mean(dx) + torch.mean(dy))
    else:
        loss = loss_data
        
    loss.backward()
    
    # Gradient Clipping is MANDATORY for Wave Eq
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    
    optimizer.step()
    loss_hist.append(loss.item())
    
    if epoch % 100 == 0:
        print(f"Ep {epoch} | Loss: {loss.item():.6f}")

# ==========================================
# 4. RESULTS
# ==========================================
with torch.no_grad():
    c_final = model.get_k_map(RES).cpu().numpy() # (Reuse method name get_k_map for c_map)

plt.figure(figsize=(12, 5))

plt.subplot(1, 3, 1)
plt.imshow(C_TRUE.cpu(), cmap='viridis')
plt.title("True Speed ($c$)")

plt.subplot(1, 3, 2)
plt.imshow(c_final, cmap='viridis')
plt.title("KAN Wave Inversion")

plt.subplot(1, 3, 3)
plt.imshow(OBSERVATIONS['left'][-1].cpu(), cmap='RdBu')
plt.title("Wave Field (Final Frame)")

plt.show()