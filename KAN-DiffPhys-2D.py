import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output

device = torch.device("cpu")

# ==========================================
# 1. PHYSICS ENGINE (With Diagnostic Probes)
# ==========================================
def solve_heat_multi_source(k_map, source_pos, steps=50):
    H, W = k_map.shape
    u = torch.zeros((H, W), device=device)
    
    mask = torch.zeros_like(u)
    # Sources closer to center to ensure heat actually hits the tumor
    mid = H // 2
    if source_pos == 'top':    mask[2:5, mid-5:mid+5] = 1.0
    if source_pos == 'bottom': mask[-5:-2, mid-5:mid+5] = 1.0
    if source_pos == 'left':   mask[mid-5:mid+5, 2:5] = 1.0
    if source_pos == 'right':  mask[mid-5:mid+5, -5:-2] = 1.0
        
    dt = 0.01
    
    for _ in range(steps):
        u_up    = torch.roll(u, -1, dims=0); u_up[-1, :] = u[-1, :]
        u_down  = torch.roll(u, 1, dims=0);  u_down[0, :] = u[0, :]
        u_left  = torch.roll(u, -1, dims=1); u_left[:, -1] = u[:, -1]
        u_right = torch.roll(u, 1, dims=1);  u_right[:, 0] = u[:, 0]
        
        laplacian = (u_up + u_down + u_left + u_right - 4*u)
        
        # Physics
        du_dt = k_map * laplacian + 20.0 * mask # Increased heat intensity
        u = u + dt * du_dt
        
    return u

# ==========================================
# 2. DATA GENERATION
# ==========================================
RES = 32 # Lower resolution for easier debugging
POSITIONS = ['top', 'left'] # Fewer views to simplify gradients

def create_ground_truth(res):
    k = torch.ones((res, res)) * 0.2
    mid = res // 2
    # Cross
    k[mid-8:mid+8, mid-2:mid+2] = 2.0
    k[mid-2:mid+2, mid-8:mid+8] = 2.0
    return k

K_TRUE = create_ground_truth(RES)
OBSERVATIONS = {}

print("Generating Ground Truth Data...")
for pos in POSITIONS:
    clean = solve_heat_multi_source(K_TRUE, pos, steps=100)
    # Low noise to ensure signal exists
    OBSERVATIONS[pos] = clean + torch.randn_like(clean) * 0.01

# ==========================================
# 3. KAN ARCHITECTURE (Simplified)
# ==========================================
class SimpleKAN(nn.Module):
    def __init__(self):
        super().__init__()
        # Standard Coordinate MLP first (Baseline check)
        # If this fails, a KAN will definitely fail.
        # We start simple to ensure gradients flow.
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        # Force initialization to background value (0.2)
        # Inverse sigmoid(0.2) approx -1.38
        with torch.no_grad():
            self.net[-1].bias.fill_(-1.38)
            self.net[-1].weight.data *= 0.01 # Small weights start flat

    def get_k_map(self, res):
        x = torch.linspace(-1, 1, res)
        y = torch.linspace(-1, 1, res)
        gx, gy = torch.meshgrid(x, y, indexing='ij')
        coords = torch.stack([gx.flatten(), gy.flatten()], dim=1)
        
        out = self.net(coords)
        k = F.sigmoid(out) * 3.0 # Output range [0, 3.0]
        return k.view(res, res)

# ==========================================
# 4. DIAGNOSTIC TRAINING LOOP
# ==========================================
model = SimpleKAN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

print("\n=== STARTING DIAGNOSTIC RUN ===")
print(f"{'Epoch':<5} | {'Loss':<8} | {'GradNorm':<8} | {'Min K':<6} | {'Max K':<6} | {'Mean K':<6}")

loss_history = []

for epoch in range(501):
    optimizer.zero_grad()
    
    k_pred = model.get_k_map(RES)
    
    loss_total = 0
    for pos in POSITIONS:
        u_sim = solve_heat_multi_source(k_pred, pos, steps=100)
        # Normalize loss by size
        loss_total += torch.mean((u_sim - OBSERVATIONS[pos])**2)
    
    loss = loss_total
    
    # Calculate Gradients
    loss.backward()
    
    # --- DIAGNOSTICS ---
    # 1. Calculate Gradient Norm (Is the physics talking to the network?)
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item()
            
    # 2. Record Stats
    k_min = k_pred.min().item()
    k_max = k_pred.max().item()
    k_mean = k_pred.mean().item()
    
    optimizer.step()
    loss_history.append(loss.item())
    
    if epoch % 50 == 0:
        print(f"{epoch:<5} | {loss.item():.6f} | {total_norm:.6f} | {k_min:.3f}  | {k_max:.3f}  | {k_mean:.3f}")

# ==========================================
# 5. VISUALIZATION
# ==========================================
with torch.no_grad():
    k_final = model.get_k_map(RES).numpy()

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.title("True K")
plt.imshow(K_TRUE)
plt.colorbar()

plt.subplot(1, 3, 2)
plt.title("Reconstructed K")
plt.imshow(k_final, vmin=0, vmax=3)
plt.colorbar()

plt.subplot(1, 3, 3)
plt.title("Loss History")
plt.plot(loss_history)
plt.yscale('log')
plt.show()