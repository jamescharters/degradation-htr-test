import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import imageio
import os
import time

# --- Step 1 & 2: Generate Data (Same as before) ---

# Physics parameters
g = 9.8; vx0 = 2.0; vy0 = 10.0; x0, y0 = 0, 5; e = 0.85; dt = 0.01; T = 4.0

# Generate full ground truth data
t_full = np.arange(0, T, dt)
x_full, y_full = [], []
vx, vy = vx0, vy0
x, y = x0, y0
for t in t_full:
    x += vx * dt; y += vy * dt; vy -= g * dt
    if y < 0:
        y = 0; vy = -vy * e
    x_full.append(x); y_full.append(y)
x_full = np.array(x_full); y_full = np.array(y_full)

# Create sparse training dataset
num_training_points = 8
idx = np.linspace(0, len(t_full) - 1, num_training_points).astype(int)
t_data = torch.tensor(t_full[idx], dtype=torch.float32).view(-1, 1)
x_data = torch.tensor(x_full[idx], dtype=torch.float32).view(-1, 1)
y_data = torch.tensor(y_full[idx], dtype=torch.float32).view(-1, 1)
pos_data = torch.cat([x_data, y_data], dim=1)

# Known Initial Conditions
t_ic = torch.tensor([0.0], dtype=torch.float32).view(-1, 1)
pos_ic = torch.tensor([x0, y0], dtype=torch.float32)
vel_ic = torch.tensor([vx0, vy0], dtype=torch.float32)

# --- Step 3: Define the Advanced PINN Architecture ---

class AdvancedPINN(nn.Module):
    def __init__(self):
        super(AdvancedPINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 2)
        )

    def forward(self, t):
        return self.net(t)

# --- Step 4: Train the Advanced PINN ---

pinn_model = AdvancedPINN()
loss_fn = nn.MSELoss()

# Loss weights
lambda_data = 1.0
lambda_ic = 1.0
lambda_bounce = 1.0

# Phase 1: Adam Optimizer
print("--- Phase 1: Training with Adam Optimizer ---")
optimizer_adam = torch.optim.Adam(pinn_model.parameters(), lr=1e-3)
adam_epochs = 8000
start_time = time.time()

for epoch in range(adam_epochs):
    optimizer_adam.zero_grad()
    
    # 1. Data Loss
    pos_pred_data = pinn_model(t_data)
    data_loss = loss_fn(pos_pred_data, pos_data)

    # 2. Initial Condition Loss
    t_ic.requires_grad_(True)
    pos_pred_ic = pinn_model(t_ic)
    vel_pred_ic = torch.autograd.grad(pos_pred_ic, t_ic, grad_outputs=torch.ones_like(pos_pred_ic), create_graph=True)[0]
    ic_loss = loss_fn(pos_pred_ic.squeeze(), pos_ic) + loss_fn(vel_pred_ic.squeeze(), vel_ic)

    # 3. Physics Loss
    t_physics = torch.linspace(0, T, 100, requires_grad=True).view(-1, 1)
    pos_pred_physics = pinn_model(t_physics)
    x_pred, y_pred = pos_pred_physics[:, 0], pos_pred_physics[:, 1]
    
    vx_pred = torch.autograd.grad(x_pred, t_physics, torch.ones_like(x_pred), create_graph=True)[0]
    vy_pred = torch.autograd.grad(y_pred, t_physics, torch.ones_like(y_pred), create_graph=True)[0]
    ax_pred = torch.autograd.grad(vx_pred, t_physics, torch.ones_like(vx_pred), create_graph=True)[0]
    ay_pred = torch.autograd.grad(vy_pred, t_physics, torch.ones_like(vy_pred), create_graph=True)[0]
    
    physics_loss = loss_fn(ax_pred, torch.zeros_like(ax_pred)) + loss_fn(ay_pred, -g * torch.ones_like(ay_pred))
    
    # 4. Bounce Condition Loss
    t_bounce_est_idx = torch.argmin(y_pred.detach())
    t_bounce = t_physics[t_bounce_est_idx].clone().detach().requires_grad_(True)
    
    epsilon = 1e-4
    pos_before = pinn_model(t_bounce - epsilon)
    pos_after = pinn_model(t_bounce + epsilon)
    
    vy_before = torch.autograd.grad(pos_before[1], t_bounce, create_graph=True)[0] # <-- CORRECTED
    vy_after = torch.autograd.grad(pos_after[1], t_bounce, create_graph=True)[0]  # <-- CORRECTED
    
    y_at_bounce = pinn_model(t_bounce)[1]                                       # <-- CORRECTED
    bounce_loss = loss_fn(vy_after, -e * vy_before) + loss_fn(y_at_bounce, torch.zeros_like(y_at_bounce))

    # 5. Total Loss with Annealing
    lambda_physics = 0.1 * min(1.0, epoch / (adam_epochs * 0.5))
    total_loss = (lambda_data * data_loss + 
                  lambda_ic * ic_loss + 
                  lambda_physics * physics_loss +
                  lambda_bounce * bounce_loss)
    
    total_loss.backward()
    optimizer_adam.step()

    if epoch % 1000 == 0:
        print(f"Adam Epoch {epoch}: Total={total_loss.item():.4f}, Data={data_loss.item():.4f}, IC={ic_loss.item():.4f}, Phys={physics_loss.item():.4f}, Bounce={bounce_loss.item():.4f}")

print(f"Adam training finished in {time.time() - start_time:.2f} seconds.")

# Phase 2: L-BFGS Optimizer
print("\n--- Phase 2: Fine-tuning with L-BFGS Optimizer ---")
optimizer_lbfgs = torch.optim.LBFGS(pinn_model.parameters(), 
                                    max_iter=5000, 
                                    line_search_fn="strong_wolfe")

def closure():
    optimizer_lbfgs.zero_grad()
    pos_pred_data = pinn_model(t_data)
    data_loss = loss_fn(pos_pred_data, pos_data)
    t_ic.requires_grad_(True)
    pos_pred_ic = pinn_model(t_ic)
    vel_pred_ic = torch.autograd.grad(pos_pred_ic, t_ic, grad_outputs=torch.ones_like(pos_pred_ic), create_graph=True)[0]
    ic_loss = loss_fn(pos_pred_ic.squeeze(), pos_ic) + loss_fn(vel_pred_ic.squeeze(), vel_ic)
    t_physics = torch.linspace(0, T, 100, requires_grad=True).view(-1, 1)
    pos_pred_physics = pinn_model(t_physics)
    x_pred, y_pred = pos_pred_physics[:, 0], pos_pred_physics[:, 1]
    vx_pred = torch.autograd.grad(x_pred, t_physics, torch.ones_like(x_pred), create_graph=True)[0]
    vy_pred = torch.autograd.grad(y_pred, t_physics, torch.ones_like(y_pred), create_graph=True)[0]
    ax_pred = torch.autograd.grad(vx_pred, t_physics, torch.ones_like(vx_pred), create_graph=True)[0]
    ay_pred = torch.autograd.grad(vy_pred, t_physics, torch.ones_like(vy_pred), create_graph=True)[0]
    physics_loss = loss_fn(ax_pred, torch.zeros_like(ax_pred)) + loss_fn(ay_pred, -g * torch.ones_like(ay_pred))
    
    t_bounce_est_idx = torch.argmin(y_pred.detach())
    t_bounce = t_physics[t_bounce_est_idx].clone().detach().requires_grad_(True)
    
    epsilon = 1e-4
    pos_before = pinn_model(t_bounce - epsilon)
    pos_after = pinn_model(t_bounce + epsilon)
    
    vy_before = torch.autograd.grad(pos_before[1], t_bounce, create_graph=True)[0] # <-- CORRECTED
    vy_after = torch.autograd.grad(pos_after[1], t_bounce, create_graph=True)[0]  # <-- CORRECTED
    
    y_at_bounce = pinn_model(t_bounce)[1]                                       # <-- CORRECTED
    bounce_loss = loss_fn(vy_after, -e * vy_before) + loss_fn(y_at_bounce, torch.zeros_like(y_at_bounce))
    
    total_loss = (lambda_data * data_loss + 
                  lambda_ic * ic_loss + 
                  0.1 * physics_loss +
                  lambda_bounce * bounce_loss)
    
    total_loss.backward()
    print(f"L-BFGS Loss: {total_loss.item():.6f}", end='\r')
    return total_loss

start_time = time.time()
optimizer_lbfgs.step(closure)
print(f"\nL-BFGS training finished in {time.time() - start_time:.2f} seconds.")


# --- Step 5: Generate GIF ---
print("\nGenerating final GIF...")
t_eval = torch.tensor(t_full, dtype=torch.float32).view(-1, 1)
with torch.no_grad():
    pinn_pred = pinn_model(t_eval).numpy()

filenames = []
frame_dir = "advanced_pinn_frames"
os.makedirs(frame_dir, exist_ok=True)

for i in range(0, len(t_full), 4):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x_full, y_full, 'k--', label='Ground Truth', alpha=0.6, linewidth=2.5)
    ax.plot(x_data.numpy(), y_data.numpy(), 'ro', markersize=10, label='Training Data')
    ax.plot(pinn_pred[:i, 0], pinn_pred[:i, 1], 'g-', label='Advanced PINN', linewidth=2)
    ax.plot(pinn_pred[i, 0], pinn_pred[i, 1], 'go', markersize=15)
    ax.set_xlim(min(x_full) - 1, max(x_full) + 1); ax.set_ylim(-1, max(y_full) + 1)
    ax.set_title(f"Advanced PINN Simulation | Time: {t_full[i]:.2f}s")
    ax.set_xlabel("Horizontal Position (x)"); ax.set_ylabel("Vertical Position (y)")
    ax.legend(loc='upper right'); ax.grid(True)
    
    filename = os.path.join(frame_dir, f'frame_{i:04d}.png')
    filenames.append(filename)
    plt.savefig(filename)
    plt.close()

with imageio.get_writer('bouncing_ball_advanced.gif', mode='I', duration=0.04) as writer:
    for filename in filenames:
        writer.append_data(imageio.imread(filename))

for filename in filenames:
    os.remove(filename)
os.rmdir(frame_dir)

print("\nDone! Check for 'bouncing_ball_advanced.gif' in your directory.")