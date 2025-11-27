import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# --- 1. Parameters & Configuration ---
BETA, GAMMA, N = 0.3, 0.1, 1000.0
S0, I0, R0 = 990.0, 10.0, 0.0
t_begin, t_train_end, t_plot_end = 0.0, 80.0, 150.0

# Normalized Initial Conditions
s0_norm = S0 / N
i0_norm = I0 / N
r0_norm = R0 / N

# Time scaling factor (to map t=[0, 80] -> network_input=[0, 1])
# This improves gradient descent stability significantly.
T_SCALE = t_train_end 

# --- 2. Ground Truth Generation (Scipy) ---
def sir_ode(t, y, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]

t_gt = np.linspace(t_begin, t_plot_end, 500)
sol = solve_ivp(sir_ode, [t_begin, t_plot_end], [S0, I0, R0], 
                args=(N, BETA, GAMMA), t_eval=t_gt)
S_true, I_true, R_true = sol.y

# --- 3. The PINN Architecture (Standard MLP) ---
class SIR_PINN(nn.Module):
    def __init__(self):
        super().__init__()
        # Standard MLP: Input (1) -> Hidden (3x64) -> Output (3)
        # Tanh activation is preferred for PINNs (smooth derivatives).
        self.net = nn.Sequential(
            nn.Linear(1, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 3)  # Output: s, i, r
        )

    def forward(self, t):
        return self.net(t)

# --- 4. Physics Loss Calculation ---
def compute_loss(model, t_physics, t_initial):
    # A. Initial Condition Loss (at t=0)
    y_init = model(t_initial)
    s_init, i_init, r_init = y_init[:, 0], y_init[:, 1], y_init[:, 2]
    loss_ic = (torch.mean((s_init - s0_norm)**2) + 
               torch.mean((i_init - i0_norm)**2) + 
               torch.mean((r_init - r0_norm)**2))

    # B. Physics Derivatives
    # Enable gradients for t_physics
    t_physics.requires_grad_(True)
    y_pred = model(t_physics)
    s, i, r = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2]

    # Compute gradients (dy/dt_network)
    # Note: gradients are w.r.t normalized time input
    grads_s = torch.autograd.grad(s, t_physics, torch.ones_like(s), create_graph=True)[0]
    grads_i = torch.autograd.grad(i, t_physics, torch.ones_like(i), create_graph=True)[0]
    grads_r = torch.autograd.grad(r, t_physics, torch.ones_like(r), create_graph=True)[0]

    # Apply Chain Rule: dy/dt_real = (dy/dt_network) * (1 / T_SCALE)
    ds_dt = grads_s * (1.0 / T_SCALE)
    di_dt = grads_i * (1.0 / T_SCALE)
    dr_dt = grads_r * (1.0 / T_SCALE)

    # C. SIR ODE Residuals
    # ds/dt = -beta * s * i
    res_s = ds_dt - (-BETA * s * i)
    # di/dt = beta * s * i - gamma * i
    res_i = di_dt - (BETA * s * i - GAMMA * i)
    # dr/dt = gamma * i
    res_r = dr_dt - (GAMMA * i)

    loss_ode = torch.mean(res_s**2) + torch.mean(res_i**2) + torch.mean(res_r**2)
    
    # D. Conservation Loss (Optional but helpful constraint: S+I+R = 1)
    loss_conserv = torch.mean(((s + i + r) - 1.0)**2)

    return loss_ic + loss_ode + loss_conserv

# --- 5. Training Loop ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pinn = SIR_PINN().to(device)
optimizer = torch.optim.Adam(pinn.parameters(), lr=1e-3)

# Data Preparation
# t_initial is exactly 0.0 (normalized)
t_init_ts = torch.tensor([[0.0]], dtype=torch.float32).to(device)

# t_physics samples from [0, 1] (representing 0 to 80 days)
t_phys_ts = torch.linspace(0, 1, 300).view(-1, 1).to(device)

epochs = 10000
print(f"--- Starting Training on {device} ---")

for epoch in range(epochs):
    optimizer.zero_grad()
    loss = compute_loss(pinn, t_phys_ts, t_init_ts)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 1000 == 0:
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {loss.item():.8f}")

print("Training finished.")

# --- 6. Evaluation & Extrapolation ---
pinn.eval()
with torch.no_grad():
    # We predict up to t=150 (Plot End). 
    # Since training ended at 80, we normalize 150 by 80.
    # Input range: [0, 1.875]
    t_eval_norm = torch.tensor(t_gt / T_SCALE, dtype=torch.float32).view(-1, 1).to(device)
    
    y_pred = pinn(t_eval_norm).cpu().numpy()
    
    # Scale back to population count
    S_pred = y_pred[:, 0] * N
    I_pred = y_pred[:, 1] * N
    R_pred = y_pred[:, 2] * N

# --- 7. Visualization ---
plt.figure(figsize=(12, 5))

# Plot S
plt.subplot(1, 2, 1)
plt.plot(t_gt, S_true, 'k-', alpha=0.6, linewidth=3, label='Ground Truth')
plt.plot(t_gt, S_pred, 'r--', linewidth=2, label='PINN Prediction')
plt.axvline(x=t_train_end, color='gray', linestyle=':', label='Train Boundary')
plt.title('Susceptible (S)')
plt.xlabel('Days')
plt.ylabel('Population')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot I
plt.subplot(1, 2, 2)
plt.plot(t_gt, I_true, 'k-', alpha=0.6, linewidth=3, label='Ground Truth')
plt.plot(t_gt, I_pred, 'b--', linewidth=2, label='PINN Prediction')
plt.axvline(x=t_train_end, color='gray', linestyle=':', label='Train Boundary')
plt.title('Infected (I)')
plt.xlabel('Days')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()