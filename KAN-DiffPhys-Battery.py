import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# Set device
device = torch.device("cpu")

# ==========================================
# 1. GENERATE BATTERY DATA (The "Knee Point")
# ==========================================
def generate_battery_data():
    print("Simulating Battery Lifecycle...")
    cycles = 1000
    t = torch.linspace(0, 1, cycles) # Normalized time (0 to 100% of test)
    dt = t[1] - t[0]
    
    # Initial Capacity (100%)
    Q = 1.0
    data_Q = []
    
    # The Hidden Physics: The Degradation Rate Lambda
    for i in range(cycles):
        cur_t = t[i]
        
        # The Knee Point Switch
        switch = torch.sigmoid(50 * (cur_t - 0.6))
        degradation_rate = 0.2 * (1 - switch) + 5.0 * switch
        
        # The ODE: dQ/dt = -lambda * Q
        dQ = -degradation_rate * Q
        Q = Q + dt * dQ
        
        # Add Measurement Noise (Sensor noise)
        noise = torch.randn(1) * 0.005 
        Q_observed = Q + noise
        
        data_Q.append(Q_observed)
        
    return t.view(-1, 1), torch.stack(data_Q).view(-1, 1), dt

t_train, Q_train, dt = generate_battery_data()

# --- THE FIX IS HERE ---
# We must enable gradient tracking on Time so we can calculate d(Lambda)/dt later
t_train.requires_grad_(True)

# ==========================================
# 2. KAN ARCHITECTURE
# ==========================================
class RBFKanLayer(nn.Module):
    def __init__(self, in_features, out_features, grid_size=30):
        super().__init__()
        self.grid_size = grid_size
        grid = torch.linspace(0, 1, grid_size)
        self.register_buffer("grid", grid)
        self.spline_weight = nn.Parameter(torch.randn(in_features * grid_size, out_features) * 0.05)
        self.base_weight = nn.Parameter(torch.randn(out_features, in_features) * 0.05)

    def forward(self, x):
        base_output = F.linear(x, self.base_weight)
        x_expanded = x.unsqueeze(-1)
        grid_expanded = self.grid.view(1, 1, -1)
        basis = torch.exp(-torch.pow(x_expanded - grid_expanded, 2) * 20.0)
        basis = basis.view(x.size(0), -1)
        return base_output + torch.matmul(basis, self.spline_weight)

class BatteryDiffPhys(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Input: Time (Cycle)
        # Output: Degradation Rate (Lambda)
        self.lambda_kan = nn.Sequential(
            RBFKanLayer(1, 16, grid_size=40), # High res for the knee
            RBFKanLayer(16, 16),
            RBFKanLayer(16, 1)
        )

    def get_degradation_rate(self, t):
        # Lambda must be positive
        return F.softplus(self.lambda_kan(t))

    def forward(self, t_steps, initial_Q, dt):
        simulated_Q = []
        Q = initial_Q.clone()
        
        # Predict dynamic degradation rate for entire timeline
        lambdas = self.get_degradation_rate(t_steps)
        
        for i in range(len(t_steps)):
            deg_rate = lambdas[i]
            
            # Physics Update
            dQ = -deg_rate * Q
            Q = Q + dt * dQ
            
            # Safety Clamp (Battery can't have negative charge)
            Q = torch.clamp(Q, 0.0, 1.2)
            
            simulated_Q.append(Q)
            
        return torch.stack(simulated_Q)

# ==========================================
# 3. TRAINING (Prognostics)
# ==========================================
model = BatteryDiffPhys().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

print("\n=== Starting Battery Knee Detection ===")

# Detach targets
Q_target = Q_train.detach()
Q_init = torch.tensor([1.0]) # We know it starts new

loss_hist = []

for epoch in range(2001):
    optimizer.zero_grad()
    
    # Forward Simulation
    Q_pred = model(t_train, Q_init, dt)
    
    # Loss: MSE
    loss = torch.mean((Q_pred - Q_target)**2)
    
    # Total Variation Regularization 
    # This is why we needed t_train.requires_grad = True
    lambdas = model.get_degradation_rate(t_train)
    
    # Calculate d(Lambda)/dt to enforce smoothness
    lambda_dt = torch.autograd.grad(lambdas, t_train, torch.ones_like(lambdas), create_graph=True)[0]
    loss_tv = torch.mean(torch.abs(lambda_dt))
    
    # Combine Losses
    total_loss = loss + 0.0001 * loss_tv
    
    total_loss.backward()
    optimizer.step()
    loss_hist.append(total_loss.item())
    
    if epoch % 500 == 0:
        print(f"Ep {epoch} | Loss: {total_loss.item():.6f}")

# ==========================================
# 4. RESULTS VISUALIZATION
# ==========================================
with torch.no_grad():
    final_Q = model(t_train, Q_init, dt).numpy()
    final_lambda = model.get_degradation_rate(t_train).numpy()

# Reconstruct Truth for Plotting
t_np = t_train.detach().numpy()
true_switch = 1.0 / (1.0 + np.exp(-50 * (t_np - 0.6)))
true_lambda = 0.2 * (1 - true_switch) + 5.0 * true_switch

plt.figure(figsize=(12, 5))

# Plot 1: Capacity Fade (The Observable)
plt.subplot(1, 2, 1)
plt.plot(t_train.detach(), Q_train, 'k.', alpha=0.1, label='Sensor Data (Noisy)')
plt.plot(t_train.detach(), final_Q, 'r-', linewidth=3, label='KAN Prognosis')
plt.title("Battery Capacity Fade")
plt.xlabel("Normalized Cycles")
plt.ylabel("Capacity (Q)")
plt.legend()
plt.grid(True)

# Plot 2: Degradation Rate (The Hidden Parameter)
plt.subplot(1, 2, 2)
plt.plot(t_train.detach(), true_lambda, 'k--', label='True Physics')
plt.plot(t_train.detach(), final_lambda, 'b-', linewidth=3, label='Discovered Degradation')
plt.title("Detected 'Knee Point' (Degradation Rate)")
plt.xlabel("Normalized Cycles")
plt.ylabel("Lambda")
plt.legend()
plt.grid(True)

plt.show()