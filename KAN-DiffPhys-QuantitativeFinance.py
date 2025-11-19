import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# Set device
device = torch.device("cpu")

# ==========================================
# 1. DATA GENERATION (The Flash Crash Sim)
# ==========================================
def generate_financial_data():
    print("Simulating Market Regime Change...")
    days = 1000
    dt = 1/365 
    t = torch.linspace(0, 1, days)
    
    # True Regimes (Hidden Parameter)
    sigma_true = []
    for time in t:
        if 0.4 < time < 0.6:
            sigma_true.append(0.8) # CRISIS (High Volatility)
        elif time >= 0.6:
            sigma_true.append(0.2) # New Normal
        else:
            sigma_true.append(0.1) # Calm
            
    sigma_true = torch.tensor(sigma_true, dtype=torch.float32)
    
    # Returns (The Observable Noise)
    # Formula: return = sigma * noise * sqrt(dt)
    torch.manual_seed(42) # Fixed seed for reproducibility
    noise = torch.randn(days)
    returns = sigma_true * noise * torch.sqrt(torch.tensor(dt))
    
    # Price Path (Cumulative Product)
    price = [100.0]
    for r in returns:
        price.append(price[-1] * (1 + r))
        
    return t.view(-1, 1), returns.view(-1, 1), sigma_true.view(-1, 1), price

# Generate
t_train, returns_train, sigma_true, price_path = generate_financial_data()
t_train.requires_grad_(True) # Enable time gradients for KAN

# ==========================================
# 2. KAN ARCHITECTURE (Shallow & Wide)
# ==========================================
class RBFKanLayer(nn.Module):
    def __init__(self, in_features, out_features, grid_size=30):
        super().__init__()
        self.grid_size = grid_size
        grid = torch.linspace(0, 1, grid_size)
        self.register_buffer("grid", grid)
        
        # Initialize weights with slight variance to break symmetry
        self.spline_weight = nn.Parameter(torch.randn(in_features * grid_size, out_features) * 0.1)
        self.base_weight = nn.Parameter(torch.randn(out_features, in_features) * 0.1)

    def forward(self, x):
        base_output = F.linear(x, self.base_weight)
        x_expanded = x.unsqueeze(-1)
        grid_expanded = self.grid.view(1, 1, -1)
        
        # RBF Basis (Gaussian Bumps)
        basis = torch.exp(-torch.pow(x_expanded - grid_expanded, 2) * 20.0)
        basis = basis.view(x.size(0), -1)
        return base_output + torch.matmul(basis, self.spline_weight)

class VolatilityKAN(nn.Module):
    def __init__(self):
        super().__init__()
        # Architecture: 1 -> 64 -> 1 (Log Sigma)
        self.net = nn.Sequential(
            RBFKanLayer(1, 64, grid_size=30),
            nn.Tanh(),
            nn.Linear(64, 1) 
        )
        
    def get_log_sigma(self, t):
        return self.net(t)

model = VolatilityKAN().to(device)

# --- SMART INITIALIZATION ---
# Bias the final layer to start at the dataset's global average volatility.
# This prevents the "Cold Start" problem where the model outputs ~0 and dies.
global_std_annual = torch.std(returns_train) * np.sqrt(365)
with torch.no_grad():
    model.net[-1].bias.fill_(torch.log(global_std_annual))

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
dt_tensor = torch.tensor(1/365).float()

# ==========================================
# 3. TRAINING LOOP
# ==========================================
print(f"Global Annual Vol approx: {global_std_annual:.3f}")
print("Starting Training...")

for epoch in range(5001):
    optimizer.zero_grad()
    
    # 1. Predict Annual Log-Volatility
    log_sigma_annual = model.get_log_sigma(t_train)
    
    # 2. Physics Loss (Negative Log Likelihood)
    # We transform Annual Sigma -> Daily Variance
    log_var_daily = 2 * log_sigma_annual + torch.log(dt_tensor)
    var_daily = torch.exp(log_var_daily)
    
    nll_loss = torch.mean(log_var_daily + (returns_train**2 / var_daily))
    
    # 3. TV Regularization (Smoothness Constraint)
    # Calculate d(Sigma)/dt
    u_grad = torch.autograd.grad(log_sigma_annual, t_train, torch.ones_like(log_sigma_annual), create_graph=True)[0]
    tv_loss = torch.mean(torch.abs(u_grad))
    
    # Annealing Schedule
    # Phase 1: Exploration (tv=0). Find the spikes.
    # Phase 2: Smoothing (tv=0.1). Flatten the plateaus.
    tv_weight = 0.0
    if epoch > 1000: tv_weight = 0.04 # Your optimized value
    
    loss = nll_loss + tv_weight * tv_loss
    
    loss.backward()
    optimizer.step()
    
    if epoch % 1000 == 0:
        print(f"Ep {epoch:<4} | Loss: {nll_loss.item():.4f} | TV Loss: {tv_loss.item():.4f}")

# ==========================================
# 4. VISUALIZATION
# ==========================================
with torch.no_grad():
    pred_sigma_curve = torch.exp(model.get_log_sigma(t_train)).numpy()

plt.figure(figsize=(12, 10)) # Taller figure for 3 plots

# Plot 1: The Observable (Stock Price)
plt.subplot(3, 1, 1)
plt.plot(price_path, 'k-', linewidth=1.5)
plt.title("1. The Observable: Stock Price (Geometric Brownian Motion)", fontsize=14)
plt.ylabel("Price ($)")
plt.grid(True, alpha=0.5)
plt.margins(x=0)

# Plot 2: The Input Data (Returns)
plt.subplot(3, 1, 2)
plt.plot(returns_train.numpy(), 'grey', alpha=0.6, linewidth=1)
plt.title("2. The Input Data: Daily Returns (Noise)", fontsize=14)
plt.ylabel("Daily Return")
plt.grid(True, alpha=0.5)
plt.margins(x=0)

# Plot 3: The Discovery (Volatility)
plt.subplot(3, 1, 3)
plt.plot(t_train.detach().numpy(), sigma_true.numpy(), 'k--', label='True Volatility Regime', linewidth=2.5)
plt.plot(t_train.detach().numpy(), pred_sigma_curve, 'r-', label='KAN Discovery', linewidth=3.5)
plt.title("3. The Hidden Parameter: KAN Regime Detection", fontsize=14)
plt.xlabel("Time (Years)", fontsize=12)
plt.ylabel("Annualized Volatility", fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.5)
plt.ylim(0, 1.0)
plt.margins(x=0)

plt.tight_layout()
plt.show()