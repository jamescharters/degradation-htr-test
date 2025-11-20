import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cpu")

# ==========================================
# 1. DATA GENERATION (Standard Gaussian)
# ==========================================
def generate_financial_data():
    print("1. Simulating Market (True Physics: Gaussian, Alpha=2.0)...")
    days = 1000
    dt = 1/365 
    t = torch.linspace(0, 1, days)
    
    sigma_true = []
    for time in t:
        if 0.4 < time < 0.6:
            sigma_true.append(0.8) # CRISIS
        elif time >= 0.6:
            sigma_true.append(0.2) # New Normal
        else:
            sigma_true.append(0.1) # Calm
            
    sigma_true = torch.tensor(sigma_true, dtype=torch.float32)
    
    # TRUE PHYSICS: Normal Distribution (Alpha = 2)
    torch.manual_seed(42) 
    noise = torch.randn(days) 
    returns = sigma_true * noise * torch.sqrt(torch.tensor(dt))
    
    return t.view(-1, 1), returns.view(-1, 1), sigma_true.view(-1, 1)

t_train, returns_train, sigma_true = generate_financial_data()
t_train.requires_grad_(True)
dt_tensor = torch.tensor(1/365).float()

# ==========================================
# 2. KAN WITH LEARNABLE PHYSICS
# ==========================================
class RBFKanLayer(nn.Module):
    def __init__(self, in_features, out_features, grid_size=30):
        super().__init__()
        self.grid_size = grid_size
        grid = torch.linspace(0, 1, grid_size)
        self.register_buffer("grid", grid)
        self.spline_weight = nn.Parameter(torch.randn(in_features * grid_size, out_features) * 0.1)
        self.base_weight = nn.Parameter(torch.randn(out_features, in_features) * 0.1)

    def forward(self, x):
        base_output = F.linear(x, self.base_weight)
        x_expanded = x.unsqueeze(-1)
        grid_expanded = self.grid.view(1, 1, -1)
        basis = torch.exp(-torch.pow(x_expanded - grid_expanded, 2) * 20.0)
        basis = basis.view(x.size(0), -1)
        return base_output + torch.matmul(basis, self.spline_weight)

class KAN_LearnPhys(nn.Module):
    def __init__(self):
        super().__init__()
        # 1. The Neural Network (Learns the Volatility Curve)
        self.net = nn.Sequential(
            RBFKanLayer(1, 64, grid_size=30),
            nn.Tanh(),
            nn.Linear(64, 1) 
        )
        
        # 2. THE LEARNABLE PHYSICS PARAMETER (Alpha)
        # We use a Parameter that we will constrain to be positive later
        # Initialize at 1.5 (Fat Tailed guess)
        self.raw_alpha = nn.Parameter(torch.tensor(1.5)) 

    def get_log_sigma(self, t):
        return self.net(t)
    
    def get_alpha(self):
        # Constrain alpha to be positive (can't have negative exponents)
        # We use Softplus to keep it smooth and > 0
        return F.softplus(self.raw_alpha)

model = KAN_LearnPhys().to(device)

# Initialize Weights smart
global_std = torch.std(returns_train) * np.sqrt(365)
with torch.no_grad():
    model.net[-1].bias.fill_(torch.log(global_std))

# We use a lower LR for stability when learning physics parameters
optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

# ==========================================
# 3. TRAINING LOOP (Discovery)
# ==========================================
print("2. Starting Physics Discovery...")
print("   Initial Guess for Alpha: ~1.50 (Assumption: Fat Tailed)")
print("   Goal: Discover Alpha -> 2.00 (Truth: Gaussian)")

alpha_history = []
loss_history = []

for epoch in range(5001):
    optimizer.zero_grad()
    
    # A. Predict Volatility (The Curve)
    log_sigma_annual = model.get_log_sigma(t_train)
    
    # Convert Annual Sigma -> Daily Scale (s)
    # For generalized distributions, "Scale" != "Standard Deviation", but they are proportional.
    # We treat prediction as the scale parameter.
    log_scale_daily = log_sigma_annual + 0.5 * torch.log(dt_tensor)
    scale_daily = torch.exp(log_scale_daily)
    
    # B. Retrieve Current "Laws of Physics" (Alpha)
    alpha = model.get_alpha()
    alpha_history.append(alpha.item())
    
    # C. THE RIGOROUS PHYSICS LOSS (Generalized Error Distribution NLL)
    # This includes the Partition Function (Gamma) to prevent cheating.
    # PDF(x) ~ exp(-0.5 * |x/s|^alpha) / NormalizingConstant
    
    # 1. The Error Term (The Fit)
    z = torch.abs(returns_train) / scale_daily
    term_error = 0.5 * torch.pow(z, alpha)
    
    # 2. The Normalizing Constant (The Cheat Prevention)
    # Log(Z) = log(scale) + lgamma(1/alpha) - log(alpha) - (1/alpha)*log(0.5)
    # This term explodes if alpha -> 0, punishing the model.
    term_norm = log_scale_daily + torch.lgamma(1.0/alpha) - torch.log(alpha) - (1.0/alpha)*np.log(0.5)
    
    nll = torch.mean(term_error + term_norm)
    
    # Regularization (TV) for the Volatility Curve
    u_grad = torch.autograd.grad(log_sigma_annual, t_train, torch.ones_like(log_sigma_annual), create_graph=True)[0]
    tv_loss = torch.mean(torch.abs(u_grad))
    
    # Annealing: Let physics stabilize before enforcing smoothness
    tv_weight = 0.0 if epoch < 1000 else 0.05
    
    total_loss = nll + tv_weight * tv_loss
    
    total_loss.backward()
    optimizer.step()
    
    if epoch % 500 == 0:
        print(f"Ep {epoch:<4} | Alpha: {alpha.item():.4f} (Target: 2.0) | Loss: {nll.item():.4f}")

# ==========================================
# 4. VISUALIZATION
# ==========================================
print(f"\nFinal Discovered Physics Parameter (Alpha): {model.get_alpha().item():.4f}")

with torch.no_grad():
    learned_vol = torch.exp(model.get_log_sigma(t_train)).numpy()

plt.figure(figsize=(10, 10))

# Plot 1: The Physics Discovery (Alpha Convergence)
plt.subplot(3, 1, 1)
plt.plot(alpha_history, 'r-', linewidth=3)
plt.axhline(y=2.0, color='k', linestyle='--', label='Ground Truth (Gaussian)')
plt.axhline(y=1.0, color='grey', linestyle=':', label='Laplace (Fat Tail)')
plt.title("1. Scientific Discovery: Learning the Exponent", fontsize=14)
plt.ylabel("Alpha Parameter")
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: The Volatility Discovery
plt.subplot(3, 1, 2)
plt.plot(t_train.detach().numpy(), sigma_true.numpy(), 'k--', label='True Volatility', linewidth=2)
plt.plot(t_train.detach().numpy(), learned_vol, 'b-', label='KAN Discovery', linewidth=3)
plt.title("2. Resulting Volatility Curve (Using Learned Physics)", fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 3: What happened?
plt.subplot(3, 1, 3)
plt.plot(returns_train.numpy(), color='grey', alpha=0.5, label='Returns')
plt.title("3. The Data Source", fontsize=14)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()