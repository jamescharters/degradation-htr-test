import torch
import torch.nn as nn
from kan import KAN
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import square

# --- 1. Setup and Data Generation ---
def generate_data(n_points=400):
    x = torch.linspace(-2 * np.pi, 2 * np.pi, n_points).view(-1, 1)
    y = torch.from_numpy(square(x.numpy())).view(-1, 1).float()
    y += torch.randn(y.shape) * 0.05
    return x, y

X, y = generate_data()

# --- 2. Model Definitions ---
# A standard MLP
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )
    def forward(self, x):
        return self.layers(x)

# A small KAN
kan_model = KAN(width=[1, 5, 1], grid=10, k=3, seed=42)
mlp_model = SimpleMLP()

# --- 3. Training Function (Greatly Simplified and Corrected) ---
# The logic is now identical for both MLP and KAN, as the modern pykan API
# behaves just like a standard nn.Module.
def train_model(model, X, y, steps=5000, lr=1e-3):
    # Regularization is handled by the optimizer's weight_decay, a standard practice.
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    
    print(f"--- Training {model.__class__.__name__} ---")
    for step in range(steps):
        
        # FORWARD PASS: Now identical for both models.
        pred = model(X)
        
        loss = loss_fn(pred, y)
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 500 == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}")
    
    final_loss = loss.item()
    print(f"Final Loss for {model.__class__.__name__}: {final_loss:.4f}\n")
    return final_loss


# --- 4. Train Both Models ---
train_model(mlp_model, X, y)
train_model(kan_model, X, y)


# --- 5. Visualization (Simplified and Corrected) ---
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(10, 6))

x_plot = torch.linspace(-2 * np.pi, 2 * np.pi, 1000).view(-1, 1)

with torch.no_grad():
    mlp_pred = mlp_model(x_plot)
    
    # KAN forward pass is now simple, just like any other model.
    kan_pred = kan_model(x_plot)

ax.scatter(X.numpy(), y.numpy(), label='Original Noisy Data', alpha=0.3, s=15, color='gray')
ax.plot(x_plot.numpy(), mlp_pred.numpy(), label='MLP Prediction', color='red', linewidth=2.5)
ax.plot(x_plot.numpy(), kan_pred.numpy(), label='KAN Prediction', color='blue', linewidth=2.5)

ax.set_title("KAN vs. MLP on a Square Wave Function", fontsize=16)
ax.set_xlabel("Input (x)", fontsize=12)
ax.set_ylabel("Output (y)", fontsize=12)
ax.legend(fontsize=12)
ax.set_ylim(-1.5, 1.5)
plt.show()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

mlp_params = count_parameters(mlp_model)
print(f"MLP Parameters: {mlp_params}")

kan_params = count_parameters(kan_model)
print(f"KAN Parameters: {kan_params}")