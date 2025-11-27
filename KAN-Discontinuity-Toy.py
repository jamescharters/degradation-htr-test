import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from kan import KAN

# For reproducibility
torch.manual_seed(42)
np.random.seed(42)

# --- 1. Define the Square Wave Function ---
def square_wave(x, period=2.0, amplitude=1.0):
    """A square wave function"""
    phase = (x % period) / period
    return np.where(phase < 0.5, amplitude, -amplitude)

# --- 2. Fourier Feature Network (Alternative Basis) ---
class FourierFeatureNetwork(nn.Module):
    """
    Neural network using Fourier features as basis functions
    
    Instead of splines (smooth, local basis), use Fourier basis:
    - sin(ωx), cos(ωx) for various frequencies ω
    - These are GLOBAL basis functions (each affects entire domain)
    - Naturally represent periodic functions
    - Can represent discontinuities through superposition
    
    This is more appropriate for periodic signals with jumps.
    """
    def __init__(self, n_frequencies=10):
        super().__init__()
        self.n_frequencies = n_frequencies
        
        # Learnable frequency scaling and phase
        # Start with frequencies that match the square wave period
        base_freqs = torch.arange(1, n_frequencies + 1) * np.pi
        self.frequencies = nn.Parameter(base_freqs.float())
        
        # Linear layer to combine Fourier features
        # Input: 2*n_frequencies features (sin and cos for each frequency)
        self.linear = nn.Linear(2 * n_frequencies, 1)
        
    def forward(self, x):
        # Compute Fourier features: [sin(f1*x), cos(f1*x), sin(f2*x), cos(f2*x), ...]
        features = []
        for freq in self.frequencies:
            features.append(torch.sin(freq * x))
            features.append(torch.cos(freq * x))
        
        # Stack features: shape (batch_size, 2*n_frequencies)
        features = torch.cat(features, dim=1)
        
        # Linear combination of features
        return self.linear(features)

# --- 3. Piecewise Constant Network (Best basis for this problem!) ---
class PiecewiseConstantNetwork(nn.Module):
    """
    Neural network using piecewise constant basis functions
    
    This is the IDEAL basis for square waves:
    - Each basis function is constant in an interval, zero elsewhere
    - Discontinuities are natural (not fighting the basis)
    - This is essentially a learned binning/histogram approach
    
    Think of it as learnable step functions.
    """
    def __init__(self, n_bins=20, x_min=-2, x_max=2):
        super().__init__()
        self.n_bins = n_bins
        self.x_min = x_min
        self.x_max = x_max
        
        # Learnable value for each bin
        self.bin_values = nn.Parameter(torch.randn(n_bins))
        
        # Bin edges (evenly spaced)
        self.bin_edges = torch.linspace(x_min, x_max, n_bins + 1)
        
    def forward(self, x):
        # Determine which bin each x falls into
        # Digitize returns bin indices
        x_np = x.detach().cpu().numpy().squeeze()
        bin_indices = np.digitize(x_np, self.bin_edges.numpy()) - 1
        
        # Clip to valid range [0, n_bins-1]
        bin_indices = np.clip(bin_indices, 0, self.n_bins - 1)
        
        # Return the learned value for each bin
        return self.bin_values[bin_indices].unsqueeze(1)

# --- 4. Generate Training Data ---
n_points = 50
x_train = np.linspace(-2, 2, n_points)
y_train = square_wave(x_train, period=2.0, amplitude=1.0)

x_train_tensor = torch.tensor(x_train, dtype=torch.float32).view(-1, 1)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

# --- 5. Train Standard KAN (Spline Basis) ---
print("=" * 70)
print("BASELINE: KAN with SPLINE basis functions")
print("=" * 70)

model_kan = KAN(width=[1, 15, 1], grid=50, k=3, seed=0)
optimizer_kan = torch.optim.Adam(model_kan.parameters(), lr=1e-2, weight_decay=1e-4)
loss_fn = nn.MSELoss()

steps = 300
for step in range(steps):
    pred = model_kan(x_train_tensor)
    loss = loss_fn(pred, y_train_tensor)
    optimizer_kan.zero_grad()
    loss.backward()
    optimizer_kan.step()
    
    if step % 60 == 0:
        print(f"Step {step}/{steps}, Loss: {loss.item():.6f}")

print("KAN training finished.\n")

# --- 6. Train Fourier Feature Network ---
print("=" * 70)
print("COMPARISON 1: Network with FOURIER basis functions")
print("=" * 70)

model_fourier = FourierFeatureNetwork(n_frequencies=15)
optimizer_fourier = torch.optim.Adam(model_fourier.parameters(), lr=1e-2)

steps = 300
for step in range(steps):
    pred = model_fourier(x_train_tensor)
    loss = loss_fn(pred, y_train_tensor)
    optimizer_fourier.zero_grad()
    loss.backward()
    optimizer_fourier.step()
    
    if step % 60 == 0:
        print(f"Step {step}/{steps}, Loss: {loss.item():.6f}")

print("Fourier network training finished.\n")

# --- 7. Train Piecewise Constant Network ---
print("=" * 70)
print("COMPARISON 2: Network with PIECEWISE CONSTANT basis (ideal for square waves)")
print("=" * 70)

model_piecewise = PiecewiseConstantNetwork(n_bins=25, x_min=-2, x_max=2)
optimizer_piecewise = torch.optim.Adam(model_piecewise.parameters(), lr=1e-2)

steps = 300
for step in range(steps):
    pred = model_piecewise(x_train_tensor)
    loss = loss_fn(pred, y_train_tensor)
    optimizer_piecewise.zero_grad()
    loss.backward()
    optimizer_piecewise.step()
    
    if step % 60 == 0:
        print(f"Step {step}/{steps}, Loss: {loss.item():.6f}")

print("Piecewise constant network training finished.\n")

# --- 8. Evaluate All Models ---
x_test = np.linspace(-2, 2, 1000)
x_test_tensor = torch.tensor(x_test, dtype=torch.float32).view(-1, 1)

with torch.no_grad():
    y_kan = model_kan(x_test_tensor).numpy()
    y_fourier = model_fourier(x_test_tensor).numpy()
    y_piecewise = model_piecewise(x_test_tensor).numpy()

# --- 9. Visualization ---
fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# KAN (Spline basis)
axes[0].set_title("KAN with Spline Basis - Gibbs Ringing", fontsize=14, fontweight='bold')
axes[0].plot(x_test, square_wave(x_test), 'k--', label='True Square Wave', linewidth=2, alpha=0.7)
axes[0].scatter(x_train, y_train, label='Training Data (50 points)', color='red', s=20, alpha=0.5, zorder=5)
axes[0].plot(x_test, y_kan, 'b-', label='KAN (Spline Basis)', linewidth=2.5)
axes[0].set_ylabel("f(x)", fontsize=11)
axes[0].legend(fontsize=10)
axes[0].grid(True, linestyle='--', alpha=0.6)
axes[0].set_ylim(-1.5, 1.5)

# Fourier basis
axes[1].set_title("Neural Network with Fourier Basis - Natural for Periodic Functions", 
                  fontsize=14, fontweight='bold', color='purple')
axes[1].plot(x_test, square_wave(x_test), 'k--', label='True Square Wave', linewidth=2, alpha=0.7)
axes[1].scatter(x_train, y_train, label='Training Data (50 points)', color='red', s=20, alpha=0.5, zorder=5)
axes[1].plot(x_test, y_fourier, color='purple', linewidth=2.5, label='Fourier Basis')
axes[1].set_ylabel("f(x)", fontsize=11)
axes[1].legend(fontsize=10)
axes[1].grid(True, linestyle='--', alpha=0.6)
axes[1].set_ylim(-1.5, 1.5)

# Piecewise constant basis
axes[2].set_title("Neural Network with Piecewise Constant Basis - Ideal for Discontinuities", 
                  fontsize=14, fontweight='bold', color='green')
axes[2].plot(x_test, square_wave(x_test), 'k--', label='True Square Wave', linewidth=2, alpha=0.7)
axes[2].scatter(x_train, y_train, label='Training Data (50 points)', color='red', s=20, alpha=0.5, zorder=5)
axes[2].plot(x_test, y_piecewise, 'g-', label='Piecewise Constant Basis', linewidth=2.5)
axes[2].set_xlabel("x", fontsize=12)
axes[2].set_ylabel("f(x)", fontsize=11)
axes[2].legend(fontsize=10)
axes[2].grid(True, linestyle='--', alpha=0.6)
axes[2].set_ylim(-1.5, 1.5)

plt.tight_layout()
plt.savefig('kan_basis_comparison.png', dpi=150, bbox_inches='tight')
print("Comparison plot saved as 'kan_basis_comparison.png'")
plt.show()

# --- 10. Quantitative Analysis ---
print("\n" + "=" * 70)
print("QUANTITATIVE COMPARISON: Oscillation in flat region [-0.5, 0]")
print("=" * 70)

flat_region_mask = (x_test > -0.5) & (x_test < 0)
kan_std = np.std(y_kan[flat_region_mask])
fourier_std = np.std(y_fourier[flat_region_mask])
piecewise_std = np.std(y_piecewise[flat_region_mask])

print(f"\nStandard deviation in flat region:")
print(f"  KAN (Spline):              {kan_std:.6f}")
print(f"  Fourier Basis:             {fourier_std:.6f}")
print(f"  Piecewise Constant:        {piecewise_std:.6f}")

print("\n" + "=" * 70)
print("KEY INSIGHT: The choice of basis functions matters!")
print()
print("SPLINE basis (KAN): Smooth, local → fights discontinuities")
print("FOURIER basis: Global, periodic → natural for square waves")
print("PIECEWISE CONSTANT: Step functions → IDEAL for discontinuities")
print()
print("For discontinuous functions, matching the basis to the problem")
print("structure is more effective than any regularization technique!")
print("=" * 70)