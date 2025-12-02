"""
Van der Pol Oscillator Inverse Problem: KAN vs MLP-PINN Comparison

This code compares KAN-based and MLP-based Physics-Informed Neural Networks
for identifying the unknown parameter μ in the Van der Pol equation.

Equation: ẍ - μ(1-x²)ẋ + x = 0
Rewritten as system: ẋ₁ = x₂, ẋ₂ = μ(1-x₁²)x₂ - x₁

Goal: Identify μ from sparse, noisy observations of x(t)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, List
import time

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# =============================================================================
# PART 1: GROUND TRUTH DATA GENERATION
# =============================================================================

def van_der_pol_ode(state: np.ndarray, t: float, mu: float) -> np.ndarray:
    """
    Van der Pol oscillator ODE system.
    
    Args:
        state: [x1, x2] where x1=x, x2=ẋ
        t: time
        mu: stiffness parameter
        
    Returns:
        derivatives [ẋ₁, ẋ₂]
    """
    x1, x2 = state
    dx1_dt = x2
    dx2_dt = mu * (1 - x1**2) * x2 - x1
    return [dx1_dt, dx2_dt]


def generate_ground_truth_data(mu_true: float = 100.0, 
                               t_span: Tuple[float, float] = (0, 20),
                               n_points: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate ground truth solution using scipy's ODE solver.
    
    Args:
        mu_true: true value of μ parameter
        t_span: (start_time, end_time)
        n_points: number of time points
        
    Returns:
        t: time array
        solution: [n_points, 2] array with [x1, x2] values
    """
    print(f"Generating ground truth data with μ = {mu_true}...")
    
    # Initial conditions
    initial_state = [2.0, 0.0]  # [x(0), ẋ(0)]
    
    # Time points
    t = np.linspace(t_span[0], t_span[1], n_points)
    
    # Solve ODE
    solution = odeint(van_der_pol_ode, initial_state, t, args=(mu_true,))
    
    print(f"Ground truth generated: x1 range [{solution[:, 0].min():.3f}, {solution[:, 0].max():.3f}]")
    return t, solution


def create_training_data(t: np.ndarray, 
                         solution: np.ndarray, 
                         n_obs: int = 20,
                         noise_level: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sparse, noisy observations for training.
    
    Args:
        t: full time array
        solution: full solution array
        n_obs: number of observation points
        noise_level: standard deviation of Gaussian noise (as fraction of signal std)
        
    Returns:
        t_obs: observed time points
        x_obs: noisy observations [n_obs, 2]
    """
    print(f"\nCreating training data: {n_obs} observations with {noise_level*100}% noise...")
    
    # Sample uniformly from the time span
    indices = np.linspace(0, len(t)-1, n_obs, dtype=int)
    t_obs = t[indices]
    x_clean = solution[indices]
    
    # Add Gaussian noise
    noise_std = noise_level * np.std(x_clean, axis=0)
    noise = np.random.normal(0, noise_std, x_clean.shape)
    x_obs = x_clean + noise
    
    print(f"Observation times: [{t_obs[0]:.2f}, {t_obs[-1]:.2f}]")
    print(f"Noise std: x1={noise_std[0]:.4f}, x2={noise_std[1]:.4f}")
    
    return t_obs, x_obs


# =============================================================================
# PART 2: NEURAL NETWORK ARCHITECTURES
# =============================================================================

class MLP_PINN(nn.Module):
    """
    Standard Multi-Layer Perceptron for Physics-Informed Neural Network.
    Architecture: [1] -> [64] -> [64] -> [64] -> [2]
    """
    
    def __init__(self, hidden_size: int = 64, n_layers: int = 3):
        super(MLP_PINN, self).__init__()
        
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        
        # Build layers
        layers = []
        layers.append(nn.Linear(1, hidden_size))  # Input: time
        layers.append(nn.Tanh())
        
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.Tanh())
        
        layers.append(nn.Linear(hidden_size, 2))  # Output: [x1, x2]
        
        self.network = nn.Sequential(*layers)
        
        # Learnable parameter μ (initialize near 1.0, far from true value)
        self.mu = nn.Parameter(torch.tensor([1.0]))
        
        print(f"\nMLP-PINN Architecture:")
        print(f"  Layers: {n_layers} hidden layers")
        print(f"  Hidden size: {hidden_size}")
        print(f"  Total parameters: {sum(p.numel() for p in self.parameters())}")
        print(f"  Initial μ: {self.mu.item():.4f}")
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            t: time tensor [n, 1]
            
        Returns:
            state: [n, 2] tensor with [x1, x2]
        """
        return self.network(t)


class KAN_Layer(nn.Module):
    """
    Kolmogorov-Arnold Network layer with learnable B-spline activations.
    Each connection has its own learnable activation function.
    """
    
    def __init__(self, in_features: int, out_features: int, grid_size: int = 5, spline_order: int = 3):
        super(KAN_Layer, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        
        # Learnable spline coefficients for each edge
        # Shape: [out_features, in_features, grid_size + spline_order]
        self.spline_coeffs = nn.Parameter(
            torch.randn(out_features, in_features, grid_size + spline_order) * 0.1
        )
        
        # Grid points (fixed, not learned)
        self.register_buffer('grid', torch.linspace(-1, 1, grid_size + 2 * spline_order + 1))
        
    def b_spline_basis(self, x: torch.Tensor, i: int, k: int) -> torch.Tensor:
        """
        Compute B-spline basis function using Cox-de Boor recursion.
        
        Args:
            x: input tensor
            i: basis function index
            k: spline order
            
        Returns:
            basis function values
        """
        if k == 0:
            return ((x >= self.grid[i]) & (x < self.grid[i + 1])).float()
        
        # Recursive computation
        left_term = torch.zeros_like(x)
        right_term = torch.zeros_like(x)
        
        denom_left = self.grid[i + k] - self.grid[i]
        if denom_left > 1e-6:
            left_term = (x - self.grid[i]) / denom_left * self.b_spline_basis(x, i, k - 1)
        
        denom_right = self.grid[i + k + 1] - self.grid[i + 1]
        if denom_right > 1e-6:
            right_term = (self.grid[i + k + 1] - x) / denom_right * self.b_spline_basis(x, i + 1, k - 1)
        
        return left_term + right_term
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through KAN layer.
        
        Args:
            x: input tensor [batch, in_features]
            
        Returns:
            output tensor [batch, out_features]
        """
        batch_size = x.shape[0]
        
        # Normalize input to [-1, 1]
        x_normalized = torch.tanh(x)
        
        # Initialize output
        output = torch.zeros(batch_size, self.out_features, device=x.device)
        
        # For each output neuron
        for j in range(self.out_features):
            # For each input neuron
            for i in range(self.in_features):
                # Compute activation function using B-splines
                x_in = x_normalized[:, i]
                
                # Evaluate spline
                phi = torch.zeros_like(x_in)
                for k in range(self.grid_size + self.spline_order):
                    basis = self.b_spline_basis(x_in, k, self.spline_order)
                    phi += self.spline_coeffs[j, i, k] * basis
                
                output[:, j] += phi
        
        return output


class KAN_PINN(nn.Module):
    """
    KAN-based Physics-Informed Neural Network.
    Uses learnable activation functions on edges instead of fixed activations.
    """
    
    def __init__(self, hidden_size: int = 32, n_layers: int = 3, grid_size: int = 5):
        super(KAN_PINN, self).__init__()
        
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        
        # Build KAN layers
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(KAN_Layer(1, hidden_size, grid_size=grid_size))
        
        # Hidden layers
        for _ in range(n_layers - 1):
            self.layers.append(KAN_Layer(hidden_size, hidden_size, grid_size=grid_size))
        
        # Output layer
        self.layers.append(KAN_Layer(hidden_size, 2, grid_size=grid_size))
        
        # Learnable parameter μ
        self.mu = nn.Parameter(torch.tensor([1.0]))
        
        print(f"\nKAN-PINN Architecture:")
        print(f"  Layers: {n_layers} hidden KAN layers")
        print(f"  Hidden size: {hidden_size}")
        print(f"  Grid size: {grid_size}")
        print(f"  Total parameters: {sum(p.numel() for p in self.parameters())}")
        print(f"  Initial μ: {self.mu.item():.4f}")
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            t: time tensor [n, 1]
            
        Returns:
            state: [n, 2] tensor with [x1, x2]
        """
        x = t
        for layer in self.layers:
            x = layer(x)
        return x


# =============================================================================
# PART 3: PHYSICS-INFORMED LOSS FUNCTIONS
# =============================================================================

def compute_physics_loss(model: nn.Module, 
                        t_physics: torch.Tensor) -> torch.Tensor:
    """
    Compute physics loss (PDE residual).
    
    The Van der Pol equation: ẍ - μ(1-x²)ẋ + x = 0
    As a system: ẋ₁ = x₂, ẋ₂ = μ(1-x₁²)x₂ - x₁
    
    Args:
        model: PINN model
        t_physics: time points for physics loss [n, 1]
        
    Returns:
        physics loss (mean squared residual)
    """
    # Enable gradient computation
    t_physics.requires_grad_(True)
    
    # Forward pass
    state = model(t_physics)  # [n, 2]
    x1 = state[:, 0:1]  # [n, 1]
    x2 = state[:, 1:2]  # [n, 1]
    
    # Compute derivatives using automatic differentiation
    dx1_dt = torch.autograd.grad(x1, t_physics, 
                                  grad_outputs=torch.ones_like(x1),
                                  create_graph=True)[0]
    
    dx2_dt = torch.autograd.grad(x2, t_physics,
                                  grad_outputs=torch.ones_like(x2),
                                  create_graph=True)[0]
    
    # Physics equations
    residual_1 = dx1_dt - x2
    residual_2 = dx2_dt - model.mu * (1 - x1**2) * x2 + x1
    
    # Mean squared error
    loss = torch.mean(residual_1**2) + torch.mean(residual_2**2)
    
    return loss


def compute_data_loss(model: nn.Module,
                     t_obs: torch.Tensor,
                     x_obs: torch.Tensor) -> torch.Tensor:
    """
    Compute data loss (observation fitting).
    
    Args:
        model: PINN model
        t_obs: observed time points [n, 1]
        x_obs: observed states [n, 2]
        
    Returns:
        data loss (mean squared error)
    """
    # Forward pass
    state_pred = model(t_obs)
    
    # Mean squared error
    loss = torch.mean((state_pred - x_obs)**2)
    
    return loss


def compute_initial_condition_loss(model: nn.Module,
                                   t0: torch.Tensor,
                                   x0: torch.Tensor) -> torch.Tensor:
    """
    Compute initial condition loss.
    
    Args:
        model: PINN model
        t0: initial time [1, 1]
        x0: initial state [1, 2]
        
    Returns:
        initial condition loss
    """
    state_pred = model(t0)
    loss = torch.mean((state_pred - x0)**2)
    return loss


def compute_total_loss(model: nn.Module,
                      t_physics: torch.Tensor,
                      t_obs: torch.Tensor,
                      x_obs: torch.Tensor,
                      t0: torch.Tensor,
                      x0: torch.Tensor,
                      lambda_physics: float = 1.0,
                      lambda_data: float = 100.0,
                      lambda_ic: float = 100.0) -> Tuple[torch.Tensor, dict]:
    """
    Compute total weighted loss.
    
    Args:
        model: PINN model
        t_physics: time points for physics loss
        t_obs: observed time points
        x_obs: observed states
        t0: initial time
        x0: initial state
        lambda_physics: weight for physics loss
        lambda_data: weight for data loss
        lambda_ic: weight for initial condition loss
        
    Returns:
        total_loss: weighted sum of losses
        loss_dict: dictionary with individual losses
    """
    loss_physics = compute_physics_loss(model, t_physics)
    loss_data = compute_data_loss(model, t_obs, x_obs)
    loss_ic = compute_initial_condition_loss(model, t0, x0)
    
    total_loss = (lambda_physics * loss_physics + 
                  lambda_data * loss_data + 
                  lambda_ic * loss_ic)
    
    loss_dict = {
        'total': total_loss.item(),
        'physics': loss_physics.item(),
        'data': loss_data.item(),
        'ic': loss_ic.item()
    }
    
    return total_loss, loss_dict


# =============================================================================
# PART 4: TRAINING FUNCTIONS
# =============================================================================

class TrainingHistory:
    """Track training history."""
    
    def __init__(self):
        self.epochs = []
        self.losses = {'total': [], 'physics': [], 'data': [], 'ic': []}
        self.mu_values = []
        self.times = []
    
    def update(self, epoch: int, loss_dict: dict, mu: float, elapsed_time: float):
        self.epochs.append(epoch)
        for key, value in loss_dict.items():
            self.losses[key].append(value)
        self.mu_values.append(mu)
        self.times.append(elapsed_time)


def train_pinn(model: nn.Module,
               t_physics: np.ndarray,
               t_obs: np.ndarray,
               x_obs: np.ndarray,
               mu_true: float,
               n_epochs: int = 5000,
               lr: float = 1e-3,
               print_every: int = 500) -> TrainingHistory:
    """
    Train a PINN model.
    
    Args:
        model: PINN model (MLP or KAN)
        t_physics: collocation points for physics loss
        t_obs: observed time points
        x_obs: observed states
        mu_true: true value of μ (for monitoring)
        n_epochs: number of training epochs
        lr: learning rate
        print_every: print frequency
        
    Returns:
        training history
    """
    # Convert to tensors
    t_physics_tensor = torch.FloatTensor(t_physics.reshape(-1, 1))
    t_obs_tensor = torch.FloatTensor(t_obs.reshape(-1, 1))
    x_obs_tensor = torch.FloatTensor(x_obs)
    t0_tensor = torch.FloatTensor([[0.0]])
    x0_tensor = torch.FloatTensor([[2.0, 0.0]])
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                      factor=0.5, patience=500)
    
    # Training history
    history = TrainingHistory()
    
    # Training loop
    start_time = time.time()
    print(f"\nStarting training for {n_epochs} epochs...")
    print(f"True μ = {mu_true:.4f}")
    print("-" * 80)
    
    for epoch in range(n_epochs):
        # Zero gradients
        optimizer.zero_grad()
        
        # Compute loss
        total_loss, loss_dict = compute_total_loss(
            model, t_physics_tensor, t_obs_tensor, x_obs_tensor,
            t0_tensor, x0_tensor
        )
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping (helps with stability)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update parameters
        optimizer.step()
        
        # Update scheduler
        scheduler.step(total_loss)
        
        # Record history
        elapsed_time = time.time() - start_time
        history.update(epoch, loss_dict, model.mu.item(), elapsed_time)
        
        # Print progress
        if epoch % print_every == 0 or epoch == n_epochs - 1:
            mu_current = model.mu.item()
            mu_error = abs(mu_current - mu_true) / mu_true * 100
            print(f"Epoch {epoch:5d} | Loss: {loss_dict['total']:.6f} | "
                  f"Physics: {loss_dict['physics']:.6f} | Data: {loss_dict['data']:.6f} | "
                  f"μ: {mu_current:7.3f} (error: {mu_error:5.2f}%) | "
                  f"Time: {elapsed_time:.2f}s")
    
    print("-" * 80)
    print(f"Training completed in {elapsed_time:.2f}s")
    print(f"Final μ: {model.mu.item():.4f} (true: {mu_true:.4f})")
    print(f"Final error: {abs(model.mu.item() - mu_true) / mu_true * 100:.2f}%")
    
    return history


# =============================================================================
# PART 5: VISUALIZATION FUNCTIONS
# =============================================================================

def plot_training_comparison(history_mlp: TrainingHistory,
                            history_kan: TrainingHistory,
                            mu_true: float,
                            save_path: str = 'training_comparison.png'):
    """
    Plot training curves comparing MLP and KAN.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Total loss
    ax = axes[0, 0]
    ax.semilogy(history_mlp.epochs, history_mlp.losses['total'], 
                label='MLP-PINN', linewidth=2, alpha=0.8)
    ax.semilogy(history_kan.epochs, history_kan.losses['total'], 
                label='KAN-PINN', linewidth=2, alpha=0.8)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Total Loss (log scale)', fontsize=12)
    ax.set_title('Total Loss Convergence', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Physics loss
    ax = axes[0, 1]
    ax.semilogy(history_mlp.epochs, history_mlp.losses['physics'], 
                label='MLP-PINN', linewidth=2, alpha=0.8)
    ax.semilogy(history_kan.epochs, history_kan.losses['physics'], 
                label='KAN-PINN', linewidth=2, alpha=0.8)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Physics Loss (log scale)', fontsize=12)
    ax.set_title('Physics Loss (PDE Residual)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Parameter evolution
    ax = axes[1, 0]
    ax.plot(history_mlp.epochs, history_mlp.mu_values, 
            label='MLP-PINN', linewidth=2, alpha=0.8)
    ax.plot(history_kan.epochs, history_kan.mu_values, 
            label='KAN-PINN', linewidth=2, alpha=0.8)
    ax.axhline(y=mu_true, color='red', linestyle='--', linewidth=2, 
               label=f'True μ = {mu_true}')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('μ Value', fontsize=12)
    ax.set_title('Parameter Identification', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Parameter error
    ax = axes[1, 1]
    mlp_error = [abs(mu - mu_true) / mu_true * 100 for mu in history_mlp.mu_values]
    kan_error = [abs(mu - mu_true) / mu_true * 100 for mu in history_kan.mu_values]
    ax.semilogy(history_mlp.epochs, mlp_error, 
                label='MLP-PINN', linewidth=2, alpha=0.8)
    ax.semilogy(history_kan.epochs, kan_error, 
                label='KAN-PINN', linewidth=2, alpha=0.8)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Parameter Error (%)', fontsize=12)
    ax.set_title('Parameter Identification Error', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nTraining comparison plot saved to: {save_path}")
    plt.close()


def plot_solution_comparison(t_true: np.ndarray,
                            x_true: np.ndarray,
                            t_obs: np.ndarray,
                            x_obs: np.ndarray,
                            model_mlp: nn.Module,
                            model_kan: nn.Module,
                            save_path: str = 'solution_comparison.png'):
    """
    Plot solution predictions from both models.
    """
    # Generate predictions
    t_pred = torch.FloatTensor(t_true.reshape(-1, 1))
    
    with torch.no_grad():
        x_pred_mlp = model_mlp(t_pred).numpy()
        x_pred_kan = model_kan(t_pred).numpy()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # x1 time series
    ax = axes[0, 0]
    ax.plot(t_true, x_true[:, 0], 'k-', linewidth=2, label='Ground Truth', alpha=0.7)
    ax.plot(t_true, x_pred_mlp[:, 0], '--', linewidth=2, label='MLP-PINN', alpha=0.8)
    ax.plot(t_true, x_pred_kan[:, 0], ':', linewidth=2, label='KAN-PINN', alpha=0.8)
    ax.scatter(t_obs, x_obs[:, 0], s=100, c='red', marker='o', 
               label='Observations', zorder=5, edgecolors='black', linewidth=1)
    ax.set_xlabel('Time t', fontsize=12)
    ax.set_ylabel('x₁(t)', fontsize=12)
    ax.set_title('State x₁ vs Time', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # x2 time series
    ax = axes[0, 1]
    ax.plot(t_true, x_true[:, 1], 'k-', linewidth=2, label='Ground Truth', alpha=0.7)
    ax.plot(t_true, x_pred_mlp[:, 1], '--', linewidth=2, label='MLP-PINN', alpha=0.8)
    ax.plot(t_true, x_pred_kan[:, 1], ':', linewidth=2, label='KAN-PINN', alpha=0.8)
    ax.scatter(t_obs, x_obs[:, 1], s=100, c='red', marker='o', 
               label='Observations', zorder=5, edgecolors='black', linewidth=1)
    ax.set_xlabel('Time t', fontsize=12)
    ax.set_ylabel('x₂(t)', fontsize=12)
    ax.set_title('State x₂ vs Time', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Phase portrait
    ax = axes[1, 0]
    ax.plot(x_true[:, 0], x_true[:, 1], 'k-', linewidth=2, 
            label='Ground Truth', alpha=0.7)
    ax.plot(x_pred_mlp[:, 0], x_pred_mlp[:, 1], '--', linewidth=2, 
            label='MLP-PINN', alpha=0.8)
    ax.plot(x_pred_kan[:, 0], x_pred_kan[:, 1], ':', linewidth=2, 
            label='KAN-PINN', alpha=0.8)
    ax.scatter(x_obs[:, 0], x_obs[:, 1], s=100, c='red', marker='o', 
               label='Observations', zorder=5, edgecolors='black', linewidth=1)
    ax.set_xlabel('x₁', fontsize=12)
    ax.set_ylabel('x₂', fontsize=12)
    ax.set_title('Phase Portrait', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # Prediction errors
    ax = axes[1, 1]
    error_mlp = np.sqrt(np.sum((x_pred_mlp - x_true)**2, axis=1))
    error_kan = np.sqrt(np.sum((x_pred_kan - x_true)**2, axis=1))
    ax.plot(t_true, error_mlp, '--', linewidth=2, label='MLP-PINN', alpha=0.8)
    ax.plot(t_true, error_kan, ':', linewidth=2, label='KAN-PINN', alpha=0.8)
    ax.set_xlabel('Time t', fontsize=12)
    ax.set_ylabel('L2 Error', fontsize=12)
    ax.set_title('Prediction Error vs Time', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Solution comparison plot saved to: {save_path}")
    plt.close()


def visualize_kan_activation_functions(model_kan: KAN_PINN,
                                       save_path: str = 'kan_activations.png'):
    """
    Visualize the learned activation functions in the KAN.
    This shows what the KAN has learned about the problem structure.
    """
    print("\nVisualizing KAN activation functions...")
    
    # Sample input range
    x_range = torch.linspace(-1, 1, 200).reshape(-1, 1)
    
    # Get first KAN layer (most interpretable)
    first_layer = model_kan.layers[0]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    # Visualize activation functions for each output of first layer
    for out_idx in range(min(6, first_layer.out_features)):
        ax = axes[out_idx]
        
        # Input index 0 (since input is 1D: time)
        in_idx = 0
        
        # Compute activation function
        with torch.no_grad():
            phi = torch.zeros(len(x_range))
            for k in range(first_layer.grid_size + first_layer.spline_order):
                basis = first_layer.b_spline_basis(x_range.squeeze(), k, 
                                                   first_layer.spline_order)
                phi += first_layer.spline_coeffs[out_idx, in_idx, k] * basis
        
        # Plot
        ax.plot(x_range.numpy(), phi.numpy(), linewidth=2.5, color='blue')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
        ax.set_xlabel('Input (normalized time)', fontsize=11)
        ax.set_ylabel('Activation φ(x)', fontsize=11)
        ax.set_title(f'KAN Layer 1: Output Neuron {out_idx+1}', 
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add reference lines for common activation shapes
        x_np = x_range.numpy()
        ax.plot(x_np, np.tanh(2*x_np), '--', alpha=0.3, color='red', 
                linewidth=1, label='tanh(2x)')
        ax.plot(x_np, x_np**2, '--', alpha=0.3, color='green', 
                linewidth=1, label='x²')
        ax.plot(x_np, np.exp(-x_np**2), '--', alpha=0.3, color='orange', 
                linewidth=1, label='exp(-x²)')
        ax.legend(fontsize=8, loc='best')
    
    plt.suptitle('Learned KAN Activation Functions (First Layer)', 
                 fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"KAN activation visualization saved to: {save_path}")
    plt.close()


def print_performance_summary(history_mlp: TrainingHistory,
                             history_kan: TrainingHistory,
                             mu_true: float):
    """
    Print detailed performance comparison.
    """
    print("\n" + "=" * 80)
    print("PERFORMANCE SUMMARY")
    print("=" * 80)
    
    # Final parameter errors
    mlp_mu_final = history_mlp.mu_values[-1]
    kan_mu_final = history_kan.mu_values[-1]
    mlp_error = abs(mlp_mu_final - mu_true) / mu_true * 100
    kan_error = abs(kan_mu_final - mu_true) / mu_true * 100
    
    print(f"\n1. PARAMETER IDENTIFICATION ACCURACY:")
    print(f"   True μ:              {mu_true:.4f}")
    print(f"   MLP-PINN final μ:    {mlp_mu_final:.4f} (error: {mlp_error:.2f}%)")
    print(f"   KAN-PINN final μ:    {kan_mu_final:.4f} (error: {kan_error:.2f}%)")
    print(f"   → KAN improvement:   {(mlp_error - kan_error):.2f}% better")
    
    # Training efficiency
    mlp_time = history_mlp.times[-1]
    kan_time = history_kan.times[-1]
    
    print(f"\n2. TRAINING TIME:")
    print(f"   MLP-PINN:            {mlp_time:.2f} seconds")
    print(f"   KAN-PINN:            {kan_time:.2f} seconds")
    print(f"   → Time ratio:        {kan_time/mlp_time:.2f}x")
    
    # Convergence speed (epochs to reach 10% error)
    mlp_epochs_to_10pct = None
    kan_epochs_to_10pct = None
    
    for i, mu in enumerate(history_mlp.mu_values):
        if abs(mu - mu_true) / mu_true * 100 < 10:
            mlp_epochs_to_10pct = i
            break
    
    for i, mu in enumerate(history_kan.mu_values):
        if abs(mu - mu_true) / mu_true * 100 < 10:
            kan_epochs_to_10pct = i
            break
    
    print(f"\n3. CONVERGENCE SPEED (epochs to reach <10% parameter error):")
    print(f"   MLP-PINN:            {mlp_epochs_to_10pct if mlp_epochs_to_10pct else 'Did not converge'}")
    print(f"   KAN-PINN:            {kan_epochs_to_10pct if kan_epochs_to_10pct else 'Did not converge'}")
    
    if mlp_epochs_to_10pct and kan_epochs_to_10pct:
        speedup = mlp_epochs_to_10pct / kan_epochs_to_10pct
        print(f"   → KAN speedup:       {speedup:.2f}x faster")
    
    # Final losses
    print(f"\n4. FINAL LOSSES:")
    print(f"   MLP-PINN total:      {history_mlp.losses['total'][-1]:.6f}")
    print(f"   KAN-PINN total:      {history_kan.losses['total'][-1]:.6f}")
    print(f"   MLP-PINN physics:    {history_mlp.losses['physics'][-1]:.6f}")
    print(f"   KAN-PINN physics:    {history_kan.losses['physics'][-1]:.6f}")
    
    print("\n" + "=" * 80)


# =============================================================================
# PART 6: MAIN EXPERIMENT
# =============================================================================

def main():
    """
    Main experiment: Compare MLP-PINN vs KAN-PINN on Van der Pol inverse problem.
    """
    print("=" * 80)
    print("VAN DER POL INVERSE PROBLEM: MLP-PINN vs KAN-PINN")
    print("=" * 80)
    
    # -------------------------------------------------------------------------
    # Step 1: Generate ground truth data
    # -------------------------------------------------------------------------
    mu_true = 100.0  # True parameter value (moderately stiff)
    t_span = (0, 20)
    n_points = 1000
    
    t_true, x_true = generate_ground_truth_data(mu_true, t_span, n_points)
    
    # -------------------------------------------------------------------------
    # Step 2: Create sparse, noisy training data
    # -------------------------------------------------------------------------
    n_observations = 20
    noise_level = 0.05
    
    t_obs, x_obs = create_training_data(t_true, x_true, n_observations, noise_level)
    
    # -------------------------------------------------------------------------
    # Step 3: Create collocation points for physics loss
    # -------------------------------------------------------------------------
    n_collocation = 100
    t_physics = np.linspace(t_span[0], t_span[1], n_collocation)
    
    print(f"\nCollocation points for physics loss: {n_collocation}")
    
    # -------------------------------------------------------------------------
    # Step 4: Initialize models
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("INITIALIZING MODELS")
    print("=" * 80)
    
    # MLP-PINN with larger network
    model_mlp = MLP_PINN(hidden_size=64, n_layers=3)
    
    # KAN-PINN with smaller network (but more expressive)
    model_kan = KAN_PINN(hidden_size=32, n_layers=3, grid_size=5)
    
    # -------------------------------------------------------------------------
    # Step 5: Train MLP-PINN
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("TRAINING MLP-PINN")
    print("=" * 80)
    
    history_mlp = train_pinn(
        model_mlp,
        t_physics,
        t_obs,
        x_obs,
        mu_true,
        n_epochs=5000,
        lr=1e-3,
        print_every=500
    )
    
    # -------------------------------------------------------------------------
    # Step 6: Train KAN-PINN
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("TRAINING KAN-PINN")
    print("=" * 80)
    
    history_kan = train_pinn(
        model_kan,
        t_physics,
        t_obs,
        x_obs,
        mu_true,
        n_epochs=5000,
        lr=1e-3,
        print_every=500
    )
    
    # -------------------------------------------------------------------------
    # Step 7: Visualize results
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)
    
    plot_training_comparison(history_mlp, history_kan, mu_true, 
                            'training_comparison.png')
    
    plot_solution_comparison(t_true, x_true, t_obs, x_obs, 
                            model_mlp, model_kan,
                            'solution_comparison.png')
    
    visualize_kan_activation_functions(model_kan, 'kan_activations.png')
    
    # -------------------------------------------------------------------------
    # Step 8: Print summary
    # -------------------------------------------------------------------------
    print_performance_summary(history_mlp, history_kan, mu_true)
    
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print("\nGenerated files:")
    print("  1. training_comparison.png  - Training curves and parameter evolution")
    print("  2. solution_comparison.png  - Solution predictions and errors")
    print("  3. kan_activations.png      - Learned KAN activation functions")


if __name__ == "__main__":
    main()