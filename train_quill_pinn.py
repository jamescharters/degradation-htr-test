#!/usr/bin/env python3
"""
Physics-Informed Neural Network for Quill Writing Reconstruction
Learns to reconstruct temporal writing process from static final image
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import json
import matplotlib.pyplot as plt
from pathlib import Path

print("=" * 60)
print("QUILL PHYSICS PINN TRAINING")
print("=" * 60)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"✓ Using device: {device}\n")

# ============================================
# PINN ARCHITECTURE
# ============================================

class QuillPINN(nn.Module):
    """
    Physics-Informed Neural Network for quill writing
    
    Input: (x, y, t) - spatial coordinates and time
    Output: h - ink height at that spatiotemporal point
    """
    def __init__(self, hidden_dim=128, num_layers=8):
        super().__init__()
        
        # Input: (x, y, t) normalized to [0, 1]
        layers = [nn.Linear(3, hidden_dim), nn.Tanh()]
        
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh()
            ])
        
        layers.append(nn.Linear(hidden_dim, 1))  # Output: h(x,y,t)
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights (Xavier)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x, y, t):
        """
        Args:
            x, y: spatial coordinates [0, 1]
            t: time [0, 1]
        Returns:
            h: predicted ink height
        """
        inputs = torch.stack([x, y, t], dim=-1)
        h = self.network(inputs)
        return h.squeeze(-1)

# ============================================
# PHYSICS EQUATIONS
# ============================================

def compute_gradients(output, input_tensor, order=1):
    """
    Compute gradients of output w.r.t. input
    Supports arbitrary order derivatives
    """
    grad = output
    for _ in range(order):
        grad = torch.autograd.grad(
            grad.sum(), 
            input_tensor,
            create_graph=True,
            retain_graph=True
        )[0]
    return grad

def physics_loss(model, x, y, t, 
                viscosity=0.01, 
                evaporation_rate=0.001,
                capillary_coeff=0.1):
    """
    Compute physics-based loss enforcing thin film fluid dynamics
    
    PDE: ∂h/∂t = -∇·(h³∇p) - k·h
    
    Simplified: ∂h/∂t = c·∇²h - k·h
    (Laplacian approximation of thin film equation)
    """
    # Ensure gradients are tracked
    x = x.requires_grad_(True)
    y = y.requires_grad_(True)
    t = t.requires_grad_(True)
    
    # Predict ink height
    h = model(x, y, t)
    
    # Compute derivatives
    h_t = compute_gradients(h, t, order=1)
    h_x = compute_gradients(h, x, order=1)
    h_y = compute_gradients(h, y, order=1)
    h_xx = compute_gradients(h_x, x, order=1)
    h_yy = compute_gradients(h_y, y, order=1)
    
    # Laplacian (diffusion term)
    laplacian = h_xx + h_yy
    
    # PDE residual: ∂h/∂t - c·∇²h + k·h = 0
    diffusion_term = capillary_coeff * laplacian
    evaporation_term = -evaporation_rate * h
    
    pde_residual = h_t - diffusion_term - evaporation_term
    
    # Mean squared residual
    return (pde_residual ** 2).mean()

# ============================================
# DATA LOADING
# ============================================

class QuillDataset:
    """
    Load synthetic letter data for training
    """
    def __init__(self, letter_path, metadata_path):
        # Load final image
        self.image = np.array(Image.open(letter_path).convert('L'))
        self.image = self.image.astype(np.float32) / 255.0
        self.image = 1.0 - self.image  # Invert: high values = ink
        
        self.height, self.width = self.image.shape
        
        # Load metadata (stroke timing, etc.)
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        self.total_time = self.metadata['total_time']
        
        # Normalize coordinates to [0, 1]
        self.x_coords = np.linspace(0, 1, self.width)
        self.y_coords = np.linspace(0, 1, self.height)
        
        print(f"✓ Loaded letter: {self.metadata['letter']}")
        print(f"  Image size: {self.width}x{self.height}")
        print(f"  Total strokes: {len(self.metadata['strokes'])}")
        print(f"  Total time: {self.total_time:.2f}s")
    
    def sample_points(self, n_points):
        """
        Sample random spatiotemporal points for training
        """
        # Random spatial points
        x_idx = np.random.randint(0, self.width, n_points)
        y_idx = np.random.randint(0, self.height, n_points)
        
        # Random time points
        t = np.random.uniform(0, 1, n_points)
        
        # Normalize coordinates
        x = x_idx / self.width
        y = y_idx / self.height
        
        return x, y, t
    
    def get_final_image_points(self, n_points):
        """
        Sample points from final image (t=1) for boundary condition
        """
        x_idx = np.random.randint(0, self.width, n_points)
        y_idx = np.random.randint(0, self.height, n_points)
        
        x = x_idx / self.width
        y = y_idx / self.height
        h_true = self.image[y_idx, x_idx]
        
        return x, y, h_true
    
    def get_initial_condition_points(self, n_points):
        """
        Initial condition: h(x, y, 0) = 0 (no ink at t=0)
        """
        x_idx = np.random.randint(0, self.width, n_points)
        y_idx = np.random.randint(0, self.height, n_points)
        
        x = x_idx / self.width
        y = y_idx / self.height
        
        return x, y

# ============================================
# TRAINING LOOP
# ============================================

def train_pinn(model, dataset, epochs=5000, 
               n_physics=1000, n_boundary=500, n_initial=500,
               lr=1e-3):
    """
    Train PINN with physics loss + boundary conditions
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Loss weights
    lambda_physics = 1.0
    lambda_boundary = 10.0
    lambda_initial = 5.0
    
    loss_history = {
        'total': [],
        'physics': [],
        'boundary': [],
        'initial': []
    }
    
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # ========================================
        # Physics Loss (PDE in interior)
        # ========================================
        x_phys, y_phys, t_phys = dataset.sample_points(n_physics)
        x_phys = torch.tensor(x_phys, dtype=torch.float32, device=device)
        y_phys = torch.tensor(y_phys, dtype=torch.float32, device=device)
        t_phys = torch.tensor(t_phys, dtype=torch.float32, device=device)
        
        loss_phys = physics_loss(model, x_phys, y_phys, t_phys)
        
        # ========================================
        # Boundary Condition Loss (final image at t=1)
        # ========================================
        x_bc, y_bc, h_true = dataset.get_final_image_points(n_boundary)
        x_bc = torch.tensor(x_bc, dtype=torch.float32, device=device)
        y_bc = torch.tensor(y_bc, dtype=torch.float32, device=device)
        t_bc = torch.ones_like(x_bc, device=device)  # t=1 (final time)
        h_true = torch.tensor(h_true, dtype=torch.float32, device=device)
        
        h_pred = model(x_bc, y_bc, t_bc)
        loss_bc = ((h_pred - h_true) ** 2).mean()
        
        # ========================================
        # Initial Condition Loss (h=0 at t=0)
        # ========================================
        x_ic, y_ic = dataset.get_initial_condition_points(n_initial)
        x_ic = torch.tensor(x_ic, dtype=torch.float32, device=device)
        y_ic = torch.tensor(y_ic, dtype=torch.float32, device=device)
        t_ic = torch.zeros_like(x_ic, device=device)  # t=0
        
        h_pred_ic = model(x_ic, y_ic, t_ic)
        loss_ic = (h_pred_ic ** 2).mean()
        
        # ========================================
        # Total Loss
        # ========================================
        loss = (lambda_physics * loss_phys + 
                lambda_boundary * loss_bc + 
                lambda_initial * loss_ic)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Record losses
        loss_history['total'].append(loss.item())
        loss_history['physics'].append(loss_phys.item())
        loss_history['boundary'].append(loss_bc.item())
        loss_history['initial'].append(loss_ic.item())
        
        # Print progress
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Total Loss:    {loss.item():.6f}")
            print(f"  Physics Loss:  {loss_phys.item():.6f}")
            print(f"  Boundary Loss: {loss_bc.item():.6f}")
            print(f"  Initial Loss:  {loss_ic.item():.6f}")
    
    return loss_history

# ============================================
# VISUALIZATION & EVALUATION
# ============================================

def visualize_reconstruction(model, dataset, output_path='reconstruction.png'):
    """
    Visualize reconstruction at different time points
    """
    model.eval()
    
    # Time points to visualize
    time_points = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    fig, axes = plt.subplots(2, len(time_points), figsize=(15, 6))
    
    with torch.no_grad():
        for i, t_val in enumerate(time_points):
            # Create grid of spatial points
            x_grid = np.linspace(0, 1, dataset.width)
            y_grid = np.linspace(0, 1, dataset.height)
            X, Y = np.meshgrid(x_grid, y_grid)
            
            x_flat = torch.tensor(X.flatten(), dtype=torch.float32, device=device)
            y_flat = torch.tensor(Y.flatten(), dtype=torch.float32, device=device)
            t_flat = torch.ones_like(x_flat) * t_val
            
            # Predict
            h_pred = model(x_flat, y_flat, t_flat)
            h_pred = h_pred.cpu().numpy().reshape(dataset.height, dataset.width)
            
            # Invert for display (dark = ink)
            h_display = 1.0 - h_pred
            h_display = np.clip(h_display, 0, 1)
            
            # Plot
            axes[0, i].imshow(h_display, cmap='gray', vmin=0, vmax=1)
            axes[0, i].set_title(f't={t_val:.2f}')
            axes[0, i].axis('off')
            
            # Plot ink height (colormap)
            im = axes[1, i].imshow(h_pred, cmap='hot', vmin=0, vmax=h_pred.max())
            axes[1, i].set_title(f'Ink height')
            axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved reconstruction: {output_path}")
    plt.close()

def plot_loss_curves(loss_history, output_path='loss_curves.png'):
    """
    Plot training loss curves
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].plot(loss_history['total'])
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_yscale('log')
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(loss_history['physics'])
    axes[0, 1].set_title('Physics Loss (PDE Residual)')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_yscale('log')
    axes[0, 1].grid(True)
    
    axes[1, 0].plot(loss_history['boundary'])
    axes[1, 0].set_title('Boundary Loss (Final Image)')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True)
    
    axes[1, 1].plot(loss_history['initial'])
    axes[1, 1].set_title('Initial Condition Loss')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved loss curves: {output_path}")
    plt.close()

# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    import sys
    
    # Check if synthetic data exists
    if not Path("letter_A.png").exists():
        print("✗ No synthetic data found!")
        print("Run quill_simulator.py first to generate training data")
        sys.exit(1)
    
    # Load dataset
    print("\n[1] Loading synthetic letter data...")
    dataset = QuillDataset(
        "synthetic_letters/CANONE_letter_0_C.png",
        "synthetic_letters/CANONE_letter_0_C_metadata.json"
    )
    
    # Create model
    print("\n[2] Creating PINN model...")
    model = QuillPINN(hidden_dim=128, num_layers=8).to(device)
    print(f"✓ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    print("\n[3] Training PINN...")
    loss_history = train_pinn(
        model, 
        dataset,
        epochs=2000,
        n_physics=1000,
        n_boundary=500,
        n_initial=500,
        lr=1e-3
    )
    
    # Visualize
    print("\n[4] Generating visualizations...")
    visualize_reconstruction(model, dataset, 'pinn_reconstruction.png')
    plot_loss_curves(loss_history, 'pinn_loss_curves.png')
    
    # Save model
    print("\n[5] Saving model...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'loss_history': loss_history
    }, 'quill_pinn_model.pt')
    print("✓ Saved: quill_pinn_model.pt")
    
    print("\n" + "=" * 60)
    print("✓ TRAINING COMPLETE!")
    print("\nGenerated files:")
    print("  - pinn_reconstruction.png (temporal evolution)")
    print("  - pinn_loss_curves.png (training progress)")
    print("  - quill_pinn_model.pt (trained model)")
    print("=" * 60)
